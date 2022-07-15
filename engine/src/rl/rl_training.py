"""
@file: rl_training.py
Created on 20.11.19
@project: CrazyAra
@author: queensgambit

Functionality for conducting a single NN update within the reinforcement learning loop
"""

import sys
import glob
import logging
import mxnet as mx
try:
    import mxnet.metric as metric
except ModuleNotFoundError:
    import mxnet.gluon.metric as metric
from mxnet import nd
from mxnet import gluon
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

sys.path.append("../../../")
from DeepCrazyhouse.configs.train_config import TrainConfig, TrainObjects
from DeepCrazyhouse.src.preprocessing.dataset_loader import load_pgn_dataset
from DeepCrazyhouse.src.training.trainer_agent_mxnet import prepare_policy, prepare_plys_label, value_to_wdl_label
from DeepCrazyhouse.src.training.lr_schedules.lr_schedules import MomentumSchedule, LinearWarmUp,\
    CosineAnnealingSchedule
from DeepCrazyhouse.src.domain.neural_net.onnx.convert_to_onnx import convert_mxnet_model_to_onnx
from DeepCrazyhouse.src.training.train_util import get_metrics


def update_network(queue, nn_update_idx, symbol_filename, params_filename, tar_filename, convert_to_onnx, main_config, train_config: TrainConfig, model_contender_dir):
    """
    Creates a new NN checkpoint in the model contender directory after training using the game files stored in the
     training directory
    :param queue: Queue object used to return items
    :param nn_update_idx: Defines how many updates of the nn has already been done. This index should be incremented
    after every update.
    :param symbol_filename: Architecture definition file
    :param params_filename: Weight file which will be loaded before training
    :param tar_filename: Filepath to the model for pytorch
    Updates the neural network with the newly acquired games from the replay memory
    :param convert_to_onnx: Boolean indicating if the network shall be exported to ONNX to allow TensorRT inference
    :param main_config: Dict of the main_config (imported from main_config.py)
    :param train_config: Dict of the train_config (imported from train_config.py)
    :param model_contender_dir: String of the contender directory path
    :return: k_steps_final
    """

    # set the context on CPU, switch to GPU if there is one available (strongly recommended for training)
    ctx = mx.gpu(train_config.device_id) if train_config.context == "gpu" else mx.cpu()
    # set a specific seed value for reproducibility
    train_config.nb_parts = len(glob.glob(main_config["planes_train_dir"] + '**/*.zip'))
    logging.info("number parts for training: %d" % train_config.nb_parts)
    train_objects = TrainObjects()

    if train_config.nb_parts <= 0:
        raise Exception('No .zip files for training available. Check the path in main_config["planes_train_dir"]:'
                        ' %s' % main_config["planes_train_dir"])

    val_data, x_val = _get_val_loader(train_config)

    input_shape = x_val[0].shape
    # calculate how many iterations per epoch exist
    nb_it_per_epoch = (len(x_val) * train_config.nb_parts) // train_config.batch_size
    # one iteration is defined by passing 1 batch and doing backprop
    train_config.total_it = int(nb_it_per_epoch * train_config.nb_training_epochs)

    train_objects.lr_schedule = CosineAnnealingSchedule(train_config.min_lr, train_config.max_lr, max(train_config.total_it * .7, 1))
    train_objects.lr_schedule = LinearWarmUp(train_objects.lr_schedule, start_lr=train_config.min_lr, length=max(train_config.total_it * .25, 1))
    train_objects.momentum_schedule = MomentumSchedule(train_objects.lr_schedule, train_config.min_lr, train_config.max_lr,
                                         train_config.min_momentum, train_config.max_momentum)

    net = _get_net(ctx, input_shape, main_config, params_filename, symbol_filename, tar_filename, train_config)

    train_objects.metrics = get_metrics(train_config)

    train_config.export_weights = True  # save intermediate results to handle spikes
    if train_config.framework == 'gluon':
        from DeepCrazyhouse.src.training.trainer_agent_gluon import TrainerAgentGluon
        train_agent = TrainerAgentGluon(net, val_data, train_config, train_objects, use_rtpt=False)
    elif train_config.framework == 'pytorch':
        from DeepCrazyhouse.src.training.trainer_agent_pytorch import TrainerAgentPytorch, load_torch_state, \
            save_torch_state, get_context, export_to_onnx
        train_agent = TrainerAgentPytorch(net, val_data, train_config, train_objects, use_rtpt=False)

    # iteration counter used for the momentum and learning rate schedule
    cur_it = train_config.k_steps_initial * train_config.batch_steps
    (k_steps_final, val_value_loss_final, val_policy_loss_final, val_value_acc_sign_final,
     val_policy_acc_final), (_, _) = train_agent.train(cur_it)
    prefix = "%smodel-%.5f-%.5f-%.3f-%.3f" % (model_contender_dir, val_value_loss_final, val_policy_loss_final,
                                                                   val_value_acc_sign_final, val_policy_acc_final)

    sym_file = prefix + "-symbol.json"
    params_file = prefix + "-" + "%04d.params" % nn_update_idx

    _export_net(convert_to_onnx, input_shape, k_steps_final, net, nn_update_idx, params_file, prefix, sym_file,
                train_config, model_contender_dir)

    logging.info("k_steps_final %d" % k_steps_final)
    queue.put(k_steps_final)


def _export_net(convert_to_onnx, input_shape, k_steps_final, net, nn_update_idx, params_file, prefix, sym_file,
                train_config, model_contender_dir):
    """
    Export function saves both the architecture and the weights and optionally saves it as onnx
    """
    if train_config.framework == 'gluon':
        net.export(prefix, epoch=nn_update_idx)
    elif train_config.framework == 'pytorch':
        from DeepCrazyhouse.src.training.trainer_agent_pytorch import save_torch_state, get_context, export_to_onnx
        save_torch_state(net, torch.optim.SGD(net.parameters(), lr=train_config.max_lr),
                         '%s-%04d.tar' % (prefix, k_steps_final))
    print()
    logging.info("Saved checkpoint to %s-%04d.params", prefix, nn_update_idx)
    if convert_to_onnx:
        if train_config.framework == 'gluon':
            convert_mxnet_model_to_onnx(sym_file, params_file, ["value_out_output", "policy_out_output"], input_shape,
                                        [1, 8, 16], False)
        elif train_config.framework == 'pytorch':
            model_prefix = "%s-%04d" % (prefix, k_steps_final)
            with torch.no_grad():
                ctx = get_context(train_config.context, train_config.device_id)
                dummy_input = torch.zeros(1, input_shape[0], input_shape[1], input_shape[2]).to(ctx)
                export_to_onnx(net, 1,
                               dummy_input,
                               Path(model_contender_dir), model_prefix,
                               train_config.use_wdl and train_config.use_plys_to_end,
                               True)
                for batch_size in [1, 8, 16]:
                    dummy_input = torch.zeros(batch_size, input_shape[0], input_shape[1], input_shape[2]).to(ctx)
                    export_to_onnx(net, batch_size,
                                   dummy_input,
                                   Path(model_contender_dir), model_prefix,
                                   train_config.use_wdl and train_config.use_plys_to_end, False)


def _get_net(ctx, input_shape, main_config, params_filename, symbol_filename, tar_filename, train_config):
    """
    Loads the network object and weights.
    """
    if train_config.framework == 'gluon':
        inputs = mx.sym.var('data', dtype='float32')
        symbol = mx.sym.load(symbol_filename)
        value_out = symbol.get_internals()[main_config['value_output'] + '_output']
        policy_out = symbol.get_internals()[main_config['policy_output'] + '_output']
        sym = mx.symbol.Group([value_out, policy_out])
        net = mx.gluon.SymbolBlock(sym, inputs)
        net.collect_params().load(params_filename, ctx)
    elif train_config.framework == 'pytorch':
        from DeepCrazyhouse.src.training.trainer_agent_pytorch import load_torch_state
        from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.rise_mobile_v3 import get_rise_v33_model_by_train_config
        net = get_rise_v33_model_by_train_config(input_shape, train_config)
        if torch.cuda.is_available():
            net.cuda(torch.device(f"cuda:{train_config.device_id}"))
        load_torch_state(net, torch.optim.SGD(net.parameters(), lr=train_config.max_lr), tar_filename, train_config.device_id)
    return net


def _get_val_loader(train_config):
    """
    Returns the validation loader and x-Data object.
    """
    _, x_val, y_val_value, y_val_policy, plys_to_end, _ = load_pgn_dataset(dataset_type="val",
                                                                           part_id=0,
                                                                           normalize=train_config.normalize,
                                                                           verbose=False,
                                                                           q_value_ratio=train_config.q_value_ratio)
    y_val_policy = prepare_policy(y_val_policy, train_config.select_policy_from_plane,
                                  train_config.sparse_policy_label, train_config.is_policy_from_plane_data)
    if train_config.framework == 'gluon':
        if train_config.use_wdl and train_config.use_plys_to_end:
            val_dataset = gluon.data.ArrayDataset(nd.array(x_val), nd.array(y_val_value), nd.array(y_val_policy),
                                                  nd.array(value_to_wdl_label(y_val_value)),
                                                  nd.array(prepare_plys_label(plys_to_end)))
        else:
            val_dataset = gluon.data.ArrayDataset(nd.array(x_val), nd.array(y_val_value), nd.array(y_val_policy))
        val_data = gluon.data.DataLoader(val_dataset, train_config.batch_size, shuffle=False,
                                         num_workers=train_config.cpu_count)
    elif train_config.framework == 'pytorch':
        if train_config.use_wdl and train_config.use_wdl:
            val_dataset = TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val_value),
                                        torch.Tensor(y_val_policy),
                                        torch.Tensor(value_to_wdl_label(y_val_value)),
                                        torch.Tensor(prepare_plys_label(plys_to_end)))
        else:
            val_dataset = TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val_value),
                                        torch.Tensor(y_val_policy))
        val_data = DataLoader(val_dataset, shuffle=True, batch_size=train_config.batch_size,
                              num_workers=train_config.cpu_count)
    return val_data, x_val
