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

sys.path.append("../../../")
from DeepCrazyhouse.configs.train_config import TrainConfig, TrainObjects
from DeepCrazyhouse.src.preprocessing.dataset_loader import load_pgn_dataset
from DeepCrazyhouse.src.training.trainer_agent import acc_sign, cross_entropy, acc_distribution
from DeepCrazyhouse.src.training.trainer_agent_mxnet import prepare_policy
from DeepCrazyhouse.src.training.trainer_agent import TrainerAgent
from DeepCrazyhouse.src.training.lr_schedules.lr_schedules import MomentumSchedule, LinearWarmUp,\
    CosineAnnealingSchedule
from DeepCrazyhouse.src.domain.neural_net.onnx.convert_to_onnx import convert_mxnet_model_to_onnx


def update_network(queue, nn_update_idx, symbol_filename, params_filename, convert_to_onnx, main_config, train_config: TrainConfig, model_contender_dir):
    """
    Creates a new NN checkpoint in the model contender directory after training using the game files stored in the
     training directory
    :param queue: Queue object used to return items
    :param nn_update_idx: Defines how many updates of the nn has already been done. This index should be incremented
    after every update.
    :param symbol_filename: Architecture definition file
    :param params_filename: Weight file which will be loaded before training
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

    _, x_val, y_val_value, y_val_policy, _, _ = load_pgn_dataset(dataset_type="val",
                                                                 part_id=0,
                                                                 normalize=train_config.normalize,
                                                                 verbose=False,
                                                                 q_value_ratio=train_config.q_value_ratio)
    y_val_policy = prepare_policy(y_val_policy, train_config.select_policy_from_plane,
                                  train_config.sparse_policy_label, train_config.is_policy_from_plane_data)
    val_dataset = gluon.data.ArrayDataset(nd.array(x_val), nd.array(y_val_value), nd.array(y_val_policy))
    val_data = gluon.data.DataLoader(val_dataset, train_config.batch_size, shuffle=False, num_workers=train_config.cpu_count)

    symbol = mx.sym.load(symbol_filename)

    # calculate how many iterations per epoch exist
    nb_it_per_epoch = (len(x_val) * train_config.nb_parts) // train_config.batch_size
    # one iteration is defined by passing 1 batch and doing backprop
    train_config.total_it = int(nb_it_per_epoch * train_config.nb_training_epochs)

    train_objects.lr_schedule = CosineAnnealingSchedule(train_config.min_lr, train_config.max_lr, max(train_config.total_it * .7, 1))
    train_objects.lr_schedule = LinearWarmUp(train_objects.lr_schedule, start_lr=train_config.min_lr, length=max(train_config.total_it * .25, 1))
    train_objects.momentum_schedule = MomentumSchedule(train_objects.lr_schedule, train_config.min_lr, train_config.max_lr,
                                         train_config.min_momentum, train_config.max_momentum)

    input_shape = x_val[0].shape
    inputs = mx.sym.var('data', dtype='float32')
    value_out = symbol.get_internals()[main_config['value_output'] + '_output']
    policy_out = symbol.get_internals()[main_config['policy_output'] + '_output']
    sym = mx.symbol.Group([value_out, policy_out])
    net = mx.gluon.SymbolBlock(sym, inputs)
    net.collect_params().load(params_filename, ctx)

    metrics_gluon = {
        'value_loss': metric.MSE(name='value_loss', output_names=['value_output']),

        'value_acc_sign': metric.create(acc_sign, name='value_acc_sign', output_names=['value_output'],
                                        label_names=['value_label']),
    }

    if train_config.sparse_policy_label:
        print("train with sparse labels")
        # the default cross entropy only supports sparse labels
        metrics_gluon['policy_loss'] = metric.CrossEntropy(name='policy_loss', output_names=['policy_output'],
                                           label_names=['policy_label']),
        metrics_gluon['policy_acc'] = metric.Accuracy(axis=1, name='policy_acc', output_names=['policy_output'],
                                      label_names=['policy_label'])
    else:
        metrics_gluon['policy_loss'] = metric.create(cross_entropy, name='policy_loss', output_names=['policy_output'],
                                        label_names=['policy_label'])
        metrics_gluon['policy_acc'] = metric.create(acc_distribution, name='policy_acc', output_names=['policy_output'],
                       label_names=['policy_label'])

    train_objects.metrics = metrics_gluon

    train_config.export_weights = False  # don't save intermediate weights
    train_agent = TrainerAgent(net, val_data, train_config, train_objects, use_rtpt=False)

    # iteration counter used for the momentum and learning rate schedule
    cur_it = train_config.k_steps_initial * train_config.batch_steps
    (k_steps_final, val_value_loss_final, val_policy_loss_final, val_value_acc_sign_final,
     val_policy_acc_final), (_, _) = train_agent.train(cur_it)

    prefix = "%smodel-%.5f-%.5f-%.3f-%.3f" % (model_contender_dir, val_value_loss_final, val_policy_loss_final,
                                                                  val_value_acc_sign_final, val_policy_acc_final)

    sym_file = prefix + "-symbol.json"
    params_file = prefix + "-" + "%04d.params" % nn_update_idx

    # the export function saves both the architecture and the weights
    net.export(prefix, epoch=nn_update_idx)
    print()
    logging.info("Saved checkpoint to %s-%04d.params", prefix, nn_update_idx)

    if convert_to_onnx:
        convert_mxnet_model_to_onnx(sym_file, params_file, ["value_out_output", "policy_out_output"], input_shape,
                                    [1, 8, 16], False)

    logging.info("k_steps_final %d" % k_steps_final)
    queue.put(k_steps_final)
