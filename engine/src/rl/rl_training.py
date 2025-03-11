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
from pathlib import Path
import torch

sys.path.append("../../../")
from DeepCrazyhouse.configs.train_config import TrainConfig, TrainObjects
from DeepCrazyhouse.src.training.lr_schedules.lr_schedules import MomentumSchedule, LinearWarmUp,\
    CosineAnnealingSchedule
from DeepCrazyhouse.src.training.train_util import get_metrics
from DeepCrazyhouse.src.training.train_cli_util import create_pytorch_model, get_validation_data
from DeepCrazyhouse.src.training.trainer_agent_pytorch import TrainerAgentPytorch, load_torch_state, save_torch_state,\
    get_context, export_to_onnx


def update_network(queue, nn_update_idx: int, tar_filename: Path, convert_to_onnx: bool, main_config,
                   train_config: TrainConfig, model_contender_dir: Path):
    """
    Creates a new NN checkpoint in the model contender directory after training using the game files stored in the
     training directory
    :param queue: Queue object used to return items
    :param nn_update_idx: Defines how many updates of the nn has already been done. This index should be incremented
    after every update.
    :param tar_filename: Filepath to the model for pytorch
    Updates the neural network with the newly acquired games from the replay memory
    :param convert_to_onnx: Boolean indicating if the network shall be exported to ONNX to allow TensorRT inference
    :param main_config: Dict of the main_config (imported from main_config.py)
    :param train_config: Dict of the train_config (imported from train_config.py)
    :param model_contender_dir: String of the contender directory path
    :return: k_steps_final
    """

    # set a specific seed value for reproducibility
    train_config.nb_parts = len(glob.glob(main_config["planes_train_dir"] + '**/*.zip'))
    logging.info("number parts for training: %d" % train_config.nb_parts)
    train_objects = TrainObjects()

    if train_config.nb_parts <= 0:
        raise Exception('No .zip files for training available. Check the path in main_config["planes_train_dir"]:'
                        ' %s' % main_config["planes_train_dir"])

    val_data, x_val = get_validation_data(train_config)

    input_shape = x_val[0].shape
    # calculate how many iterations per epoch exist
    nb_it_per_epoch = (len(x_val) * train_config.nb_parts) // train_config.batch_size
    # one iteration is defined by passing 1 batch and doing backprop
    train_config.total_it = int(nb_it_per_epoch * train_config.nb_training_epochs)

    train_objects.lr_schedule = CosineAnnealingSchedule(train_config.min_lr, train_config.max_lr, max(train_config.total_it * .7, 1))
    train_objects.lr_schedule = LinearWarmUp(train_objects.lr_schedule, start_lr=train_config.min_lr, length=max(train_config.total_it * .25, 1))
    train_objects.momentum_schedule = MomentumSchedule(train_objects.lr_schedule, train_config.min_lr, train_config.max_lr,
                                         train_config.min_momentum, train_config.max_momentum)

    logging.info(f"Load checkpoint {tar_filename}")
    net = _get_net(input_shape, tar_filename, train_config)
    train_objects.metrics = get_metrics(train_config)

    train_config.export_weights = True  # save intermediate results to handle spikes
    train_agent = TrainerAgentPytorch(net, val_data, train_config, train_objects, use_rtpt=False)

    # iteration counter used for the momentum and learning rate schedule
    cur_it = train_config.k_steps_initial * train_config.batch_steps
    (k_steps_final, val_value_loss_final, val_policy_loss_final, val_value_acc_sign_final,
     val_policy_acc_final), (_, _) = train_agent.train(cur_it)
    prefix = "%smodel-%.5f-%.5f-%.3f-%.3f" % (model_contender_dir, val_value_loss_final, val_policy_loss_final,
                                                                   val_value_acc_sign_final, val_policy_acc_final)

    _export_net(convert_to_onnx, input_shape, k_steps_final, net, nn_update_idx, prefix, train_config, model_contender_dir)

    logging.info("k_steps_final %d" % k_steps_final)
    queue.put(k_steps_final)


def _export_net(convert_to_onnx: bool, input_shape: tuple, k_steps_final: int, net, nn_update_idx: int, prefix: str,
                train_config: TrainConfig, model_contender_dir: Path):
    """
    Export function saves both the architecture and the weights and optionally saves it as onnx
    """
    net.eval()
    save_torch_state(net, torch.optim.SGD(net.parameters(), lr=train_config.max_lr),
                     Path('%s-%04d.tar' % (prefix, k_steps_final)))
    print()
    logging.info("Saved checkpoint to %s-%04d.params", prefix, nn_update_idx)
    if convert_to_onnx:
        model_prefix = "%s-%04d" % (prefix, k_steps_final)
        with torch.no_grad():
            ctx = get_context(train_config.context, train_config.device_id)
            dummy_input = torch.zeros(1, input_shape[0], input_shape[1], input_shape[2]).to(ctx)
            export_to_onnx(net, 1,
                           dummy_input,
                           Path(model_contender_dir), model_prefix,
                           train_config.use_wdl and train_config.use_plys_to_end,
                           True)


def _get_net(input_shape: tuple, tar_filename: Path, train_config: TrainConfig):
    """
    Loads the network object and weights.
    """
    net = create_pytorch_model(input_shape, train_config)
    if torch.cuda.is_available():
        net.cuda(torch.device(f"cuda:{train_config.device_id}"))
    load_torch_state(net, torch.optim.SGD(net.parameters(), lr=train_config.max_lr), tar_filename, train_config.device_id)
    return net
