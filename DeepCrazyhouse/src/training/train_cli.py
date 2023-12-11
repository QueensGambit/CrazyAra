"""
@file: train_cli.py
Created on 10.11.23
@project: CrazyAra
@author: queensgambit

Provides a command-line interface to run training experiments.
For the main configuration files please modify
* CrazyAra/DeepCrazyhouse/configs/main_config.py
* CrazyAra/DeepCrazyhouse/configs/train_config.py

A decision was made to only support the Pytorch framework.
"""

import argparse
import sys
import torch
import logging

sys.path.insert(0, '../../../')

from DeepCrazyhouse.src.runtime.color_logger import enable_color_logging
from DeepCrazyhouse.configs.train_config import TrainConfig, TrainObjects
from DeepCrazyhouse.src.training.train_cli_util import create_pytorch_model, get_validation_data, fill_train_objects,\
    print_model_summary, export_best_model_state, fill_train_config, export_configs, create_export_dirs
from DeepCrazyhouse.src.training.trainer_agent_pytorch import TrainerAgentPytorch


def parse_args():
    """Defines the command-line arguments and parses them."""
    parser = argparse.ArgumentParser(description="Training script for training CNNs or Transformer networks."
                                                 "For all remaining configuration options, please refer to:"
                                                 "CrazyAra/configs/train_config.py")
    parser.add_argument("--model-type", type=str,
                        help="Type of neural network architecture (resnet, vit, risev2, risev3, alphavile,"
                             "alphavile-tiny, alphavile-small, alphavile-normal, alphavile-large, NextViT)",
                        default="resnet")
    parser.add_argument("--name-initials", type=str, help="Name initials which are used to identify running training "
                                                          "processes with the rtpt library", default="XX")
    parser.add_argument("--export-dir", type=str, help="Export directory where the model files and tensorboard logs"
                                                       "will be saved.", default="./")
    parser.add_argument("--use-custom-architecture", type=bool, help="Decides if a custom network architecture should be"
                                                                     "used, defined in the model_config.py file.", default=False)

    return parser.parse_args()


def update_train_config_via_args(args, train_config):
    """Update the config parameters with values provided through command-line arguments"""
    for arg, value in vars(args).items():
        if value is not None:
            setattr(train_config, arg, value)


def main():
    args = parse_args()
    enable_color_logging()

    # Create an instance of the TrainConfig class
    train_config = TrainConfig()
    update_train_config_via_args(args, train_config)

    val_data, x_val, _ = get_validation_data(train_config)
    input_shape = x_val[0].shape
    fill_train_config(train_config, x_val)

    model = create_pytorch_model(input_shape, train_config)
    print_model_summary(input_shape, model, x_val)
    if torch.cuda.is_available():
        model.cuda(torch.device(f"cuda:{train_config.device_id}"))

    train_objects = TrainObjects()
    fill_train_objects(train_config, train_objects)

    create_export_dirs(train_config)
    export_configs(args, train_config)

    train_agent = TrainerAgentPytorch(model, val_data, train_config, train_objects, use_rtpt=True)

    # iteration counter used for the momentum and learning rate schedule
    cur_it = train_config.k_steps_initial * train_config.batch_steps
    (k_steps_final, value_loss_final, policy_loss_final, value_acc_sign_final, val_p_acc_final), (
        k_steps_best, val_metric_values_best) = train_agent.train(cur_it)

    val_loss_best = val_metric_values_best["loss"]
    val_p_acc_best = val_metric_values_best["policy_acc"]
    logging.info('best val_loss: %.5f with v_policy_acc: %.5f at k_steps_best %d' % (val_loss_best, val_p_acc_best, k_steps_best))

    export_best_model_state(k_steps_best, k_steps_final, model, policy_loss_final, input_shape, train_config,
                            val_metric_values_best, val_p_acc_final)


if __name__ == "__main__":
    main()
