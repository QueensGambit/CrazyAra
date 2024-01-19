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
    print_model_summary, export_best_model_state, fill_train_config, export_configs, create_export_dirs, export_cmd_args
from DeepCrazyhouse.src.training.trainer_agent_pytorch import TrainerAgentPytorch


def parse_args(train_config: TrainConfig):
    """Defines the command-line arguments and parses them."""
    parser = argparse.ArgumentParser(description="Training script for training CNNs or Transformer networks."
                                                 "For all remaining configuration options, please refer to:"
                                                 "CrazyAra/configs/train_config.py")

    info_prefix, info_str_dict = extract_info_strings(train_config)
    fill_parser(info_prefix, info_str_dict, parser, train_config)

    return parser.parse_args()


def fill_parser(info_prefix: str, info_str_dict: dict, parser, train_config: TrainConfig):
    """Fills the parser object with member and help information."""
    for member in vars(train_config):
        member_str = member.replace("_", "-")
        if not member_str.startswith(info_prefix):
            default_value = vars(train_config)[member]
            if member_str in info_str_dict:
                help_str = info_str_dict[member_str]
            else:
                help_str = "?"
            parser.add_argument(f"--{member_str}", type=type(default_value),
                                help=f"{help_str} (default: {default_value})",
                                default=default_value)


def extract_info_strings(train_config: TrainConfig):
    """Extract info strings from the train_config.py file amd returns a dictionary object."""
    info_prefix = "info-"
    info_str_dict = {}
    for member in vars(train_config):
        member_str = member.replace("_", "-")
        default_value = vars(train_config)[member]
        if member_str.startswith(info_prefix):
            info_str_dict[member_str.replace(info_prefix, "")] = default_value
    return info_prefix, info_str_dict


def update_train_config_via_args(args, train_config):
    """Update the config parameters with values provided through command-line arguments"""
    for arg, value in vars(args).items():
        if value is not None:
            setattr(train_config, arg, value)


def main():
    # Create an instance of the TrainConfig class
    train_config = TrainConfig()
    args = parse_args(train_config)
    enable_color_logging()

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
    export_cmd_args(train_config)

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
