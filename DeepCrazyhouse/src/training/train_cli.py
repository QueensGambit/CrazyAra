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

sys.path.insert(0, '../../../')

from DeepCrazyhouse.src.runtime.color_logger import enable_color_logging
from DeepCrazyhouse.configs.train_config import TrainConfig, TrainObjects
from DeepCrazyhouse.src.training.train_cli_util import create_pytorch_model, create_validation_data, fill_train_objects, \
    export_best_model_state
from DeepCrazyhouse.src.training.trainer_agent_pytorch import TrainerAgentPytorch


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for training CNNs or Transformer networks."
                                                 "For all remaining configuration options, please refer to:"
                                                 "CrazyAra/configs/train_config.py")
    parser.add_argument("--model-type", type=str,
                        help="Type of neural network architecture (resnet, vit, risev2, risev3))",
                        default="resnet")

    # Add other command-line arguments corresponding to TrainConfig parameters
    parser.add_argument("--name-initials", type=str, help="Name initials which are used to identify running training "
                                                          "processes with the rtpt library", default="XX")
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

    # Create an instance of the TrainConfig class
    train_config = TrainConfig()

    update_train_config_via_args(args, train_config)

    enable_color_logging()

    val_data, x_val = create_validation_data(train_config)
    input_shape = x_val[0].shape

    model = create_pytorch_model(args.model_type, input_shape, train_config)

    train_objects = TrainObjects()
    fill_train_objects(train_config, train_objects)

    train_agent = TrainerAgentPytorch(model, val_data, train_config, train_objects, use_rtpt=True)

    # iteration counter used for the momentum and learning rate schedule
    cur_it = train_config.k_steps_initial * train_config.batch_steps
    (k_steps_final, value_loss_final, policy_loss_final, value_acc_sign_final, val_p_acc_final), (
        k_steps_best, val_metric_values_best) = train_agent.train(cur_it)

    export_best_model_state(k_steps_best, k_steps_final, model, policy_loss_final, train_config, val_metric_values_best,
                            val_p_acc_final)


if __name__ == "__main__":
    main()
