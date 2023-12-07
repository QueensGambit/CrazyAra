"""
@file: train_cli_util.py
Created on 10.11.23
@project: CrazyAra
@author: queensgambit

Provides utility functions for train_cli.py.
"""
import logging
import os
import shutil

from DeepCrazyhouse.src.domain.variants.constants import NB_POLICY_MAP_CHANNELS, NB_LABELS

# architectures
from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.rise_mobile_v3 import RiseV3, \
    get_rise_v2_model, get_rise_v33_model
from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.vision_transformer import VisionTransformer,\
    get_vision_transformer_model
from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.vit_configs import get_b8_config
from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.le_vit import LeViT
from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.mobile_vit import MobileViT
from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.trt_vit import TrtViT
from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.next_vit_official import NextVit
from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.a0_resnet import AlphaZeroResnet, get_alpha_zero_model
from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.alpha_vile import get_alpha_vile_model
from DeepCrazyhouse.configs.train_config import TrainConfig
from DeepCrazyhouse.src.preprocessing.dataset_loader import load_pgn_dataset
from DeepCrazyhouse.src.training.lr_schedules.lr_schedules import plot_schedule, ConstantSchedule, OneCycleSchedule,\
    LinearWarmUp, MomentumSchedule

# Pytorch imports
import torch


from DeepCrazyhouse.src.training.trainer_agent_pytorch import TrainerAgentPytorch, save_torch_state, \
    load_torch_state, export_to_onnx, get_context, get_data_loader, evaluate_metrics


def create_args_by_train_config(input_shape: tuple, tc: TrainConfig):
    class Args:
        pass

    args = Args()
    args.input_shape = input_shape
    args.channels_policy_head = NB_POLICY_MAP_CHANNELS
    args.n_labels = NB_LABELS
    args.select_policy_from_plane = tc.select_policy_from_plane
    args.use_wdl = tc.use_wdl
    args.use_plys_to_end = tc.use_plys_to_end
    args.use_mlp_wdl_ply = tc.use_mlp_wdl_ply
    return args


def create_pytorch_model(model_type: str, input_shape: tuple, train_config: TrainConfig):
    """Implement logic to create a PyTorch model based on model_type."""
    args = create_args_by_train_config(input_shape, train_config)

    if model_type == 'resnet':
        return get_alpha_zero_model(args)
    elif model_type == 'vit':
        return get_vision_transformer_model(args)
    elif model_type == 'risev2':
        return get_rise_v2_model(args)
    elif model_type == 'risev3':
        return get_rise_v33_model(args)
    elif model_type == 'alphavile':
        return get_alpha_vile_model(args)
    elif model_type == 'alphavile-tiny':
        return get_alpha_vile_model(args, model_size='tiny')
    elif model_type == 'alphavile-small':
        return get_alpha_vile_model(args, model_size='small')
    elif model_type == 'alphavile-normal':
        return get_alpha_vile_model(args, model_size='normal')
    elif model_type == 'alphavile-large':
        return get_alpha_vile_model(args, model_size='large')
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def export_best_model_state(k_steps_best, k_steps_final, model, policy_loss_final, train_config, val_metric_values_best,
                            val_p_acc_final):
    prefix = train_config.export_dir + "weights/model-%.5f-%.3f" % (policy_loss_final, val_p_acc_final)
    # the export function saves both the architecture and the weights
    save_torch_state(model, torch.optim.SGD(model.parameters(), lr=train_config.max_lr), '%s-%04d.tar' % (prefix,
                                                                                                          k_steps_final))
    print(val_metric_values_best)
    # ## Copy best model to best-model directory
    val_loss_best = val_metric_values_best["loss"]
    val_p_acc_best = val_metric_values_best["policy_acc"]
    model_name = "model-%.5f-%.3f" % (val_loss_best, val_p_acc_best)
    model_prefix = train_config.export_dir + "weights/" + model_name
    model_tar_path = '%s-%04d.tar' % (model_prefix, k_steps_best)
    if not os.path.exists(train_config.export_dir + "best-model"):
        os.mkdir(train_config.export_dir + "best-model")
    best_model_prefix = train_config.export_dir + "best-model/" + model_name
    best_model_tar_path = '%s-%04d.tar' % (best_model_prefix, k_steps_best)
    shutil.copy(model_tar_path, best_model_tar_path)


def fill_train_objects(train_config, train_objects) -> None:
    if "adam" in train_config.optimizer_name:
        train_objects.lr_schedule = ConstantSchedule(train_config.min_lr)
    else:
        train_objects.lr_schedule = OneCycleSchedule(start_lr=train_config.max_lr / 8, max_lr=train_config.max_lr,
                                                     cycle_length=train_config.total_it * .3,
                                                     cooldown_length=train_config.total_it * .6,
                                                     finish_lr=train_config.min_lr)
    train_objects.lr_schedule = LinearWarmUp(train_objects.lr_schedule, start_lr=train_config.min_lr,
                                             length=train_config.total_it / 30)
    logging.getLogger().setLevel(logging.WARNING)
    plot_schedule(train_objects.lr_schedule, iterations=train_config.total_it)
    logging.getLogger().setLevel(logging.DEBUG)
    # ### Momentum schedule
    # In[ ]:
    train_objects.momentum_schedule = MomentumSchedule(train_objects.lr_schedule, train_config.min_lr,
                                                       train_config.max_lr, train_config.min_momentum,
                                                       train_config.max_momentum)
    plot_schedule(train_objects.momentum_schedule, iterations=train_config.total_it, ylabel='Momentum')


def create_validation_data(train_config):
    s_idcs_val, x_val, yv_val, yp_val, plys_to_end, pgn_datasets_val = load_pgn_dataset(dataset_type='val', part_id=0,
                                                                                        verbose=True,
                                                                                        normalize=train_config.normalize)
    val_data = get_data_loader(x_val, yv_val, yp_val, plys_to_end, train_config, shuffle=False)
    return val_data, x_val
