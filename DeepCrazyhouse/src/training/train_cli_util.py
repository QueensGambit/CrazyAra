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
import numpy as np
from pathlib import Path
import glob

# Pytorch imports
import torch
from fvcore.nn import flop_count_table
from torchsummary import summary
from fvcore.nn import FlopCountAnalysis

from DeepCrazyhouse.src.domain.variants.constants import NB_POLICY_MAP_CHANNELS, NB_LABELS
from DeepCrazyhouse.configs.main_config import main_config
# architectures
from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.rise_mobile_v3 import RiseV3, \
    get_rise_v2_model, get_rise_v33_model
from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.vision_transformer import VisionTransformer,\
    get_vision_transformer_model
from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.vit_configs import get_b8_config
from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.next_vit_official import NextVit, get_next_vit_model
from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.a0_resnet import AlphaZeroResnet, get_alpha_zero_model
from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.alpha_vile import get_alpha_vile_model
from DeepCrazyhouse.configs.train_config import TrainConfig, TrainObjects
from DeepCrazyhouse.configs.model_config import ModelConfig
from DeepCrazyhouse.src.preprocessing.dataset_loader import load_pgn_dataset
from DeepCrazyhouse.src.training.lr_schedules.lr_schedules import plot_schedule, ConstantSchedule, OneCycleSchedule,\
    LinearWarmUp, MomentumSchedule
from DeepCrazyhouse.src.training.train_util import get_metrics
from DeepCrazyhouse.src.training.trainer_agent_pytorch import save_torch_state, export_to_onnx, get_context,\
    get_data_loader


class Args:
    """Dummy class to mimic command-line arguments."""
    input_shape = None
    channels_policy_head = None
    n_labels = None
    select_policy_from_plane = None
    use_wdl = None
    use_plys_to_end = None
    use_mlp_wdl_ply = None


def create_args_by_train_config(input_shape: tuple, tc: TrainConfig) -> Args:
    """Creates an Args objects and fills it with the train config."""
    args = Args()
    args.input_shape = input_shape
    args.channels_policy_head = NB_POLICY_MAP_CHANNELS
    args.n_labels = NB_LABELS
    args.select_policy_from_plane = tc.select_policy_from_plane
    args.use_wdl = tc.use_wdl
    args.use_plys_to_end = tc.use_plys_to_end
    args.use_mlp_wdl_ply = tc.use_mlp_wdl_ply
    return args


def fill_train_config(train_config: TrainConfig, x_val) -> None:
    """Fills train config items based on other items."""
    train_config.nb_parts = len(glob.glob(main_config['planes_train_dir'] + '**/*'))
    nb_it_per_epoch = (len(
        x_val) * train_config.nb_parts) // train_config.batch_size  # calculate how many iterations per epoch exist
    # one iteration is defined by passing 1 batch and doing backprop
    train_config.total_it = int(nb_it_per_epoch * train_config.nb_training_epochs)


def get_custom_model(model_type: str, args: Args):
    """Returns a custom pytorch model based on the model_config.py settings."""
    mc = ModelConfig()
    if model_type == 'resnet':
        model = AlphaZeroResnet(channels=mc.channels, channels_value_head=mc.channels_value_head,
                                channels_policy_head=args.channels_policy_head,
                                value_fc_size=mc.value_fc_size, num_res_blocks=mc.num_res_blocks, act_type='relu',
                                n_labels=args.n_labels, select_policy_from_plane=args.select_policy_from_plane,
                                use_wdl=args.use_wdl, use_plys_to_end=args.use_plys_to_end,
                                use_mlp_wdl_ply=args.use_mlp_wdl_ply)
        return model
    elif model_type == 'vit':
        num_classes = args.channels_policy_head * 64 if args.select_policy_from_plane else args.n_labels
        model = VisionTransformer(get_b8_config(), img_size=8, in_channels=args.input_shape[0], num_classes=num_classes,
                                  use_wdl=args.use_wdl, use_plys_to_end=args.use_plys_to_end,
                                  use_mlp_wdl_ply=args.use_mlp_wdl_ply, )
        return model
    elif model_type == 'risev2' or 'risev3':
        model = RiseV3(nb_input_channels=args.input_shape[0], board_height=args.input_shape[1],
                       board_width=args.input_shape[2],
                       channels=mc.channels, channels_operating_init=mc.channels_operating_init,
                       channel_expansion=mc.channel_expansion, act_types=mc.act_types,
                       channels_value_head=mc.channels_value_head, value_fc_size=mc.value_fc_size,
                       channels_policy_head=args.channels_policy_head,
                       dropout_rate=0, select_policy_from_plane=args.select_policy_from_plane,
                       kernels=mc.kernels, se_types=mc.se_types, use_avg_features=False, n_labels=args.n_labels,
                       use_wdl=args.use_wdl, use_plys_to_end=args.use_plys_to_end, use_mlp_wdl_ply=args.use_mlp_wdl_ply,
                       )
        return model
    elif model_type == 'alphavile':
        model = RiseV3(nb_input_channels=args.input_shape[0], board_height=args.input_shape[1],
                       board_width=args.input_shape[2],
                       channels=mc.channels, channels_operating_init=mc.channels_operating_init,
                       channel_expansion=mc.channel_expansion,
                       act_types=mc.act_types,
                       channels_value_head=mc.channels_value_head, value_fc_size=mc.value_fc_size,
                       channels_policy_head=args.channels_policy_head,
                       dropout_rate=0, select_policy_from_plane=args.select_policy_from_plane,
                       kernels=mc.kernels, se_types=mc.se_types, use_avg_features=False, n_labels=args.n_labels,
                       use_wdl=args.use_wdl, use_plys_to_end=args.use_plys_to_end, use_mlp_wdl_ply=args.use_mlp_wdl_ply,
                       use_transformers=mc.use_transformers, path_dropout=mc.path_dropout,
                       conv_block="mobile_bottlekneck_res_block",
                       kernel_5_channel_ratio=mc.kernel_5_channel_ratio
                       )
        return model
    elif model_type == 'nextVit':
        model = NextVit(
            image_size=args.input_shape[1],
            in_channels=args.input_shape[0],
            channels_policy_head=args.channels_policy_head,
            stage3_repeat=1,
            channels=mc.channels,
            select_policy_from_plane=args.select_policy_from_plane,
            use_wdl=args.use_wdl, use_plys_to_end=args.use_plys_to_end,
            use_mlp_wdl_ply=args.use_mlp_wdl_ply,
            se_type=None,
            use_simple_transformer_blocks=False,
        )
        return model
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_default_model(model_type: str, args: Args):
    """Returns a pytorch object based on the given model_type."""
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
    elif model_type == 'nextVit':
        return get_next_vit_model(args)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def create_pytorch_model(input_shape: tuple, train_config: TrainConfig):
    """Implement logic to create a PyTorch model based on model_type."""
    args = create_args_by_train_config(input_shape, train_config)

    if train_config.use_custom_architecture:
        return get_custom_model(train_config.model_type, args)

    return get_default_model(train_config.model_type, args)


def create_export_dirs(train_config: TrainConfig) -> None:
    """Creates export directories for 'best-model' and 'configs' if they don't exist already."""
    if not os.path.exists(Path(train_config.export_dir, "best-model")):
        os.mkdir(Path(train_config.export_dir, "best-model"))
    if not os.path.exists(Path(train_config.export_dir, "configs")):
        os.mkdir(Path(train_config.export_dir, "configs"))


def export_configs(args, train_config: TrainConfig) -> None:
    """Copies the main_config.py, train_config.py and model_config.py file to the export config directory."""
    logging.info("Main Config:")
    print(main_config)
    logging.info("Train Config:")
    print(train_config)
    if args.use_custom_architecture:
        logging.info("Model Config:")
        print(ModelConfig())
        export_config(train_config, "model_config.py")

    export_config(train_config, "main_config.py")
    export_config(train_config, "train_config.py")


def export_config(train_config: TrainConfig, config_file: str) -> None:
    """Exports a given config file."""
    config_src_path = Path('../../../DeepCrazyhouse/configs', config_file)
    config_dst_path = Path(train_config.export_dir, 'configs', config_file)
    shutil.copy(config_src_path, config_dst_path)


def export_best_model_state(k_steps_best: int, k_steps_final: int, model, policy_loss_final: float, input_shape: tuple,
                            train_config: TrainConfig, val_metric_values_best: dict, val_p_acc_final: float) -> None:
    """Copies the best model checkpoint file achieved during training to the 'train_config.export_dir/best-model'
     directory."""
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

    # ## Convert to onnx
    convert_model_to_onnx(input_shape, k_steps_best, model, model_name, train_config)

    print("Saved weight & onnx files of the best model to %s" % (train_config.export_dir + "best-model"))


def convert_model_to_onnx(input_shape, k_steps_best, model, model_name, train_config):
    """Converts a given model to onnx and saves the files in the 'train_config.export_dir/best-model' directory."""
    model_prefix = "%s-%04d" % (model_name, k_steps_best)
    with torch.no_grad():
        ctx = get_context(train_config.context, train_config.device_id)
        dummy_input = torch.zeros(1, input_shape[0], input_shape[1], input_shape[2]).to(ctx)
        export_to_onnx(model, 1, dummy_input,
                       Path(train_config.export_dir) / Path("best-model"), model_prefix, train_config.use_wdl and
                       train_config.use_plys_to_end, True)


def fill_train_objects(train_config: TrainConfig, train_objects: TrainObjects) -> None:
    """Fills the train objects with a learning rate schedule, momentum schedule and metrics."""
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
    train_objects.momentum_schedule = MomentumSchedule(train_objects.lr_schedule, train_config.min_lr,
                                                       train_config.max_lr, train_config.min_momentum,
                                                       train_config.max_momentum)
    plot_schedule(train_objects.momentum_schedule, iterations=train_config.total_it, ylabel='Momentum')

    # Define the metrics to use
    train_objects.metrics = get_metrics(train_config)


def get_validation_data(train_config: TrainConfig):
    """
    Returns the validation loader, x-Data and target-Policy object.
    """
    s_idcs_val, x_val, yv_val, yp_val, plys_to_end, pgn_datasets_val = load_pgn_dataset(dataset_type='val', part_id=0,
                                                                                        verbose=True,
                                                                                        normalize=train_config.normalize)
    val_data = get_data_loader(x_val, yv_val, yp_val, plys_to_end, train_config, shuffle=False)
    return val_data, x_val, yp_val


def print_model_summary(input_shape: tuple, model, x_val) -> None:
    """Prints the model summary."""
    summary(model, (input_shape[0], input_shape[1], input_shape[2]), device="cpu")
    # Analyze the Flops
    dummy_input = torch.Tensor(np.expand_dims(x_val[0], axis=0)).to('cpu')
    flops = FlopCountAnalysis(model, dummy_input)
    print(flops.total())
    print(flop_count_table(flops))
