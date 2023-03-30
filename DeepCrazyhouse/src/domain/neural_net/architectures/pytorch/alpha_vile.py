"""
@file: alpha_vile.py
Created on 30.03.23
@project: CrazyAra
@author: queensgambit

Description of the AlphaVile architecture for the sizes tiny, small, normal and large.
"""

import random
from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.rise_mobile_v3 import RiseV3


def get_alpha_vile_model(args, model_size='normal'):
    """
    Wrapper definition for AlphaVile models
    :param args: Argument dictionary
    :param model_size: Available options ['tiny', 'small', 'normal', 'large']
    :return: pytorch model object
    """

    base_channels_options = {'tiny': 192,
                     'small': 192,
                     'normal': 224,
                     'large': 224}
    nb_transformers_options = {'tiny': 1,
                     'small': 1,
                     'normal': 2,
                     'large': 2}
    depth_options = {'tiny': 15,
                     'small': 22,
                     'normal': 26,
                     'large': 37}

    expansion_ratio = 2
    kernel_5_ratio = 0.5
    base_kernel_5_channel_ratio = 0.68
    se_ratio = 0.0
    hard_swish_ratio = 0.0
    base_channels = base_channels_options[model_size]
    kernel_5_channel_ratio = (int(
        ((base_channels * expansion_ratio * base_kernel_5_channel_ratio) / 32) + 0.5) * 32) / (
                                         base_channels * expansion_ratio)
    depth = depth_options[model_size]
    nb_transformers = nb_transformers_options[model_size]

    kernels = [3] * depth
    end_idx = int(len(kernels) * kernel_5_ratio + 0.5)
    for idx in range(end_idx):
        kernels[idx] = 5
    random.shuffle(kernels)

    use_transformers = [False] * len(kernels)
    if nb_transformers > 0:
        block_size = len(kernels) // (nb_transformers + 1)
        start_idx = len(kernels) % block_size + 2 * block_size - 1
        for idx in range(start_idx, len(kernels), block_size):
            use_transformers[idx] = True

    se_types = [None] * len(kernels)
    end_idx = int(len(kernels) * se_ratio + 0.5)
    for idx in range(end_idx):
        se_types[idx] = "eca_se"
    se_types.reverse()

    act_types = ['relu'] * len(kernels)
    end_idx = int(len(kernels) * hard_swish_ratio + 0.5)
    for idx in range(end_idx):
        act_types[idx] = "hard_swish"
    act_types.reverse()

    act_types = ['relu'] * len(kernels)

    model = RiseV3(nb_input_channels=args.input_shape[0], board_height=args.input_shape[1],
                   board_width=args.input_shape[2],
                   channels=base_channels, channels_operating_init=base_channels * expansion_ratio, channel_expansion=0,
                   act_types=act_types,
                   channels_value_head=8, value_fc_size=base_channels,
                   channels_policy_head=args.channels_policy_head,
                   dropout_rate=0, select_policy_from_plane=args.select_policy_from_plane,
                   kernels=kernels, se_types=se_types, use_avg_features=False, n_labels=args.n_labels,
                   use_wdl=args.use_wdl, use_plys_to_end=args.use_plys_to_end, use_mlp_wdl_ply=args.use_mlp_wdl_ply,
                   use_transformers=use_transformers, path_dropout=0.05,
                   conv_block="mobile_bottlekneck_res_block",
                   kernel_5_channel_ratio=kernel_5_channel_ratio
                   )
    return model
