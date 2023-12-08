"""
@file: model_config.py
Created on 05.12.23
@project: CrazyAra
@author: queensgambit

Model configuration file
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Class which stores all model configurations.
    Note not all models support all parameters."""

    # all models
    channels: int = 256,
    channels_value_head: int = 8,
    channels_policy_head: int = 81,
    value_fc_size: int = 256,

    # resent
    num_res_blocks: int = 19,

    # risev2, risev3, alphavile
    channels_operating_init: int = 224,
    channel_expansion: int = 32,

    kernels = [3] * 15,
    se_types = [None] * len(kernels),
    act_types = ['relu'] * len(kernels),

    # alphavile
    use_transformers = [False] * len(kernels)

    path_dropout: float = 0.05,
    kernel_5_channel_ratio: float = 0.5,

    # nextvit
    stage3_repeat: int = 1

    def __init__(self):
        self.kernels[7] = 5
        self.kernels[11] = 5
        self.kernels[12] = 5
        self.kernels[13] = 5

        self.se_types[5] = "eca_se"
        self.se_types[8] = "eca_se"
        self.se_types[12] = "eca_se"
        self.se_types[13] = "eca_se"
        self.se_types[14] = "eca_se"

        self.use_transformers[7] = True
        self.use_transformers[14] = True
