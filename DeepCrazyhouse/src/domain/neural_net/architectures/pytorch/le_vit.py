"""
@file: le_vit.py
Created on 06.07.22
@project: CrazyAra
@author: queensgambit

Please describe what the content of this file is about
https://github.com/lucidrains/vit-pytorch
"""
import torch
from torch import nn

from vit_pytorch.levit import LeViT
from vit_pytorch.levit import Transformer, cast_tuple, Rearrange, exists, ceil, always
from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.builder_util import get_act, _ValueHead, _PolicyHead, process_value_policy_head, ClassicalResidualBlock
from DeepCrazyhouse.src.domain.variants.constants import NB_POLICY_MAP_CHANNELS, NB_LABELS



class LeViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        # num_classes,
        channels_policy_head,
        dim,
        depth,
        heads,
        mlp_mult,
        in_channels = 3,
        stages = 3,
        dim_key = 32,
        dim_value = 64,
        dropout = 0.,
        use_wdl=False, use_plys_to_end=False,
        use_mlp_wdl_ply=False,
        select_policy_from_plane=True
    ):
        super().__init__()
        self.use_wdl = use_wdl
        self.use_plys_to_end = use_plys_to_end

        dims = cast_tuple(dim, stages)
        depths = cast_tuple(depth, stages)
        layer_heads = cast_tuple(heads, stages)

        assert all(map(lambda t: len(t) == stages, (dims, depths, layer_heads))), 'dimensions, depths, and heads must be a tuple that is less than the designated number of stages'

        self.conv_embedding = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            get_act("hard_swish"),
            ClassicalResidualBlock(256, "hard_swish"),
            # ResidualBlock(256, "hard_swish"),
        )

        fmap_size = image_size # // (2 ** 4)
        layers = []

        for ind, dim, depth, heads in zip(range(stages), dims, depths, layer_heads):
            # is_last = ind == (stages - 1)
            layers.append(Transformer(dim, fmap_size, depth, heads, dim_key, dim_value, mlp_mult, dropout))

            # if not is_last:
            #     next_dim = dims[ind + 1]
            #     layers.append(Transformer(dim, fmap_size, 1, heads * 2, dim_key, dim_value, dim_out = next_dim, downsample = True))
            #     fmap_size = ceil(fmap_size / 2)

        self.backbone = nn.Sequential(*layers)

        # self.pool = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     Rearrange('... () () -> ...')
        # )
        self.value_head = _ValueHead(board_height=image_size, board_width=image_size, channels=dims[0], channels_value_head=8, fc0=256, act_type="hard_swish",
                                     nb_input_channels=256, use_wdl=use_wdl, use_plys_to_end=use_plys_to_end, use_mlp_wdl_ply=use_mlp_wdl_ply)
        self.policy_head = _PolicyHead(board_height=image_size, board_width=image_size, channels=dims[0], policy_channels=channels_policy_head, n_labels=NB_LABELS,
                                       select_policy_from_plane=select_policy_from_plane, act_type="hard_swish",)

        # self.distill_head = nn.Linear(dim, num_distill_classes) if exists(num_distill_classes) else always(None)
        # self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.conv_embedding(img)

        x = self.backbone(x)

        return process_value_policy_head(x, self.value_head, self.policy_head, self.use_plys_to_end, self.use_wdl)
