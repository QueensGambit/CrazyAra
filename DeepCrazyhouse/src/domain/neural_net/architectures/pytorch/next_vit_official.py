"""
@file: next_vit_official.py
Created on 11.08.22
@project: CrazyAra
@author: queensgambit

# Based on official NextViT code (Copyright (c) ByteDance Inc. All rights reserved.)
https://github.com/bytedance/Next-ViT/blob/main/classification/nextvit.py
"""
from torch import nn
from DeepCrazyhouse.src.domain.variants.constants import NB_POLICY_MAP_CHANNELS, NB_LABELS
from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.next_vit_official_modules import NTB, NCB, ConvBNReLU
from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.builder_util import get_act, _ValueHead, _PolicyHead, get_se, process_value_policy_head


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, ncb_layers, nct_layers, repeat, se_type, use_simple_transformer_blocks):
        super(Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ncb_layers = ncb_layers
        self.nct_layers = nct_layers
        self.repeat = repeat
        self.se_type = se_type
        self.use_simple_transformer_blocks = use_simple_transformer_blocks

        block = []
        for num in range(repeat):
            if num != repeat - 1:
                block += self._make_layer(self.in_channels, self.in_channels,
                                          self.ncb_layers, self.nct_layers)
            else:
                block += self._make_layer(self.in_channels, self.out_channels,
                                          self.ncb_layers, self.nct_layers)
        self.block = nn.Sequential(*block)

    def _make_layer(self, in_channels, out_channels, ncb_layers, nct_layers):
        self.sub_layers = []
        for _ in range(ncb_layers):
            self.sub_layers += [NCB(in_channels, out_channels)]
        for _ in range(nct_layers):
            self.sub_layers += [NTB(in_channels, out_channels, simple=self.use_simple_transformer_blocks)]
        return nn.Sequential(*self.sub_layers)

    def forward(self, x):
        return self.block(x)


class NextVit(nn.Module):
    def __init__(self,
                 image_size,
                 channels_policy_head,
                 stage3_repeat=2,
                 in_channels=52,
                 channels=256,
                 use_wdl=False, use_plys_to_end=False,
                 use_mlp_wdl_ply=False,
                 select_policy_from_plane=True,
                 se_type=None,
                 use_simple_transformer_blocks=False,
                 ):
        super().__init__()
        self.use_wdl = use_wdl
        self.use_plys_to_end = use_plys_to_end
        self.image_size = image_size
        self.stem = ConvBNReLU(in_channels, channels, kernel_size=3, stride=1)

        self.stage1 = nn.Sequential(
           Block(channels, channels, 4, 0, 1, se_type, use_simple_transformer_blocks),
        )
        self.stage3 = nn.Sequential(
            Block(channels, channels, 4, 1, stage3_repeat, se_type, use_simple_transformer_blocks),
        )

        self.value_head = _ValueHead(board_height=image_size, board_width=image_size, channels=channels, channels_value_head=8, fc0=256,
                                     nb_input_channels=channels, use_wdl=use_wdl, use_plys_to_end=use_plys_to_end, use_mlp_wdl_ply=use_mlp_wdl_ply)
        self.policy_head = _PolicyHead(board_height=image_size, board_width=image_size, channels=channels, policy_channels=channels_policy_head, n_labels=NB_LABELS,
                                       select_policy_from_plane=select_policy_from_plane)

    def forward(self, x):

        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage3(x)

        return process_value_policy_head(x, self.value_head, self.policy_head, self.use_plys_to_end, self.use_wdl)
