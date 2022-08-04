"""
@file: mobile_vit.py
Created on 07.07.22
@project: CrazyAra
@author: queensgambit

Please describe what the content of this file is about
https://github.com/lucidrains/vit-pytorch#mobilevit
"""
import torch
from vit_pytorch.mobile_vit import MobileViT, conv_nxn_bn, MobileViTBlock, conv_1x1_bn
import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Reduce
from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.builder_util import _ValueHead, _PolicyHead, process_value_policy_head
from DeepCrazyhouse.src.domain.variants.constants import NB_POLICY_MAP_CHANNELS, NB_LABELS


class MobileViT(nn.Module):
    """MobileViT.
    Paper: https://arxiv.org/abs/2110.02178
    Based on: https://github.com/chinhsuanwu/mobilevit-pytorch
    """

    def __init__(
        self,
        image_size,
        dims,
        channels,
        channels_policy_head,
        in_channels=3,
        kernel_size=3,
        patch_size=(2, 2),
        depths=(2, 4, 3),
        select_policy_from_plane=True,
        use_wdl=False, use_plys_to_end=False,
        use_mlp_wdl_ply=False
    ):
        super().__init__()
        assert len(dims) == 3, 'dims must be a tuple of 3'
        assert len(depths) == 3, 'depths must be a tuple of 3'

        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        init_dim = channels

        self.use_wdl = use_wdl
        self.use_plys_to_end = use_plys_to_end
        self.conv1 = conv_nxn_bn(in_channels, init_dim, stride=1)

        self.stem = nn.ModuleList([])
        self.stem.append(MV2Block(channels, channels, 1, 224))
        self.stem.append(MV2Block(channels, channels, 1, 256))
        self.stem.append(MV2Block(channels, channels, 1, 288))
        self.stem.append(MV2Block(channels, channels, 1, 320))

        self.trunk = nn.ModuleList([])
        self.trunk.append(nn.ModuleList([
            MV2Block(channels, channels, 1, 352),
            MobileViTBlock(dims[0], depths[0], channels,
                           kernel_size, patch_size, int(dims[0] * 2))
        ]))

        self.trunk.append(nn.ModuleList([
            MV2Block(channels, channels, 1, 384),
            MobileViTBlock(dims[1], depths[1], channels,
                           kernel_size, patch_size, int(dims[1] * 4))
        ]))

        self.trunk.append(nn.ModuleList([
            MV2Block(channels, channels, 1, 416),
            MobileViTBlock(dims[2], depths[2], channels,
                           kernel_size, patch_size, int(dims[2] * 4))
        ]))

        # self.to_logits = nn.Sequential(
        #     conv_1x1_bn(channels[-2], last_dim),
        #     Reduce('b c h w -> b c', 'mean'),
        #     nn.Linear(channels[-1], num_classes, bias=False)
        # )

        self.value_head = _ValueHead(board_height=image_size[0], board_width=image_size[1], channels=channels, channels_value_head=8, fc0=256,
                                     nb_input_channels=256, use_wdl=use_wdl, use_plys_to_end=use_plys_to_end, use_mlp_wdl_ply=use_mlp_wdl_ply)
        self.policy_head = _PolicyHead(board_height=image_size[0], board_width=image_size[1], channels=channels, policy_channels=channels_policy_head, n_labels=NB_LABELS,
                                       select_policy_from_plane=select_policy_from_plane)

    def forward(self, x):
        x = self.conv1(x)

        for conv in self.stem:
            x = conv(x)

        for conv, attn in self.trunk:
            x = conv(x)
            x = attn(x)

        return process_value_policy_head(x, self.value_head, self.policy_head, self.use_plys_to_end, self.use_wdl)


class MV2Block(nn.Module):
    """MV2 block described in MobileNetV2.
    Paper: https://arxiv.org/pdf/1801.04381
    Based on: https://github.com/tonylins/pytorch-mobilenet-v2
    """

    def __init__(self, inp, oup, stride=1, expansion_channels=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = expansion_channels
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion_channels == inp:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            out = out + x
        return out
