"""
@file: next_vit.py
Created on 01.08.22
@project: CrazyAra
@author: queensgambit

Please describe what the content of this file is about
https://github.com/wilile26811249/Next-ViT/blob/main/model/module.py
"""
import torch
import torch.nn as nn

from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.builder_util import get_act, _ValueHead, _PolicyHead, process_value_policy_head
from DeepCrazyhouse.src.domain.variants.constants import NB_POLICY_MAP_CHANNELS, NB_LABELS
from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.next_vit_modules import Stem, PatchEmbed, Block, Conv3x3


class NextVit(nn.Module):
    def __init__(self,
                 image_size,
                 channels_policy_head,
                 stage3_repeat=7,#2,
                 in_channels=52,
                 channels=256,
                 use_wdl=False, use_plys_to_end=False,
                 use_mlp_wdl_ply=False,
                 select_policy_from_plane=True
                 ):
        super().__init__()
        self.use_wdl = use_wdl
        self.use_plys_to_end = use_plys_to_end
        self.image_size = image_size

        self.next_vit_channel = [96, 192, 384, 768]

        # Next-Vit Layer
        # self.stem = Stem(in_channels, channels)
        self.stem = Conv3x3(in_channels, channels)
        self.stage1 = nn.Sequential(
            # PatchEmbed(64, self.next_vit_channel[0]),
            Block(channels, channels, 1, 0, 1),
        )
        # self.stage2 = nn.Sequential(
        #     PatchEmbed(self.next_vit_channel[0], self.next_vit_channel[1]),
        #     Block(self.next_vit_channel[1], 256, 3, 1, 1),
        # )
        self.stage3 = nn.Sequential(
            # PatchEmbed(256, self.next_vit_channel[2]),
            # Block(self.next_vit_channel[2], 512, 4, 1, stage3_repeat),
            Block(channels, channels, 4, 1, stage3_repeat),
        )
        # self.stage4 = nn.Sequential(
        #     PatchEmbed(512, self.next_vit_channel[3]),
        #     Block(self.next_vit_channel[3], 1024, 2, 1, 1),
        # )

        # # Global Average Pooling
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #
        # # FC
        # self.fc = nn.Sequential(
        #     nn.Linear(1024, 1280),
        #     nn.ReLU(inplace = True),
        # )
        #
        # # Final Classifier
        # self.classifier = nn.Linear(1280, num_class)

        self.value_head = _ValueHead(board_height=image_size, board_width=image_size, channels=channels, channels_value_head=8, fc0=256,
                                     nb_input_channels=256, use_wdl=use_wdl, use_plys_to_end=use_plys_to_end, use_mlp_wdl_ply=use_mlp_wdl_ply)
        self.policy_head = _PolicyHead(board_height=image_size, board_width=image_size, channels=channels, policy_channels=channels_policy_head, n_labels=NB_LABELS,
                                       select_policy_from_plane=select_policy_from_plane,)

    def forward(self, x):

        x = self.stem(x)
        x = self.stage1(x)
        # print('x after stage 1:', x.shape)
        x = self.stage3(x)
        # print('x after stage 3:', x.shape)

        return process_value_policy_head(x, self.value_head, self.policy_head, self.use_plys_to_end, self.use_wdl)

    # def forward(self, x):
    #     x = self.stem(x)
    #     x = self.stage1(x)
    #     x = self.stage2(x)
    #     x = self.stage3(x)
    #     x = self.stage4(x)
    #
    #     x = self.avg_pool(x)
    #     x = torch.flatten(x, 1)
    #
    #     x = self.fc(x)
    #     logit = self.classifier(x)
    #     return logit


def NextViT_S():
    net = NextVit(stage3_repeat = 2)
    return net

def NextViT_B():
    net = NextVit(stage3_repeat = 4)
    return net

def NextViT_L():
    net = NextVit(stage3_repeat = 6)
    return net
