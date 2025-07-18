"""
@file: moe_gating.py
Created on 17.07.25
@project: CrazyAra
@author: queensgambit

A small simple gating network to evaluate having probabilities in our MoE / game phases paper.
"""

import torch
from torch.nn import Sequential, Conv2d, BatchNorm2d, Linear

from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.a0_resnet import ResidualBlock
from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.builder_util import get_act, _Stem


class MoEGatingNet(torch.nn.Module):

    def __init__(
        self,
        n_labels=3,
        channels=64,
        nb_input_channels=52,
        board_height=8,
        board_width=8,
        num_res_blocks=3,
        phase_head_channels=3,
        act_type="relu",
    ):
        super(MoEGatingNet, self).__init__()

        self.nb_flatten = phase_head_channels * board_width * board_height

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResidualBlock(channels, act_type, use_se=False))

        self.body = Sequential(_Stem(channels=channels, act_type=act_type,
                                     nb_input_channels=nb_input_channels),
                               *res_blocks)
        self.final_body = Sequential(Conv2d(in_channels=channels, out_channels=phase_head_channels, kernel_size=(3, 3),
                                            padding=(1, 1), bias=False),
                                     BatchNorm2d(num_features=channels),
                                     get_act(act_type))
        self.head = Sequential(Linear(in_features=self.nb_flatten, out_features=n_labels))

    def forward(self, x):
        """
        Implementation of the forward pass of the full network
        Uses a broadcast add operation for the shortcut and the output of the residual block
        :param x: Input to the ResidualBlock
        :return: Value & Policy Output
        """
        out = self.body(x)
        out = self.final_body(out).view(-1, self.nb_flatten)
        return self.head(out)

