"""
@file: moe_gating.py
Created on 17.07.25
@project: CrazyAra
@author: queensgambit

A small simple gating network to evaluate having probabilities in our MoE / game phases paper.
"""

import torch
from torch.nn import Sequential, Conv2d, BatchNorm2d, Linear, MaxPool2d

from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.a0_resnet import ResidualBlock
from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.builder_util import get_act, _Stem


class MoEGatingNet(torch.nn.Module):

    def __init__(
        self,
        n_labels=3,
        nb_input_channels=52,
        board_height=8,
        board_width=8,
        phase_head_channels=3,
        act_type="relu",
    ):
        super(MoEGatingNet, self).__init__()

        self.nb_flatten = (phase_head_channels * board_width * board_height) // 4

        self.final_body = Sequential(Conv2d(in_channels=nb_input_channels, out_channels=phase_head_channels, kernel_size=(3, 3),
                                            padding=(1, 1), bias=False),
                                     MaxPool2d((2, 2), stride=2),
                                     BatchNorm2d(num_features=phase_head_channels),
                                     get_act(act_type))
        self.head = Sequential(Linear(in_features=self.nb_flatten, out_features=n_labels))

    def forward(self, x):
        """
        Implementation of the forward pass of the full network
        Uses a broadcast add operation for the shortcut and the output of the residual block
        :param x: Input to the ResidualBlock
        :return: Value & Policy Output
        """
        out = self.final_body(x).view(-1, self.nb_flatten)
        return self.head(out)


def get_moe_gating_model(args):
    """
    Wrapper definition for the MoE-Gating model
    :param args: Argument dictionary
    :return: pytorch model object
    """

    model = MoEGatingNet()
    return model
