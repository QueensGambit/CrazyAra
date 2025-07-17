"""
@file: moe_gating.py
Created on 17.07.25
@project: CrazyAra
@author: queensgambit

A small simple gating network to evaluate having probabilities in our MoE / game phases paper.
"""

import torch
import torch.nn as nn
from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU, LeakyReLU, Sigmoid, Tanh, Linear


class MoEGatingNet(torch.nn.Module):

    def __init__(
        self,
        n_labels=3,
        channels=256,
        nb_input_channels=52,
        board_height=8,
        board_width=8,
        num_res_blocks=5,
        value_fc_size=256,
        act_type="relu",
    ):

        pass


    def forward(self, x):
        """
        Implementation of the forward pass of the full network
        Uses a broadcast add operation for the shortcut and the output of the residual block
        :param x: Input to the ResidualBlock
        :return: Value & Policy Output
        """
        out = self.body(x)

        pass

