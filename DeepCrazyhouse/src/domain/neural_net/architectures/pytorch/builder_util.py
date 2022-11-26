"""
@file: builder_util
Created on 31.05.22
@project: CrazyAra
@author: queensgambit

Utility methods for building the neural network architectures.
"""

import math
import torch
from torch import nn
from torch.nn import Sequential, Conv1d, Conv2d, BatchNorm2d, ReLU, LeakyReLU, Sigmoid, Tanh, Linear, Hardsigmoid, Hardswish,\
    Module, AdaptiveAvgPool2d, BatchNorm1d
from timm.models.layers import DropPath
#from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.next_vit_official_modules import NTB


def round_to_next_multiple_of_32(number):
    """Rounds a given number to the next multiple of 32."""
    return int((number / 32) + 0.5) * 32


def get_act(act_type):
    """Wrapper method for different non linear activation functions"""
    if act_type == "relu":
        return ReLU(inplace=True)
    if act_type == "sigmoid":
        return Sigmoid()
    if act_type == "tanh":
        return Tanh()
    if act_type == "lrelu":
        return LeakyReLU(negative_slope=0.2)
    if act_type == "hard_sigmoid":
        return Hardsigmoid()
    if act_type == "hard_swish":
        return Hardswish()
    raise NotImplementedError


def get_se(se_type, channels, use_hard_sigmoid=False):
    """Wrapper method for different squeeze excitation types"""
    if se_type == "ca_se" or se_type == "se":
        return _ChannelAttentionModule(channels=channels, use_hard_sigmoid=use_hard_sigmoid)
    if se_type == "eca_se":
        return _EfficientChannelAttentionModule(channels=channels, use_hard_sigmoid=use_hard_sigmoid)
    raise NotImplementedError


class _EfficientChannelAttentionModule(torch.nn.Module):
    def __init__(self, channels, gamma=2, b=1, use_hard_sigmoid=False):
        """
        ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks - Wang et al. - https://arxiv.org/pdf/1910.03151.pdf
        :param channels: Number of channels for 1st conv operation
        :param act_type: Activation type to use
        :param nb_input_channels: Number of input channels of the board representation
        """
        super(_EfficientChannelAttentionModule, self).__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        kernel = t if t % 2 else t + 1

        if use_hard_sigmoid:
            act_type = "hard_sigmoid"
        else:
            act_type = "sigmoid"
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.body = Sequential(
            Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel, padding=kernel//2, stride=1,
                   bias=True),
            get_act(act_type))

    def forward(self, x):
        """
        Compute forward pass
        :param x: Input data to the block
        :return: Activation maps of the block
        """
        batch_size, channels, _, _ = x.size()
        out = self.avg_pool(x).view(batch_size, channels, 1)
        out = self.body(out).view(batch_size, channels, 1, 1)
        return x * out.expand_as(x)


class _ChannelAttentionModule(torch.nn.Module):
    def __init__(self, channels, reduction=2, use_hard_sigmoid=False):
        """
        Channel-wise attention module, Squeeze-and-Excitation Networks Jie Hu1, Li Shen, Gang Sun - https://arxiv.org/pdf/1709.01507v2.pdf
        :param channels: Number of input channels
        :param reduction: Reduction factor for the number of hidden units
        """
        super(_ChannelAttentionModule, self).__init__()

        if use_hard_sigmoid:
            act_type = "hard_sigmoid"
        else:
            act_type = "sigmoid"
        self.avg_pool = AdaptiveAvgPool2d(1)

        self.fc = Sequential(
            Linear(channels, channels // reduction, bias=False),
            ReLU(inplace=True),
            Linear(channels // reduction, channels, bias=False),
            get_act(act_type)
        )

    def forward(self, x):
        """
        Compute forward pass
        :param x: Input data to the block
        :return: Activation maps of the block
        """
        batch_size, channels, _, _ = x.size()
        y = self.avg_pool(x).view(batch_size, channels)
        y = self.fc(y).view(batch_size, channels, 1, 1)
        return x * y.expand_as(x)


class MixConv(Module):
    def __init__(self, in_channels, out_channels, kernels):
        """
        Mix depth-wise convolution layers, Mingxing Tan, Quoc V. Le, https://arxiv.org/abs/1907.09595
        :param in_channels: Number of input channels
        :param out_channels: Number of convolutional channels
        :param kernels: List of kernel sizes to use
        :return: symbol
        """
        super(MixConv, self).__init__()

        self.branches = []
        self.num_splits = len(kernels)

        for kernel in kernels:
            self.branch = Sequential(Conv2d(in_channels=in_channels // self.num_splits,
                                 out_channels=out_channels // self.num_splits, kernel_size=(kernel, kernel),
                                 padding=(kernel//2, kernel//2), bias=False,
                                 groups=out_channels // self.num_splits))
            self.branches.append(self.branch)

    def forward(self, x):
        """
        Compute forward pass
        :param x: Input data to the block
        :return: Activation maps of the block
        """
        if self.num_splits == 1:
            return self.branch(x)
        else:
            conv_layers = []
            for xi, branch in zip(torch.split(x, dim=1, split_size_or_sections=self.num_splits), self.branches):
                conv_layers.append(branch(xi))

        return torch.cat(conv_layers, 0)


class _Stem(torch.nn.Module):
    def __init__(self, channels,  act_type="relu", nb_input_channels=34):
        """
        Definition of the stem proposed by the alpha zero authors
        :param channels: Number of channels for 1st conv operation
        :param act_type: Activation type to use
        :param nb_input_channels: Number of input channels of the board representation
        """

        super(_Stem, self).__init__()

        self.body = Sequential(
            Conv2d(in_channels=nb_input_channels, out_channels=channels, kernel_size=(3, 3), padding=(1, 1),
                   bias=False),
            BatchNorm2d(num_features=channels),
            get_act(act_type))

    def forward(self, x):
        """
        Compute forward pass
        :param F: Handle
        :param x: Input data to the block
        :return: Activation maps of the block
        """
        return self.body(x)


class _DepthWiseStem(Module):
    def __init__(self, channels, act_type="relu", nb_input_channels=34):
        """
        Sames as _Stem() but with group depthwise convolutions
        """
        super(_DepthWiseStem, self).__init__()
        self.body = Sequential(Conv2d(in_channels=nb_input_channels, out_channels=channels, kernel_size=(3, 3), padding=(1, 1), bias=False),
                               BatchNorm2d(num_features=channels),
                               get_act(act_type),
                               Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), padding=(1, 1), bias=False, groups=channels),
                               BatchNorm2d(num_features=channels),
                               get_act(act_type),
                               Conv2d(in_channels=channels, out_channels=channels, kernel_size=(1, 1), padding=(0, 0), bias=True),
                               )

    def forward(self, x):
        """
        Compute forward pass
        :param F: Handle
        :param x: Input data to the block
        :return: Activation maps of the block
        """
        return self.body(x)


class _PolicyHead(Module):
    def __init__(self, board_height=11, board_width=11, channels=256, policy_channels=2, n_labels=4992, act_type="relu",
                 select_policy_from_plane=False):
        """
        Definition of the value head proposed by the alpha zero authors
        :param policy_channels: Number of channels for 1st conv operation in branch 0
        :param act_type: Activation type to use
        channelwise squeeze excitation, channel-spatial-squeeze-excitation, respectively
        :param use_transformer: Decides if a transformer should be used in the head
        """

        super(_PolicyHead, self).__init__()

        self.body = Sequential()
        self.select_policy_from_plane = select_policy_from_plane
        self.nb_flatten = policy_channels * board_width * board_height
        self.body = Sequential(
            Conv2d(in_channels=channels, out_channels=channels, padding=1, kernel_size=(3, 3), bias=False),
            BatchNorm2d(num_features=channels),
            get_act(act_type),
            Conv2d(in_channels=channels, out_channels=policy_channels, padding=1, kernel_size=(3, 3), bias=False))
        if not self.select_policy_from_plane:
            self.body2 = Sequential(BatchNorm2d(num_features=policy_channels),
                                    get_act(act_type))
            self.body3 = Sequential(Linear(in_features=self.nb_flatten, out_features=n_labels))

    def forward(self, x):
        """
        Compute forward pass
        :param x: Input data to the block
        :return: Activation maps of the block
        """
        if self.select_policy_from_plane:
            return self.body(x).view(-1, self.nb_flatten)
        else:
            x = self.body(x)
            x = self.body2(x).view(-1, self.nb_flatten)
            return self.body3(x)


class _ValueHead(Module):
    def __init__(self, board_height=11, board_width=11, channels=256, channels_value_head=1, fc0=256,
                 act_type="relu", use_raw_features=False, nb_input_channels=18,
                 use_wdl=False, use_plys_to_end=False, use_mlp_wdl_ply=False,
                 use_flat_inputs=False, in_features=512):
        """
        Definition of the value head proposed by the alpha zero authors
        :param board_height: Height of the board
        :param board_width: Width of the board
        :param channels: Number of channels as input
        :param channels_value_head: Number of channels for 1st conv operation in branch 0
        :param fc0: Number of units in Dense/Fully-Connected layer
        :param act_type: Activation type to use
        :param use_wdl: If a win draw loss head shall be used
        :param use_plys_to_end: If a plys to end prediction head shall be used
        :param use_mlp_wdl_ply: If a small mlp with value output for the wdl and ply head shall be used
        :param use_transformer: Decides if a transformer should be used in the head
        :param use_flat_inputs: If the input to the Value head is already flattened
        :param in_features: Number of flattened input features. Only relevant for use_flat_inputs=True.
        """

        super(_ValueHead, self).__init__()

        self.body = Sequential(Conv2d(in_channels=channels, out_channels=channels_value_head, kernel_size=(1, 1), bias=False),
                               BatchNorm2d(num_features=channels_value_head),
                               get_act(act_type))

        self.use_raw_features = use_raw_features
        self.use_wdl = use_wdl
        self.use_plys_to_end = use_plys_to_end
        self.use_mlp_wdl_ply = use_mlp_wdl_ply
        self.use_flat_inputs = use_flat_inputs
        self.nb_flatten = board_height*board_width*channels_value_head
        if use_raw_features:
            self.nb_flatten_raw = board_height*board_width*nb_input_channels
        else:
            self.nb_flatten_raw = 0
        if use_flat_inputs:
            self.nb_flatten_raw = in_features

        if use_wdl:
            self.body_wdl = Sequential(Linear(in_features=self.nb_flatten + self.nb_flatten_raw, out_features=3))
        if use_plys_to_end:
            self.body_plys = Sequential(Linear(in_features=self.nb_flatten + self.nb_flatten_raw, out_features=1),
                                        get_act('sigmoid'))

        if use_wdl and use_plys_to_end and use_mlp_wdl_ply:
            self.body_final = Sequential(Linear(in_features=4, out_features=8),
                                               get_act(act_type),
                                               Linear(in_features=8, out_features=1),
                                               get_act("tanh"))
        else:
            self.body_final = Sequential(Linear(in_features=self.nb_flatten+self.nb_flatten_raw, out_features=fc0),
                                    get_act(act_type),
                                    Linear(in_features=fc0, out_features=1),
                                    get_act("tanh"))

    def forward(self, x, raw_data=None):
        """
        Compute forward pass
        :param x: Input data to the block
        :return: Activation maps of the block
        """
        if not self.use_flat_inputs:
            x = self.body(x).view(-1, self.nb_flatten)
        if self.use_raw_features:
            raw_data = raw_data.view(-1, self.nb_flatten_raw)
            x = torch.cat((x, raw_data), dim=1)

        if self.use_wdl and self.use_plys_to_end:
            wdl_out = self.body_wdl(x)
            plys_out = self.body_plys(x)
            if self.use_mlp_wdl_ply:
                x = torch.cat((wdl_out, plys_out), dim=1)
                return self.body_final(x), wdl_out, plys_out
            else:
                wdl_out_softmax = torch.softmax(wdl_out, dim=1)
                (loss_out, _, win_out) = torch.split(wdl_out_softmax, 1, dim=1)
                return -loss_out + win_out, wdl_out, plys_out

        return self.body_final(x)


class _ValueHeadFlat(Module):
    def __init__(self, in_features=512, fc0=256, act_type="relu"):
        """
        Value head which uses flattened features as input
        :param in_features: Number of input features
        :param fc0: Number of units in Dense/Fully-Connected layer
        :param act_type: Activation type to use
        """

        super(_ValueHeadFlat, self).__init__()

        self.body = Sequential(Linear(in_features=in_features, out_features=fc0),
                               BatchNorm1d(num_features=fc0),
                               get_act(act_type),
                               Linear(in_features=fc0, out_features=1),
                               get_act("tanh")
                               )

    def forward(self, x):
        """
        Compute forward pass
        :param x: Input data to the block
        :return: Activation maps of the block
        """
        return self.body(x)


class _PolicyHeadFlat(Module):
    def __init__(self, in_features=512, fc0=256, act_type="relu", n_labels=4992):
        """
        Definition of the value head proposed by the alpha zero authors
        :param policy_channels: Number of channels for 1st conv operation in branch 0
        :param act_type: Activation type to use
        channelwise squeeze excitation, channel-spatial-squeeze-excitation, respectively
        """

        super(_PolicyHeadFlat, self).__init__()

        self.body = Sequential()
        # self.select_policy_from_plane = select_policy_from_plane

        self.body = Sequential(Linear(in_features=in_features, out_features=fc0),
                               BatchNorm1d(num_features=fc0),
                               get_act(act_type),
                               Linear(in_features=fc0, out_features=n_labels),
                               )

    def forward(self, x):
        """
        Compute forward pass
        :param x: Input data to the block
        :return: Activation maps of the block
        """
        return self.body(x)


def process_value_policy_head(x, value_head: _ValueHead, policy_head: _PolicyHead,
                              use_plys_to_end: bool, use_wdl: bool ):
    """
    Use the output to create value/policy predictions
    """
    value_head_out = value_head(x)
    policy_out = policy_head(x)
    if use_plys_to_end and use_wdl:
        value_out, wdl_out, plys_to_end_out = value_head_out
        auxiliary_out = torch.cat((wdl_out, plys_to_end_out), dim=1)
        return value_out, policy_out, auxiliary_out, wdl_out, plys_to_end_out
    else:
        value_out = value_head_out
        return value_out, policy_out


class ClassicalResidualBlock(torch.nn.Module):
    """
    Definition of a classical residual block without any pooling operation
    """

    def __init__(self, channels, act_type, se_type=None, path_dropout=0):
        """
        :param channels: Number of channels used in the conv-operations
        :param bn_mom: Batch normalization momentum
        :param act_type: Activation function to use
        """
        super(ClassicalResidualBlock, self).__init__()
        self.act_type = act_type
        self.se_type = se_type
        if se_type:
            self.se = get_se(se_type=se_type, channels=channels, use_hard_sigmoid=True)
        self.body = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), padding=(1, 1), bias=False),
                                  nn.BatchNorm2d(num_features=channels),
                                  get_act(act_type),
                                  nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), padding=(1, 1), bias=False),
                                  nn.BatchNorm2d(num_features=channels),
                                  get_act(act_type))
        self.path_dropout = DropPath(path_dropout)

    def forward(self, x):
        """
        Implementation of the forward pass of the residual block.
        Uses a broadcast add operation for the shortcut and the output of the residual block
        :param x: Input to the ResidualBlock
        :return: Sum of the shortcut and the computed residual block computation
        """
        if self.se_type:
            x = self.se(x)
        return x + self.path_dropout(self.body(x))


class _BottlekneckResidualBlock(Module):

    def __init__(self, channels, channels_operating, use_depthwise_conv=True, kernel=3, act_type='relu', se_type=None, path_dropout=0):
        """
        Returns a residual block without any max pooling operation
        :param channels: Number of filters for all CNN-layers
        :param name: Name for the residual block
        :param act_type: Activation function to use
        :param se_type: Squeeze excitation module that will be used
        :return: symbol
        """
        super(_BottlekneckResidualBlock, self).__init__()

        self.se_type = se_type
        if se_type:
            self.se = get_se(se_type=se_type, channels=channels, use_hard_sigmoid=True)
        if use_depthwise_conv:
            groups = channels_operating
        else:
            groups = 1
        self.body = Sequential(Conv2d(in_channels=channels, out_channels=channels_operating, kernel_size=(1, 1), bias=False),
                               BatchNorm2d(num_features=channels_operating),
                               get_act(act_type),
                               Conv2d(in_channels=channels_operating, out_channels=channels_operating, kernel_size=(kernel, kernel), padding=(kernel // 2, kernel // 2), bias=False, groups=groups),
                               BatchNorm2d(num_features=channels_operating),
                               get_act(act_type),
                               Conv2d(in_channels=channels_operating, out_channels=channels, kernel_size=(1, 1), bias=False),
                               BatchNorm2d(num_features=channels))
        self.path_dropout = DropPath(path_dropout)

    def forward(self, x):
        """
        Compute forward pass
        :param x: Input data to the block
        :return: Activation maps of the block
        """
        if self.se_type:
            x = self.se(x)
        return x + self.path_dropout(self.body(x))

