"""
@file: shuffle_netv2.py
Created on 02.05.19
@project: CrazyAra
@author: queensgambit

Incorporation of shuffling as in ShuffleNetV2 into the RISE architecture.

ShuffleNet V2: Practical Guidelines for EfficientCNN Architecture Design - https://arxiv.org/pdf/1807.11164.pdf
"""

from mxnet.gluon.nn import HybridSequential, Conv2D, BatchNorm
from mxnet.gluon import HybridBlock
from DeepCrazyhouse.src.domain.neural_net.architectures.builder_util import get_act
from DeepCrazyhouse.src.domain.neural_net.architectures.rise import _StemRise, ResidualBlockX
from DeepCrazyhouse.src.domain.neural_net.architectures.rise_builder_util import (
    _ChannelSqueezeExcitation,
    _SpatialSqueezeExcitation,
    _SpatialChannelSqueezeExcitation,
)
from DeepCrazyhouse.src.domain.neural_net.architectures.a0_resnet import (
    _StemAlphaZero,
    _PolicyHeadAlphaZero,
    _ValueHeadAlphaZero,
)


class _ShuffleChannelsBlock(HybridBlock):
    """
    Block which shuffles the channels according to ShuffleNet v2
    """

    def __init__(self, groups, **kwargs):
        """
        Constructor
        :param groups: Number of groups to shuffle
        :param kwargs: Optional additional arguments
        """
        super(_ShuffleChannelsBlock, self).__init__()
        self.groups = groups

    def hybrid_forward(self, F, x):
        data = F.reshape(x, shape=(0, -4, self.groups, -1, -2))
        data = F.swapaxes(data, 1, 2)
        data = F.reshape(data, shape=(0, -3, -2))
        return data


class _ShuffleBlock(HybridBlock):
    """
    Default Shuffle Block
    """
    def __init__(self, name, in_channels, groups=2, se_type="cSE", use_residual=False, act_type="relu", id=0, **kwargs):

        super(_ShuffleBlock, self).__init__(prefix=name + "_")

        self.in_channels = in_channels
        self.nb_right_channels = in_channels // 2

        self.groups = groups

        self.body = HybridSequential(prefix="")
        self.use_residual = use_residual
        self.id = id

        with self.name_scope():
            self.body.add(BatchNorm())
            self.body.add(
                Conv2D(channels=self.nb_right_channels+28, kernel_size=3, strides=1, padding=1, groups=1, use_bias=False)
            )
            self.body.add(BatchNorm())
            self.body.add(get_act(act_type))
            self.body.add(
                Conv2D(channels=self.nb_right_channels+28, kernel_size=3, strides=1, padding=1, groups=1, use_bias=False)
            )
            if se_type:
                if se_type == "cSE":
                    # apply squeeze excitation
                    self.body.add(_ChannelSqueezeExcitation("se0", self.nb_right_channels, 16, act_type))
                elif se_type == "sSE":
                    self.body.add(_SpatialSqueezeExcitation("se0"))
                elif se_type == "scSE":
                    self.body.add(_SpatialChannelSqueezeExcitation("se0", self.nb_right_channels, 2, act_type))
                else:
                    raise Exception('Unsupported Squeeze Excitation Module: Choose either [None, "cSE", "sSE", "scSE"')

            self.body.add(BatchNorm())

        if self.use_residual:
            self.act = get_act(act_type)
        self.shufller = _ShuffleChannelsBlock(groups)

    def hybrid_forward(self, F, x):
        left_in = F.slice_axis(x, axis=1, begin=0, end=self.nb_right_channels)
        right_in = F.slice_axis(x, axis=1, begin=self.nb_right_channels, end=self.in_channels)

        if self.id % 2 == 0:
            right_in = self.body(right_in)
        else:
            left_in = self.body(left_in)

        out = F.concat(right_in, left_in, dim=1)

        return out


class _ShuffleBlockNeck(HybridBlock):
    """
    Bottlekneck Shuffle Block
    """
    def __init__(self, name, nb_in_channels, groups=2, se_type="cSE", use_residual=True, act_type="relu", **kwargs):
        """
        Constructor
        :param name: Layer name 
        :param nb_in_channels: Number of input channels 
        :param groups: Number of groups to use for shuffling
        :param se_type: Squeez excitation type
        :param use_residual: True, if a residual connection shall be used
        :param act_type: Type for the activation function
        :param kwargs: 
        """

        super(_ShuffleBlockNeck, self).__init__(prefix=name + "_")

        self.in_channels = nb_in_channels
        self.nb_right_channels = nb_in_channels // 2

        self.groups = groups

        self.body = HybridSequential(prefix="")
        self.use_residual = use_residual
        with self.name_scope():
            self.body.add(Conv2D(channels=self.nb_right_channels, kernel_size=1, strides=1, padding=0, use_bias=False))
            self.body.add(BatchNorm())
            self.body.add(get_act(act_type))
            self.body.add(
                Conv2D(channels=self.nb_right_channels, kernel_size=1, strides=1, padding=0, groups=3, use_bias=False)
            )
            self.body.add(BatchNorm())
            self.body.add(get_act(act_type))
            if se_type:
                if se_type == "cSE":
                    self.body.add(_ChannelSqueezeExcitation("se0", self.nb_right_channels, 2, act_type))
                elif se_type == "sSE":
                    self.body.add(_SpatialSqueezeExcitation("se0"))
                elif se_type == "scSE":
                    self.body.add(_SpatialChannelSqueezeExcitation("se0", self.nb_right_channels, 2, act_type))
                else:
                    raise Exception('Unsupported Squeeze Excitation Module: Choose either [None, "cSE", "sSE", "scSE"')

        if self.use_residual:
            self.act = get_act(act_type)
        self.shufller = _ShuffleChannelsBlock(groups)

    def hybrid_forward(self, F, x):
        left_in = F.slice_axis(x, axis=1, begin=0, end=self.nb_right_channels)
        right_in = F.slice_axis(x, axis=1, begin=self.nb_right_channels, end=self.in_channels)
        right_out = self.body(right_in)
        if self.use_residual:
            right_out = right_in + right_out
            right_out = self.act(right_out)

        out = F.concat(left_in, right_out, dim=1)

        return self.shufller(out)


class ShuffleRise(HybridBlock):  # Too many arguments (15/5)
    """ Implementing the shuffle RISE architecture for learning chess proposed by Johannes Czech"""

    def __init__(
        self,
        n_labels=2272,
        channels=256,
        channels_value_head=8,
        channels_policy_head=16,
        nb_res_blocks_x=7,
        nb_shuffle_blocks=19,
        nb_shuffle_blocks_neck=19,
        value_fc_size=256,
        bn_mom=0.9,
        act_type="relu",
        squeeze_excitation_type=None,
        select_policy_from_plane=True,
        use_rise_stem=False,
        **kwargs
    ):  # Too many local variables (22/15)
        """
        Creates the alpha zero gluon net description based on the given parameters.

        :param n_labels: Number of labels the for the policy
        :param channels: Used for all convolution operations. (Except the last 2)
        :param nb_res_blocks_x: Number of residual blocks to stack. In the paper they used 19 or 39 residual blocks
        :param value_fc_size: Fully Connected layer size. Used for the value output
        :param bn_mom: Batch normalization momentum
        :param squeeze_excitation_type: Available types: [None, "cSE", "sSE", "scSE", "mixed"]
                                        cSE: Channel-wise-squeeze-excitation
                                        sSE: Spatial-wise-squeeze-excitation
                                        scSE: Channel-spatial-wise-squeeze-excitation
                                        mixed: Use cSE and sSE interchangeably
        :return: gluon net description
        """

        super(ShuffleRise, self).__init__(**kwargs, prefix="")
        self.body = HybridSequential(prefix="")

        with self.name_scope():
            se_type = None

            if use_rise_stem:
                self.body.add(_StemRise(name="stem", channels=channels, se_type=squeeze_excitation_type))
            else:
                self.body.add(
                    _StemAlphaZero(name="stem", channels=channels, bn_mom=bn_mom, act_type=act_type, se_type=se_type)
                )

        for i in range(nb_res_blocks_x):
            unit_name = "res_unit%d" % i
            self.body.add(ResidualBlockX(unit_name, channels=channels, bn_mom=0.9, act_type=act_type,
                                         se_type=squeeze_excitation_type))

        for i in range(nb_shuffle_blocks):
            unit_name = "shuffle_unit%d" % i
            self.body.add(
                _ShuffleBlock(unit_name, in_channels=channels, se_type=squeeze_excitation_type, act_type=act_type, id=i)
            )
            channels += 28

        for i in range(nb_shuffle_blocks_neck):
            unit_name = "shuffle_unit_neck_%d" % i
            self.body.add(
                _ShuffleBlockNeck(unit_name, nb_in_channels=channels, se_type=squeeze_excitation_type, act_type=act_type)
            )

        se_type = None
        # create the two heads which will be used in the hybrid fwd pass
        self.value_head = _ValueHeadAlphaZero("value", channels_value_head, value_fc_size, bn_mom, act_type, se_type)
        self.policy_head = _PolicyHeadAlphaZero(
            "policy", channels_policy_head, n_labels, bn_mom, act_type, se_type, select_policy_from_plane
        )

    def hybrid_forward(self, F, x):
        """
        Implementation of the forward pass of the full network
        Uses a broadcast add operation for the shortcut and the output of the residual block
        :param F: Abstract Function handle which works for gluon & mxnet
        :param x: Input to the ResidualBlock
        :return: Value & Policy Output
        """
        out = self.body(x)
        value = self.value_head(out)
        policy = self.policy_head(out)
        return [value, policy]
