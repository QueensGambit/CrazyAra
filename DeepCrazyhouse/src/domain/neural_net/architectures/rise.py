"""
@file: Rise.py
Created on 26.09.18
@project: crazy_ara_refactor
@author: queensgambit

Description of the RISE architecture for learning chess proposed by Johannes Czech in 2018 for CrazyAra 0.2.0.
The architecture incorporates new ideas and techniques described in recent papers for Deep Learning in Computer Vision.

R - ResneXt (Aggregated Residual Transformations for Deep Neural Networks, Xie et al., http://arxiv.org/abs/1611.05431)
            (Deep Residual Learning for Image Recognition, He et al., https://arxiv.org/pdf/1512.03385.pdf)
I - Inception (Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,
                Szegedy et al., https://arxiv.org/pdf/1602.07261.pdf)
              (Rethinking the Inception Architecture for Computer Vision - https://arxiv.org/pdf/1512.00567.pdf,
               Szegedy et al., https://arxiv.org/pdf/1512.00567.pdf)
S - Squeeze (Squeeze-and-Excitation Networks,  Xie et al., https://arxiv.org/pdf/1709.01507.pdf)
E - Excitation

The proposed model architecture has fewer parameters,
faster inference and training time while maintaining an equal amount of depth
compared to the architecture proposed by DeepMind (19 residual layers with 256 filters).
On our 10,000 games benchmark dataset it achieved a lower validation error.
"""

import mxnet as mx
from mxnet.gluon.nn import HybridSequential, Conv2D, BatchNorm, Flatten, Dense
from mxnet.gluon import HybridBlock
from DeepCrazyhouse.src.domain.variants.plane_policy_representation import FLAT_PLANE_IDX
from DeepCrazyhouse.src.domain.neural_net.architectures.builder_util import get_act

from DeepCrazyhouse.src.domain.neural_net.architectures.rise_builder_util import (
    _ChannelSqueezeExcitation,
    _SpatialSqueezeExcitation,
    _SpatialChannelSqueezeExcitation,
    _GatherExcitePlus)
from DeepCrazyhouse.src.domain.neural_net.architectures.a0_resnet import (
    _StemAlphaZero,
)


class ResidualBlockX(HybridBlock):  # Too many arguments (8/5)
    """
    Definition of a residual block without any pooling operation
    """

    def __init__(self, unit_name, channels, bn_mom, act_type, se_type="scSE"):
        """

        :param channels: Number of channels used in the conv-operations
        :param bn_mom: Batch normalization momentum
        :param act_type: Activation function to use
        :param unit_name: Unit name of the residual block (only used for description (string))
        """
        super(ResidualBlockX, self).__init__(unit_name + "_")
        self.act_type = act_type
        self.unit_name = unit_name
        self.body = HybridSequential(prefix="")
        self.channels = channels

        with self.name_scope():
            if se_type:
                if se_type == "cSE":
                    # apply squeeze excitation
                    self.body.add(_ChannelSqueezeExcitation("se0", channels, 2, act_type))
                elif se_type == "sSE":
                    self.body.add(_SpatialSqueezeExcitation("se0"))
                elif se_type == "scSE":
                    self.body.add(_SpatialChannelSqueezeExcitation("se0", channels, 2, act_type))
                elif se_type == "GE+":
                    self.body.add(_GatherExcitePlus("ge0", channels, 2, act_type))
                else:
                    raise Exception('Unsupported Squeeze Excitation Module: Choose either [None, "cSE", "sSE", "scSE",'
                                    '"GE+')
            self.body.add(BatchNorm(momentum=bn_mom))
            self.body.add(get_act(self.act_type))
            self.body.add(Conv2D(channels=channels, kernel_size=3, padding=1, groups=1, use_bias=False))
            self.body.add(BatchNorm(momentum=bn_mom))
            self.body.add(get_act(self.act_type))
            self.body.add(Conv2D(channels=channels, kernel_size=3, padding=1, groups=1, use_bias=False))

    def hybrid_forward(self, F, x): #, first_half_indices, second_half_indices):
        """
        Implementation of the forward pass of the residual block.
        Uses a broadcast add operation for the shortcut and the output of the residual block
        :param F: Abstract Function handle which works for gluon & mxnet
        :param x: Input to the ResidualBlock
        :return: Sum of the shortcut and the computed residual block computation
        """
        shortcut = x
        out = self.body(x)
        out = shortcut + out

        # apply activation
        return out


class _ExpanderResidualBlock(HybridBlock):
    """
    Definition of a residual block without any pooling operation
    """

    def __init__(self, unit_name, channels, channels_expanded, kernel_size=3, bn_mom=0.9, act_type="relu", se_type="scSE", dim_match=True):
        """
        Constructor
        :param channels: Number of channels used in the conv-operations
        :param bn_mom: Batch normalization momentum
        :param act_type: Activation function to use
        :param unit_name: Unit name of the residual block (only used for description (string))
        """
        super(_ExpanderResidualBlock, self).__init__(prefix=unit_name + "_")
        self.unit_name = unit_name
        self.use_se = se_type
        self.dim_match = dim_match
        self.body = HybridSequential(prefix="")

        with self.name_scope():

            if se_type:
                if se_type == "cSE":
                    # apply squeeze excitation
                    self.body.add(_ChannelSqueezeExcitation("se0", channels, 2, act_type))
                elif se_type == "sSE":
                    self.body.add(_SpatialSqueezeExcitation("se0"))
                elif se_type == "scSE":
                    self.body.add(_SpatialChannelSqueezeExcitation("se0", channels, 2, act_type))
                elif se_type == "GE+":
                    self.body.add(_GatherExcitePlus("ge0", channels, 2, act_type))
                else:
                    raise Exception('Unsupported Squeeze Excitation Module: Choose either [None, "cSE", "sSE", "scSE"')

            self.body.add(Conv2D(channels_expanded, kernel_size=1, padding=0, use_bias=False))
            self.body.add(BatchNorm(momentum=bn_mom))
            self.body.add(get_act(act_type))
            self.body.add(Conv2D(channels_expanded, kernel_size=kernel_size, padding=kernel_size//2,
                                 groups=channels_expanded, use_bias=False))
            self.body.add(BatchNorm(momentum=bn_mom))
            self.body.add(get_act(act_type))

            # # add a 1x1 branch after concatenation
            self.body.add(Conv2D(channels=channels, kernel_size=1, padding=0, use_bias=False))
            self.body.add(BatchNorm(momentum=bn_mom))

    def hybrid_forward(self, F, x):
        """
        Compute forward pass

        :param F: Handle
        :param x: Input data to the block
        :return: Activation maps of the block
        """

        shortcut = x

        out = self.body(x)

        # connect the shortcut with the residual activations
        out = shortcut + out

        # apply activation
        return out


class _ResidualBlockXBottleneck(HybridBlock):  # Too many arguments (9/5)
    """
    Definition of a residual block without any pooling operation
    """

    def __init__(self, unit_name, channels, bn_mom=0.9, act_type="relu", se_type="scSE", dim_match=True):
        """

        :param channels: Number of channels used in the conv-operations
        :param bn_mom: Batch normalization momentum
        :param act_type: Activation function to use
        :param unit_name: Unit name of the residual block (only used for description (string))
        """
        super(_ResidualBlockXBottleneck, self).__init__(prefix=unit_name + "_")
        self.unit_name = unit_name
        self.use_se = se_type
        self.dim_match = dim_match
        self.body = HybridSequential(prefix="")

        with self.name_scope():

            if se_type:
                if se_type == "cSE":
                    # apply squeeze excitation
                    self.body.add(_ChannelSqueezeExcitation("se0", channels, 2, act_type))
                elif se_type == "sSE":
                    self.body.add(_SpatialSqueezeExcitation("se0"))
                elif se_type == "scSE":
                    self.body.add(_SpatialChannelSqueezeExcitation("se0", channels, 2, act_type))
                elif se_type == "GE+":
                    self.body.add(_GatherExcitePlus("ge0", channels, 2, act_type))
                else:
                    raise Exception('Unsupported Squeeze Excitation Module: Choose either [None, "cSE", "sSE", "scSE"')

            self.body.add(BatchNorm(momentum=bn_mom))
            self.body.add(get_act(act_type))
            self.body.add(Conv2D(channels, kernel_size=1, padding=0, use_bias=False))
            self.body.add(BatchNorm(momentum=bn_mom))
            self.body.add(get_act(act_type))
            self.body.add(Conv2D(channels, kernel_size=3, padding=1, groups=2, use_bias=False))
            self.body.add(BatchNorm(momentum=bn_mom))
            self.body.add(get_act(act_type))
            #
            # # add a 1x1 branch after concatenation
            self.body.add(Conv2D(channels=channels, kernel_size=1, padding=0, use_bias=False))

            if not self.dim_match:
                self.expander = HybridSequential(prefix="")
                self.expander.add(Conv2D(channels=channels, kernel_size=1, use_bias=False, prefix="expander_conv"))
                self.expander.add(BatchNorm())

    def hybrid_forward(self, F, x):
        """
        Compute forward pass

        :param F: Handle
        :param x: Input data to the block
        :return: Activation maps of the block
        """

        if self.dim_match:
            shortcut = x
        else:
            shortcut = self.expander(x)

        out = self.body(x)

        # connect the shortcut with the residual activations
        out = shortcut + out

        # apply activation
        return out


class _StemRise(HybridBlock):
    def __init__(self, name, channels, bn_mom=0.9, act_type="relu", se_type=None):
        """
        Definition of the stem proposed by the alpha zero authors

        :param name: name prefix for all blocks
        :param channels: Number of channels for 1st conv operation
        :param bn_mom: Batch normalization momentum parameter
        :param act_type: Activation type to use
        """

        super(_StemRise, self).__init__(prefix=name + "_")

        self.body = HybridSequential(prefix="")

        with self.name_scope():
            # add all layers to the stem
            self.body.add(Conv2D(channels=channels, kernel_size=(3, 3), padding=(1, 1), use_bias=False))
            self.body.add(BatchNorm(momentum=bn_mom))
            self.body.add(get_act(act_type))

    def hybrid_forward(self, F, x):
        """
        Compute forward pass

        :param F: Handle
        :param x: Input data to the block
        :return: Activation maps of the block
        """
        return self.body(x)


class Rise(HybridBlock):  # Too many arguments (15/5)
    """ Implementing the RISE architecture for learning chess proposed by Johannes Czech"""

    def __init__(
        self,
        n_labels=2272,
        channels=256,
        channels_value_head=8,
        channels_policy_head=16,
        nb_res_blocks_x=7,
        nb_res_blocks_x_neck=12,
        expand_res_blocks=None,
        value_fc_size=256,
        bn_mom=0.9,
        act_type="relu",
        squeeze_excitation_type=None,
        select_policy_from_plane=False,
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

        super(Rise, self).__init__(**kwargs, prefix="")
        self.body = HybridSequential(prefix="")

        with self.name_scope():
            se_type = None

            if use_rise_stem:
                self.body.add(_StemRise(name="stem", channels=channels, bn_mom=bn_mom, act_type=act_type))
            else:
                self.body.add(
                    _StemAlphaZero(name="stem", channels=channels, bn_mom=bn_mom, act_type=act_type, se_type=se_type)
                )

        channels = channels

        for i in range(nb_res_blocks_x):
            unit_name = "unit%d" % i

            if squeeze_excitation_type is None:
                se_type = None
            elif squeeze_excitation_type in ["cSE", "sSE", "scSE", "GE+"]:
                se_type = squeeze_excitation_type
            elif squeeze_excitation_type == "mixed":
                if i % 2 == 0:
                    se_type = "cSE"
                else:
                    se_type = "sSE"
            else:
                raise Exception("Unavailable SE type given.")

            if i < nb_res_blocks_x - 7 or i >= nb_res_blocks_x - 2:
                se_type = None

            self.body.add(ResidualBlockX(unit_name, channels=channels, bn_mom=0.9, act_type=act_type, se_type=se_type))

        for i in range(nb_res_blocks_x_neck):
            unit_name = "unitX%d" % i
            dim_match = True

            self.body.add(
                _ResidualBlockXBottleneck(
                    unit_name, channels, dim_match=dim_match, bn_mom=0.9, act_type=act_type,
                    se_type=squeeze_excitation_type
                )
            )

        channels_expanded = channels // 2
        for i, kernel_size in enumerate(expand_res_blocks):
            unit_name = "unit%d" % i
            se_type = squeeze_excitation_type

            if i < len(expand_res_blocks) - 5: # or i >= nb_res_blocks_x - 2:
                se_type = None

            self.body.add(
                _ExpanderResidualBlock(
                    unit_name, channels, channels_expanded=channels_expanded, kernel_size=kernel_size, bn_mom=0.9,
                    act_type=act_type, se_type=se_type
                )
            )
            channels_expanded += 64

        # create the two heads which will be used in the hybrid fwd pass
        self.value_head = _ValueHeadRise("value", channels_value_head, value_fc_size, bn_mom, act_type)
        self.policy_head = _PolicyHeadRise(
            "policy", channels_policy_head, n_labels, bn_mom, act_type, select_policy_from_plane
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


flat_plane_idx_const = mx.nd.array(FLAT_PLANE_IDX)


class _PolicyHeadRise(HybridBlock):  # Too many arguments (6/5) (too-many-arguments)
    def __init__(self, name, channels=2, n_labels=4992, bn_mom=0.9, act_type="relu",
                 select_policy_from_plane=False):
        """
        Definition of the value head. Same as alpha zero authors but changed order of Batch-Norm with RElu.

        :param name: name prefix for all blocks
        :param channels: Number of channels for 1st conv operation in branch 0
        :param bn_mom: Batch normalization momentum parameter
        :param act_type: Activation type to use
        :param se_type: SqueezeExcitation type choose either [None, "cSE", "sSE", csSE"] for no squeeze excitation,
        channelwise squeeze excitation, channel-spatial-squeeze-excitation, respectively
        """

        super(_PolicyHeadRise, self).__init__(prefix=name + "_")

        self.body = HybridSequential(prefix="")
        self.select_policy_from_plane = select_policy_from_plane

        with self.name_scope():
            if self.select_policy_from_plane:
                self.body.add(Conv2D(channels=256, padding=1, kernel_size=(3, 3), use_bias=False))
                self.body.add(BatchNorm(momentum=bn_mom))
                self.body.add(get_act(act_type))
                self.body.add(Conv2D(channels=channels, padding=1, kernel_size=(3, 3), use_bias=False))
                self.body.add(Flatten())
            else:
                self.body.add(Conv2D(channels=channels, kernel_size=(1, 1), use_bias=False))
                self.body.add(BatchNorm(momentum=bn_mom))
                self.body.add(get_act(act_type))
                # if not self.select_policy_from_plane:

                self.body.add(Flatten())
                self.body.add(Dense(units=n_labels))

    def hybrid_forward(self, F, x):
        """
        Compute forward pass

        :param F: Handle
        :param x: Input data to the block
        :return: Activation maps of the block
        """
        out = self.body(x)
        return out


class _ValueHeadRise(HybridBlock):  # Too many arguments (6/5) (too-many-arguments)
    def __init__(self, name, channels=1, fc0=256, bn_mom=0.9, act_type="relu"):
        """
        Definition of the value head. Same as alpha zero authors but changed order Batch-Norm with RElu.

        :param name: name prefix for all blocks
        :param channels: Number of channels for 1st conv operation in branch 0
        :param fc0: Number of units in Dense/Fully-Connected layer
        :param bn_mom: Batch normalization momentum parameter
        :param act_type: Activation type to use
        :param se_type: SqueezeExcitation type choose either [None, "cSE", "sSE", csSE"] for no squeeze excitation,
        channelwise squeeze excitation, channel-spatial-squeeze-excitation, respectively
        """

        super(_ValueHeadRise, self).__init__(prefix=name + "_")

        self.body = HybridSequential(prefix="")

        with self.name_scope():
            self.body.add(Conv2D(channels=channels, kernel_size=(1, 1), use_bias=False))
            self.body.add(BatchNorm(momentum=bn_mom))
            self.body.add(get_act(act_type))
            self.body.add(Flatten())
            self.body.add(Dense(units=fc0))
            self.body.add(get_act(act_type))
            self.body.add(Dense(units=1))
            self.body.add(get_act("tanh"))

    def hybrid_forward(self, F, x):
        """
        Compute forward pass

        :param F: Handle
        :param x: Input data to the block
        :return: Activation maps of the block
        """
        return self.body(x)
