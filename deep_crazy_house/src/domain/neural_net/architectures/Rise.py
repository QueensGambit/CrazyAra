"""
@file: Rise.py
Created on 26.09.18
@project: crazy_ara_refactor
@author: queensgambit

Description of the RISE architecture for learning chess proposed by Johannes Czech in 2018 for CrazyAra 0.2.0.
The architecture incorporates new ideas and techniques described in recent papers for Deep Learning in Computer Vision.

R - ResneXt (Aggregated Residual Transformations for Deep Neural Networks, Xie et al., http://arxiv.org/abs/1611.05431)
            (Deep Residual Learning for Image Recognition, He et al., https://arxiv.org/pdf/1512.03385.pdf)
I - Inception (Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,  Szegedy et al., https://arxiv.org/pdf/1602.07261.pdf)
              (Rethinking the Inception Architecture for Computer Vision - https://arxiv.org/pdf/1512.00567.pdf, Szegedy et al., https://arxiv.org/pdf/1512.00567.pdf)
S - Squeeze (Squeeze-and-Excitation Networks,  Xie et al., https://arxiv.org/pdf/1709.01507.pdf)
E - Excitation

The proposed model architecture has fewer parameters, faster inference and training time while maintaining an equal amount of depth
compared to the architecture proposed by DeepMind (19 residual layers with 256 filters).
On our 10,000 games benchmark dataset it achieved a lower validation error.
"""

from mxnet.gluon.nn import HybridSequential, Conv2D, BatchNorm
from mxnet.gluon import HybridBlock
from deep_crazy_house.src.domain.neural_net.architectures.builder_util import get_act
from deep_crazy_house.src.domain.neural_net.architectures.rise_builder_util import _SqueezeExcitation
from deep_crazy_house.src.domain.neural_net.architectures.AlphaZeroResnet import (
    _StemAlphaZero,
    _PolicyHeadAlphaZero,
    _ValueHeadAlphaZero,
)


class ResidualBlockX(HybridBlock):
    """
    Definition of a residual block without any pooling operation
    """

    def __init__(self, unit_name, cardinality, channels, bn_mom, act_type, res_scale_fac, use_se=True):
        """

        :param channels: Number of channels used in the conv-operations
        :param bn_mom: Batch normalization momentum
        :param act_type: Activation function to use
        :param unit_name: Unit name of the residual block (only used for description (string))
        """
        super(ResidualBlockX, self).__init__(unit_name + "_")
        self.act_type = act_type
        self.unit_name = unit_name
        self.res_scale_fac = res_scale_fac
        self.body = HybridSequential(prefix="")

        with self.name_scope():
            self.body.add(Conv2D(channels=channels, kernel_size=3, padding=1, use_bias=False))
            self.body.add(BatchNorm(momentum=bn_mom))
            self.body.add(get_act(self.act_type))

            self.body.add(Conv2D(channels=channels, kernel_size=3, padding=1, groups=cardinality, use_bias=False))
            self.body.add(BatchNorm(momentum=bn_mom))

            if use_se is True:
                # apply squeeze excitation
                self.body.add(_SqueezeExcitation("se0", channels, 16, act_type))

            self.act0 = get_act(act_type)

    def hybrid_forward(self, F, x):
        """
        Implemntation of the forward pass of the residual block.
        Uses a broadcast add operation for the shortcut and the output of the residual block
        :param F: Abstract Function handle which works for gluon & mxnet
        :param x: Input to the ResidualBlock
        :return: Sum of the shortcut and the computed residual block computation
        """
        shortcut = x

        out = self.body(x)

        # scale down the output of the residual block activations to stabilize training
        if self.res_scale_fac is not None:
            out = shortcut + out * self.res_scale_fac
        else:
            # connect the shortcut with the residual activations
            out = shortcut + out

        # apply activation
        out = self.act0(out)

        return out


class _ResidualBlockXBottleneck(HybridBlock):
    """
    Definition of a residual block without any pooling operation
    """

    def __init__(
        self,
        unit_name,
        cardinality,
        channels,
        bn_mom=0.9,
        act_type="relu",
        use_se=True,
        res_scale_fac=0.2,
        dim_match=True,
    ):
        """

        :param channels: Number of channels used in the conv-operations
        :param bn_mom: Batch normalization momentum
        :param act_type: Activation function to use
        :param unit_name: Unit name of the residual block (only used for description (string))
        """
        super(_ResidualBlockXBottleneck, self).__init__(prefix=unit_name + "_")
        self.unit_name = unit_name
        self.res_scale_fac = res_scale_fac

        self.use_se = use_se
        self.dim_match = dim_match
        self.body = HybridSequential(prefix="")

        with self.name_scope():
            self.body.add(get_act(act_type))
            self.body.add(Conv2D(int(channels // 2), kernel_size=1, padding=0, use_bias=False))
            self.body.add(BatchNorm(momentum=bn_mom))
            self.body.add(get_act(act_type))
            self.body.add(Conv2D(int(channels // 2), kernel_size=3, padding=1, groups=cardinality, use_bias=False))
            self.body.add(BatchNorm(momentum=bn_mom))
            self.body.add(get_act(act_type))

            # add a 1x1 branch after concatenation
            self.body.add(Conv2D(channels=channels, kernel_size=1, use_bias=False))
            self.body.add(BatchNorm(momentum=bn_mom))

            if use_se is True:
                # apply squeeze excitation
                self.body.add(_SqueezeExcitation("se0", channels, 16, act_type))

            self.act0 = get_act(act_type, prefix="%s1" % act_type)

            if self.dim_match is False:
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

        if self.dim_match is True:
            shortcut = x
        else:
            shortcut = self.expander(x)

        out = self.body(x)

        # scale down the output of the residual block activations to stabilize training
        if self.res_scale_fac is not None:
            out = shortcut + out * self.res_scale_fac
        else:
            # connect the shortcut with the residual activations
            out = shortcut + out

        # apply activation
        out = self.act0(out)

        return out


class _StemRise(HybridBlock):
    def __init__(self, name, channels, bn_mom=0.9, act_type="relu", use_se=False):
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
            self.body.add(Conv2D(channels=64, kernel_size=(3, 3), padding=(1, 1), use_bias=False))
            self.body.add(BatchNorm(momentum=bn_mom))
            self.body.add(get_act(act_type))
            self.body.add(Conv2D(channels=64, kernel_size=(3, 3), padding=(1, 1), use_bias=False))
            self.body.add(BatchNorm(momentum=bn_mom))
            self.body.add(get_act(act_type))
            self.body.add(Conv2D(channels=128, kernel_size=(3, 3), padding=(1, 1), use_bias=False))
            self.body.add(BatchNorm(momentum=bn_mom))
            self.body.add(get_act(act_type))
            self.body.add(Conv2D(channels=128, kernel_size=(3, 3), padding=(1, 1), use_bias=False))
            self.body.add(BatchNorm(momentum=bn_mom))
            self.body.add(get_act(act_type))
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
        out = self.body(x)

        return out


class Rise(HybridBlock):
    def __init__(
        self,
        n_labels=2272,
        channels=256,
        channels_value_head=8,
        channels_policy_head=16,
        nb_res_blocksX=7,
        nb_res_blocksX_neck=12,
        cardinality=1,
        cardinality_neck=2,
        value_fc_size=256,
        bn_mom=0.9,
        act_type="lrelu",
        use_se=True,
        res_scale_fac=0.2,
        use_rise_stem=False,
        **kwargs
    ):
        """
        Creates the alpha zero gluon net description based on the given parameters.

        :param n_labels: Number of labels the for the policy
        :param channels: Used for all convolution operations. (Except the last 2)
        :param nb_res_blocksX: Number of residual blocks to stack. In the paper they used 19 or 39 residual blocks
        :param value_fc_size: Fully Connected layer size. Used for the value output
        :param bn_mom: Batch normalization momentum
        :return: gluon net description
        """

        super(Rise, self).__init__(**kwargs, prefix="")

        self.body = HybridSequential(prefix="")

        with self.name_scope():
            if use_rise_stem is True:
                self.body.add(_StemRise(name="stem", channels=channels, bn_mom=bn_mom, act_type=act_type, use_se=False))
            else:
                self.body.add(_StemAlphaZero(name="stem", channels=channels, bn_mom=bn_mom, act_type=act_type))

        for i in range(nb_res_blocksX):
            unit_name = "unit%d" % i
            self.body.add(
                ResidualBlockX(
                    unit_name,
                    cardinality=cardinality,
                    channels=channels,
                    bn_mom=0.9,
                    act_type=act_type,
                    res_scale_fac=res_scale_fac,
                    use_se=False,
                )
            )

        for i in range(nb_res_blocksX_neck):
            unit_name = "unitX%d" % i

            dim_match = True

            # deactivate the SE for the last two blocks
            if i == nb_res_blocksX_neck - 2 or i == nb_res_blocksX_neck - 1:
                cur_use_se = False
                cur_res_scale_fac = res_scale_fac
            else:
                cur_use_se = use_se
                cur_res_scale_fac = res_scale_fac

            self.body.add(
                _ResidualBlockXBottleneck(
                    unit_name,
                    cardinality_neck,
                    channels,
                    dim_match=dim_match,
                    bn_mom=0.9,
                    act_type=act_type,
                    use_se=cur_use_se,
                    res_scale_fac=cur_res_scale_fac,
                )
            )

        # create the two heads which will be used in the hybrid fwd pass
        self.value_head = _ValueHeadAlphaZero("value", channels_value_head, value_fc_size, bn_mom, act_type)
        self.policy_head = _PolicyHeadAlphaZero("policy", channels_policy_head, n_labels, bn_mom, act_type)

    def hybrid_forward(self, F, x):
        """
        Implemntation of the forward pass of the full network
        Uses a broadcast add operation for the shortcut and the output of the residual block
        :param F: Abstract Function handle which works for gluon & mxnet
        :param x: Input to the ResidualBlock
        :return: Value & Policy Output
        """
        out = self.body(x)

        value = self.value_head(out)
        policy = self.policy_head(out)

        return [value, policy]
