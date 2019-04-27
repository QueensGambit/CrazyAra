"""
@file: wide_resnet_se.py
Created on 07.04.19
@project: CrazyAra
@author: queensgambit

Please describe what the content of this file is about
"""
from mxnet.gluon import HybridBlock
from mxnet.gluon.nn import HybridSequential

from DeepCrazyhouse.src.domain.neural_net.architectures.a0_resnet import _ValueHeadAlphaZero, _PolicyHeadAlphaZero, \
    _StemAlphaZero
from DeepCrazyhouse.src.domain.neural_net.architectures.rise import ResidualBlockX


class WideResnetSE(HybridBlock):
    """ Implementation of adapted version of WideResnet with Squeeze-Excitation Layer"""

    def __init__(
        self,
        nb_input_channels=34,
        n_labels=2272,
        channels=512,
        channels_value_head=4,
        channels_policy_head=8,
        nb_res_blocks=6,
        value_fc_size=512,
        bn_mom=0.9,
        act_type="relu",
        use_se=True,
        **kwargs
    ):  # Too many local variables (22/15)
        """
        Creates the alpha zero gluon net description based on the given parameters.
        :param nb_input_channels: Number of input channels of the board representation (only needed for the first SE)
        :param n_labels: Number of labels the for the policy
        :param channels: Used for all convolution operations. (Except the last 2)
        :param nb_res_blocks_x: Number of residual blocks to stack. In the paper they used 19 or 39 residual blocks
        :param value_fc_size: Fully Connected layer size. Used for the value output
        :param bn_mom: Batch normalization momentum
        :return: gluon net description
        """

        super(WideResnetSE, self).__init__(**kwargs, prefix="")
        self.body = HybridSequential(prefix="")

        with self.name_scope():

            # activate squeeze excitation layers if needed
            if use_se:
                # use the combination of channel and spatial excitation because it's almost for free
                # with a low amount of channels
                se_type = "csSE"
            else:
                se_type = None

            # add the initial convolutional layer
            self.body.add(_StemAlphaZero(name="stem", channels=channels, bn_mom=bn_mom, act_type=act_type,
                                         se_type=se_type, nb_input_channels=nb_input_channels))

            for i in range(nb_res_blocks):
                unit_name = "unit%d" % i

                # add all the residual blocks
                self.body.add(
                    ResidualBlockX(
                        unit_name,
                        channels=channels,
                        bn_mom=0.9,
                        act_type=act_type,
                        se_type=None,  # deactivate SE for all middle layers because to reduce computation cost
                    )
                )

            # create the two heads which will be used in the hybrid fwd pass
            self.value_head = _ValueHeadAlphaZero("value", channels_value_head, value_fc_size, bn_mom, act_type, se_type)
            self.policy_head = _PolicyHeadAlphaZero("policy", channels_policy_head, n_labels, bn_mom, act_type, se_type)

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
