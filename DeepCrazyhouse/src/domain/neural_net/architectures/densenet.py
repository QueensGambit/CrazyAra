"""
@file: densenet.py
Created on 03.04.19
@project: CrazyAra
@author: queensgambit

Adaption of the Desnet architecture to process crazyhouse or chess games
Densenet - Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>.
introduced by Gao Huang, Zhuang Liu, Laurens van der Maaten.

The naive implementation is known to suffer from high memory requirements.
"""


from mxnet.gluon.model_zoo.vision.densenet import _make_dense_block
from mxnet.gluon.block import HybridBlock
import mxnet.gluon.nn as nn

from DeepCrazyhouse.src.domain.neural_net.architectures.a0_resnet import _ValueHeadAlphaZero, _PolicyHeadAlphaZero


class DenseNet(HybridBlock):
    """
    Definition of Densenet bottlekneck architecture
    """

    def __init__(
        self,
        channels_init=64,
        growth_rate=32,
        n_layers=10,
        bottleneck_factor=4,
        dropout=0,
        n_labels=1000,
        channels_value_head=8,
        channels_policy_head=16,
        value_fc_size=256,
        **kwargs
    ):
        """
        Constructor
        :param channels_init: Number of channels for the first convolutional layer
        :param growth_rate: Number of channels which increase per layer
        :param n_layers: Number of layers
        :param bottleneck_factor: Bottleneck factor which determines how much more layers used for the 1x1 convolution
        :param dropout: Dropout factor, if 0% then no dropout will be used
        :param n_labels: Number of final labels to predict, here moves
        :param channels_value_head: Number of channels in the final value head
        :param channels_policy_head: Number of channels in the final policy head
        :param value_fc_size: Size of the fully connected layer in the value head
        :param kwargs: Optional additional arguments
        """

        super(DenseNet, self).__init__(**kwargs)

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            # add initial convolutional layer
            self.features.add(nn.Conv2D(channels_init, kernel_size=3, padding=1, use_bias=False))

            # add dense blocks
            for layer_idx in range(n_layers):
                self.features.add(_make_dense_block(n_layers, bottleneck_factor, growth_rate, dropout, layer_idx))

            # we need to add a batch-norm and activation because _make_dense_block() starts with them
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation("relu"))

        # create the two heads which will be used in the hybrid fwd pass
        self.value_head = _ValueHeadAlphaZero("value", channels_value_head, value_fc_size, 0.9, "relu")
        self.policy_head = _PolicyHeadAlphaZero("policy", channels_policy_head, n_labels, 0.9, "relu")

    def hybrid_forward(self, F, x):
        """
        Implementation of the forward pass of the full network
        :param F: Abstract Function handle which works for gluon & mxnet
        :param x: Input to the ResidualBlock
        :return: Value & Policy Output
        """
        out = self.features(x)
        value = self.value_head(out)
        policy = self.policy_head(out)
        return [value, policy]
