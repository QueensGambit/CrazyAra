"""
@file: mixture_net.py
Created on 17.02.20
@project: CrazyAra
@author: queensgambit

Experimental network which combines the output of multiple singular networks
 by a weighted sum of a learnable weighting parameters.

Proposed by Arseny Skryagin and Johannes Czech
"""
import mxnet as mx
from DeepCrazyhouse.src.domain.neural_net.architectures.builder_util_symbol import get_act, channel_squeeze_excitation, \
    mix_conv, get_stem, value_head, policy_head
from DeepCrazyhouse.src.domain.neural_net.architectures.mxnet_alpha_zero import residual_block


def mixture_net_symbol(channels=256, num_res_blocks=7, channels_operating_init=128, channel_expansion=64, act_type='relu',
                          channels_value_head=8, channels_policy_head=81, value_fc_size=256, dropout_rate=0.15,
                          grad_scale_value=0.01, grad_scale_policy=0.99,
                          select_policy_from_plane=True, use_se=True, kernels=None, n_labels=4992):
    """
    Mixture net
    :param channels: Main number of channels
    :param channels_operating_init: Initial number of channels at the start of the net for the depthwise convolution
    :param channel_expansion: Number of channels to add after each residual block
    :param act_type: Activation type to use
    :param channels_value_head: Number of channels for the value head
    :param value_fc_size: Number of units in the fully connected layer of the value head
    :param channels_policy_head: Number of channels for the policy head
    :param dropout_rate: Droput factor to use. If 0, no dropout will be applied. Value must be in [0,1]
    :param grad_scale_value: Constant scalar which the gradient for the value outputs are being scaled width.
                            (0.01 is recommended for supervised learning with little data)
    :param grad_scale_policy: Constant scalar which the gradient for the policy outputs are being scaled width.
    :param select_policy_from_plane: True, if policy head type shall be used
    :param use_se: Indicates if a squeeze excitation layer shall be used
    :param res_blocks: Number of residual blocks
    :param n_labels: Number of policy target labels (used for select_policy_from_plane=False)
    :return: symbol
    """
    # get the input data
    data = mx.sym.Variable(name='data')

    data = get_stem(data=data, channels=channels, act_type=act_type)

    nb_subnets = 3
    bodies = [None] * nb_subnets

    # build residual tower
    for z in range(len(bodies)):
        for i in range(num_res_blocks):
            if i == 0:
                bodies[z] = residual_block(data, channels, name='b%d_block%d' % (z,i),
                                      bn_mom=0.9, workspace=1024)
            else:
                bodies[z] = residual_block(bodies[z], channels, name='b%d_block%d' % (z,i),
                                      bn_mom=0.9, workspace=1024)

    w_a = mx.sym.Variable('w_a')
    w_b = mx.sym.Variable('w_b')
    w_c = mx.sym.Variable('w_c')

    data_value = w_a * bodies[0] + w_b * bodies[1] + w_c * bodies[2]
    data_policy = w_a * bodies[0] + w_b * bodies[1] + w_c * bodies[2]

    value_out = value_head(data=data_value, act_type=act_type, use_se=False, channels_value_head=channels_value_head,
                           value_fc_size=value_fc_size, use_mix_conv=False, grad_scale_value=grad_scale_value)
    policy_out = policy_head(data=data_policy, act_type=act_type, channels_policy_head=channels_policy_head, n_labels=n_labels,
                             select_policy_from_plane=select_policy_from_plane, use_se=False, channels=channels,
                             grad_scale_policy=grad_scale_policy)
    # group value_out and policy_out together
    sym = mx.symbol.Group([value_out, policy_out])

    return sym
