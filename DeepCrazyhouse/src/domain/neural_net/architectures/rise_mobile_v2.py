"""
@file: rise_symbol.py
Created on 18.05.19
@project: CrazyAra
@author: queensgambit

RISEv2 - mobile
Updated model architecture of RISE optimized for CPU usage proposed by Johannes Czech 2019.
Additionally, it combines techniques presented in the following papers:

MobileNetV2: Inverted Residuals and Linear Bottlenecks
https://arxiv.org/pdf/1801.04381.pdf

Deep Pyramidal Residual Networks
https://arxiv.org/abs/1610.02915

Squeeze-and-Excitation Networks
https://arxiv.org/abs/1709.01507

The architecture is explained in more detail in the CrazyAra paper.
The network is up 3 times faster on CPU and 1.4 faster on GPU with similar performance.
Recent cuDNN version also improved the speed on GPU to factor of 3.
"""

import mxnet as mx
from DeepCrazyhouse.src.domain.neural_net.architectures.builder_util_symbol import get_act, channel_squeeze_excitation,\
    get_stem, policy_head, value_head
from DeepCrazyhouse.src.domain.variants.constants import NB_CHANNELS_TOTAL, NB_CHANNELS_VARIANTS


def bottleneck_residual_block(data, channels, channels_operating, name, kernel=3, act_type='relu', use_se=False,
                              data_variant=None):
    """
    Returns a residual block without any max pooling operation
    :param data: Input data
    :param channels: Number of filters for all CNN-layers
    :param channels_operating: Number of filters used for 3x3, 5x5, 7x7,.. convolution
    :param name: Name for the residual block
    :param act_type: Activation function to use
    :param use_se: If true, a squeeze excitation will be used
    :param data_variant: Data input which holds the current active variant information
    :return:
    """

    if data_variant is not None:
        first_input = mx.sym.Concat(*[data, data_variant], name=name + '_concat')
        add_channels = NB_CHANNELS_VARIANTS
    else:
        first_input = data
        add_channels = 0

    if use_se:
        se = channel_squeeze_excitation(first_input, channels+add_channels, name=name + '_se', ratio=2)
        conv1 = mx.sym.Convolution(data=se, num_filter=channels_operating, kernel=(1, 1), pad=(0, 0),
                                   no_bias=True, name=name + '_conv1')
    else:
        conv1 = mx.sym.Convolution(data=first_input, num_filter=channels_operating, kernel=(1, 1), pad=(0, 0),
                                   no_bias=True, name=name + '_conv1')
    bn1 = mx.sym.BatchNorm(data=conv1, name=name + '_bn1')
    act1 = get_act(data=bn1, act_type=act_type, name=name + '_act1')
    conv2 = mx.sym.Convolution(data=act1, num_filter=channels_operating, kernel=(kernel, kernel), stride=(1, 1),
                               num_group=channels_operating, pad=(kernel // 2, kernel // 2), no_bias=True,
                               name=name + '_conv2')
    bn2 = mx.sym.BatchNorm(data=conv2, name=name + '_bn2')
    act2 = get_act(data=bn2, act_type=act_type, name=name + '_act2')
    conv3 = mx.sym.Convolution(data=act2, num_filter=channels, kernel=(1, 1), pad=(0, 0),
                               no_bias=True, name=name + '_conv3')
    bn3 = mx.sym.BatchNorm(data=conv3, name=name + '_bn3')
    sum = mx.sym.broadcast_add(bn3, data, name=name+'_add')

    return sum


def residual_block(data, channels, name, kernel=3, act_type='relu', use_se=False):
    """
    Returns a residual block without any max pooling operation
    :param data: Input data
    :param channels: Number of filters for all CNN-layers
    :param name: Name for the residual block
    :param act_type: Activation function to use
    :param use_se: If true, a squeeze excitation will be used
    :return:
    """

    if use_se:
        se = channel_squeeze_excitation(data, channels, name=name + '_se', ratio=2, act_type=act_type)
        conv1 = mx.sym.Convolution(data=se, num_filter=channels, kernel=(kernel, kernel),
                                   pad=(kernel // 2, kernel // 2), num_group=1,
                                   no_bias=True, name=name + '_conv1')
    else:
        conv1 = mx.sym.Convolution(data=data, num_filter=channels, kernel=(kernel, kernel),
                                   pad=(kernel // 2, kernel // 2), num_group=1,
                                   no_bias=True, name=name + '_conv1')
    act1 = get_act(data=conv1, act_type=act_type, name=name + '_act1')
    bn1 = mx.sym.BatchNorm(data=act1, name=name + '_bn1')

    # kernel = 3
    conv2 = mx.sym.Convolution(data=bn1, num_filter=channels, kernel=(kernel, kernel), stride=(1, 1),
                               num_group=1, pad=(kernel // 2, kernel // 2), no_bias=True,
                               name=name + '_conv2')
    bn2 = mx.sym.BatchNorm(data=conv2, name=name + '_bn2')

    sum = mx.sym.broadcast_add(data, bn2, name=name + '_add')

    return sum


def preact_residual_block(data, channels, name, kernel=3, act_type='relu'):
    """
    Returns a residual block without any max pooling operation
    :param data: Input data
    :param channels: Number of filters for all CNN-layers
    :param name: Name for the residual block
    :param act_type: Activation function to use
    :return:
    """

    bn1 = mx.sym.BatchNorm(data=data, name=name + '_bn1')
    conv1 = mx.sym.Convolution(data=bn1, num_filter=channels, kernel=(kernel, kernel),
                               pad=(kernel // 2, kernel // 2), num_group=1,
                               no_bias=True, name=name + '_conv1')
    bn2 = mx.sym.BatchNorm(data=conv1, name=name + '_bn2')
    act1 = get_act(data=bn2, act_type=act_type, name=name + '_act1')
    conv2 = mx.sym.Convolution(data=act1, num_filter=channels, kernel=(kernel, kernel), stride=(1, 1),
                               pad=(kernel // 2, kernel // 2), no_bias=True,
                               name=name + '_conv2')
    sum = mx.sym.broadcast_add(data, conv2, name=name + '_add')

    return sum


def extract_variant_info(data, channels, name):
    """
    Extracts the variant channel information from the input data.
    Assumed to be the last 9 channels (where 9 is NB_CHANNEL_VARIANTS)
    ;oaram data: Input data
    :param channels: Number of channels of the input data
    :param name: Name of the block
    :return Symbol for the variant channel information
    """

    variant_layers = []

    for idx, xi in enumerate(mx.sym.split(data, axis=1, num_outputs=channels, name=name + '_split')):
        if idx >= NB_CHANNELS_TOTAL - NB_CHANNELS_VARIANTS:
            variant_layers.append(xi)
    return mx.sym.Concat(*variant_layers, name=name + '_concat')


def rise_mobile_v2_symbol(channels=256, channels_operating_init=128, channel_expansion=64, channels_value_head=8,
                          channels_policy_head=81, value_fc_size=256, bc_res_blocks=[], res_blocks=[3]*13, act_type='relu',
                          n_labels=4992, grad_scale_value=0.01, grad_scale_policy=0.99, select_policy_from_plane=True,
                          use_se=True, dropout_rate=0, use_extra_variant_input=False):
    """
    Creates the rise mobile model symbol based on the given parameters.

    :param channels: Used for all convolution operations. (Except the last 2)
    :param workspace: Parameter for convolution
    :param value_fc_size: Fully Connected layer size. Used for the value output
    :param num_res_blocks: Number of residual blocks to stack. In the paper they used 19 or 39 residual blocks
    :param bn_mom: batch normalization momentum
    :param act_type: Activation function which will be used for all intermediate layers
    :param n_labels: Number of labels the for the policy
    :param grad_scale_value: Constant scalar which the gradient for the value outputs are being scaled width.
                            (They used 1.0 for default and 0.01 in the supervised setting)
    :param grad_scale_policy: Constant scalar which the gradient for the policy outputs are being scaled width.
                            (They used 1.0 for default and 0.99 in the supervised setting)
    :param dropout_rate: Applies optionally droput during learning with a given factor on the last feature space before
    :param use_extra_variant_input: If true, the last 9 channel which represent the active variant are passed to each
    residual block separately and concatenated at the end of the final feature representation
    branching into value and policy head
    :return: mxnet symbol of the model
    """
    # get the input data
    data = mx.sym.Variable(name='data')

    if use_extra_variant_input:
        data_variant = extract_variant_info(data, channels=NB_CHANNELS_TOTAL, name="variants")
    else:
        data_variant = None

    # first initial convolution layer followed by batchnormalization
    body = get_stem(data=data, channels=channels, act_type=act_type)
    channels_operating = channels_operating_init

    # build residual tower
    for idx, kernel in enumerate(bc_res_blocks):
        use_squeeze_excitation = use_se

        if idx < len(bc_res_blocks) - 5:
            use_squeeze_excitation = False
        body = bottleneck_residual_block(body, channels, channels_operating, name='bc_res_block%d' % idx, kernel=kernel,
                                         use_se=use_squeeze_excitation, act_type=act_type, data_variant=data_variant)
        channels_operating += channel_expansion

    for idx, kernel in enumerate(res_blocks):
        if idx < len(res_blocks) - 5:
            use_squeeze_excitation = False
        else:
            use_squeeze_excitation = use_se

        body = residual_block(body, channels, name='res_block%d' % idx, kernel=kernel,
                              use_se=use_squeeze_excitation, act_type=act_type)

    if dropout_rate != 0:
        body = mx.sym.Dropout(body, p=dropout_rate)

    if use_extra_variant_input:
        body = mx.sym.Concat(*[body, data_variant], name='feature_concat')

    # for policy output
    policy_out = policy_head(data=body, channels=channels, act_type=act_type, channels_policy_head=channels_policy_head,
                             select_policy_from_plane=select_policy_from_plane, n_labels=n_labels,
                             grad_scale_policy=grad_scale_policy, use_se=False, no_bias=True)

    # for value output
    value_out = value_head(data=body, channels_value_head=channels_value_head, value_kernelsize=1, act_type=act_type,
                           value_fc_size=value_fc_size, grad_scale_value=grad_scale_value, use_se=False,
                           use_mix_conv=False)

    # group value_out and policy_out together
    sym = mx.symbol.Group([value_out, policy_out])

    return sym


def get_rise_v2_symbol(channels_policy_head, n_labels, select_policy_from_plane, val_loss_factor, policy_loss_factor):
    """
    Wrapper definition for RISEv2.0.
    :return: symbol
    """
    bc_res_blocks = [3] * 13
    symbol = rise_mobile_v2_symbol(channels=256, channels_operating_init=128, channel_expansion=64,
                                   channels_value_head=8,
                                   channels_policy_head=channels_policy_head, value_fc_size=256,
                                   bc_res_blocks=bc_res_blocks, res_blocks=[], act_type='relu',
                                   n_labels=n_labels, grad_scale_value=val_loss_factor,
                                   grad_scale_policy=policy_loss_factor,
                                   select_policy_from_plane=select_policy_from_plane,
                                   use_se=True, dropout_rate=0,
                                   use_extra_variant_input=False)
    return symbol


def preact_resnet_symbol(channels=256, channels_value_head=8,
                   channels_policy_head=81, value_fc_size=256, value_kernelsize=7, res_blocks=19, act_type='relu',
                   n_labels=4992, grad_scale_value=0.01, grad_scale_policy=0.99, select_policy_from_plane=True):
    """
    Creates the alpha zero model symbol based on the given parameters.

    :param channels: Used for all convolution operations. (Except the last 2)
    :param workspace: Parameter for convolution
    :param value_fc_size: Fully Connected layer size. Used for the value output
    :param num_res_blocks: Number of residual blocks to stack. In the paper they used 19 or 39 residual blocks
    :param bn_mom: batch normalization momentum
    :param act_type: Activation function which will be used for all intermediate layers
    :param n_labels: Number of labels the for the policy
    :param grad_scale_value: Constant scalar which the gradient for the value outputs are being scaled width.
                            (0.01 is recommended for supervised learning with little data)
    :param grad_scale_policy: Constant scalar which the gradient for the policy outputs are being scaled width.
    :return: mxnet symbol of the model
    """
    # get the input data
    data = mx.sym.Variable(name='data')

    body = get_stem(data=data, channels=channels, act_type=act_type)

    for idx in range(res_blocks):
        body = preact_residual_block(body, channels, name='res_block%d' % idx, kernel=3,
                              act_type=act_type)

    body = mx.sym.BatchNorm(data=body, name='stem_bn1')
    body = get_act(data=body, act_type=act_type, name='stem_act1')

    # for policy output
    policy_out = policy_head(data=body, channels=channels, act_type=act_type, channels_policy_head=channels_policy_head,
                             select_policy_from_plane=select_policy_from_plane, n_labels=n_labels,
                             grad_scale_policy=grad_scale_policy, use_se=False, no_bias=True)

    # for value output
    value_out = value_head(data=body, channels_value_head=channels_value_head, value_kernelsize=1, act_type=act_type,
                           value_fc_size=value_fc_size, grad_scale_value=grad_scale_value, use_se=False,
                           use_mix_conv=False)

    # group value_out and policy_out together
    sym = mx.symbol.Group([value_out, policy_out])

    return sym
