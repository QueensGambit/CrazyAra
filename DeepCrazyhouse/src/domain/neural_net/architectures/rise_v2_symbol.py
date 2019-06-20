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
"""

import mxnet as mx


def get_act(data, act_type, name):
    """Wrapper method for different non linear activation functions"""
    if act_type in ["relu", "sigmoid", "softrelu", "softsign", "tanh"]:
        return mx.sym.Activation(data=data, act_type=act_type, name=name)
    if act_type == "lrelu":
        return mx.sym.LeakyReLU(data=data, slope=0.2, act_type='leaky', name=name)
    raise NotImplementedError


def channel_squeeze_excitation(data, channels, name, ratio=16, act_type="relu"):
    """
    Squeeze excitation block.
    :param data:
    :param channels: Number of filters
    :param name: Prefix name of the block
    :param ratio: Ration for the number of neurons to use.
    :param act_type: Activation function to use
    :return: mxnet symbol
    """
    avg_pool = mx.sym.Pooling(data=data, kernel=(8, 8), pool_type='avg', name=name + '_pool0')
    flatten = mx.symbol.Flatten(data=avg_pool, name=name + '_flatten0')
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=channels // ratio, name=name + '_fc0')
    act1 = get_act(data=fc1, act_type=act_type, name=name + '_act0')
    fc2 = mx.symbol.FullyConnected(data=act1, num_hidden=channels, name=name + '_fc1')
    act2 = get_act(data=fc2, act_type='sigmoid', name=name + '_act1')

    return mx.symbol.broadcast_mul(data, mx.symbol.reshape(data=act2, shape=(-1, channels, 1, 1)))


def bottleneck_residual_block(data, channels, channels_operating, name, kernel=3, act_type='relu', use_se=False):
    """
    Returns a residual block without any max pooling operation
    :param data: Input data
    :param channels: Number of filters for all CNN-layers
    :param channels_operating: Number of filters used for 3x3, 5x5, 7x7,.. convolution
    :param name: Name for the residual block
    :param act_type: Activation function to use
    :return:
    """

    if use_se:
        se = channel_squeeze_excitation(data, channels, name=name + '_se', ratio=16)
        conv1 = mx.sym.Convolution(data=se, num_filter=channels_operating, kernel=(1, 1), pad=(0, 0),
                                   no_bias=True, name=name + '_conv1')
    else:
        conv1 = mx.sym.Convolution(data=data, num_filter=channels_operating, kernel=(1, 1), pad=(0, 0),
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
    :param channels_operating: Number of filters used for 3x3, 5x5, 7x7,.. convolution
    :param name: Name for the residual block
    :param act_type: Activation function to use
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
    bn1 = mx.sym.BatchNorm(data=conv1, name=name + '_bn1')
    act1 = get_act(data=bn1, act_type=act_type, name=name + '_act1')

    # kernel = 3
    conv2 = mx.sym.Convolution(data=act1, num_filter=channels, kernel=(kernel, kernel), stride=(1, 1),
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
    bn3 = mx.sym.BatchNorm(data=conv2, name=name + '_bn3')

    sum = mx.sym.broadcast_add(data, bn3, name=name + '_add')

    return sum


def rise_v2_symbol(channels=256, channels_operating_init=128, channel_expansion=64, channels_value_head=8,
                   channels_policy_head=81, value_fc_size=256, bc_res_blocks=13, res_blocks=0, act_type='relu',
                   n_labels=4992, grad_scale_value=0.01, grad_scale_policy=0.99, select_policy_from_plane=True,
                   use_se=True):
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
                            (They used 1.0 for default and 0.01 in the supervised setting)
    :param grad_scale_policy: Constant scalar which the gradient for the policy outputs are being scaled width.
                            (They used 1.0 for default and 0.99 in the supervised setting)
    :return: mxnet symbol of the model
    """
    # get the input data
    data = mx.sym.Variable(name='data')

    # first initial convolution layer followed by batchnormalization
    body = mx.sym.Convolution(data=data, num_filter=channels, kernel=(3, 3), pad=(1, 1),
                              no_bias=True, name="stem_conv0")
    body = mx.sym.BatchNorm(data=body, name='stem_bn0')
    body = get_act(data=body, act_type=act_type, name='stem_act0')
    channels_operating = channels_operating_init

    # build residual tower
    for idx in range(bc_res_blocks):
        use_squeeze_excitation = use_se

        if idx < bc_res_blocks - 5:
            use_squeeze_excitation = False
        body = bottleneck_residual_block(body, channels, channels_operating, name='bc_res_block%d' % idx, kernel=3,
                                         use_se=use_squeeze_excitation, act_type=act_type)
        channels_operating += channel_expansion

    for idx in range(res_blocks):
        if idx < res_blocks - 5:
            use_squeeze_excitation = False
        else:
            use_squeeze_excitation = use_se

        body = residual_block(body, channels, name='res_block%d' % idx, kernel=3,
                              use_se=use_squeeze_excitation, act_type=act_type)

    # for policy output
    if select_policy_from_plane:
        policy_out = mx.sym.Convolution(data=body, num_filter=channels, kernel=(3, 3), pad=(1, 1),
                                        no_bias=True, name="policy_conv0")
        policy_out = mx.sym.BatchNorm(data=policy_out, name='policy_bn0')
        policy_out = get_act(data=policy_out, act_type=act_type, name='policy_act0')

        policy_out = mx.sym.Convolution(data=policy_out, num_filter=channels_policy_head, kernel=(3, 3), pad=(1, 1),
                                        no_bias=True, name="policy_conv1")
        policy_out = mx.sym.flatten(data=policy_out, name='policy_out')
        policy_out = mx.sym.SoftmaxOutput(data=policy_out, name='policy', grad_scale=grad_scale_policy)
    else:
        policy_out = mx.sym.Convolution(data=body, num_filter=channels_policy_head, kernel=(1, 1), pad=(0, 0),
                                        no_bias=True, name="policy_conv0")
        policy_out = mx.sym.BatchNorm(data=policy_out, fix_gamma=False, name='policy_bn0')
        policy_out = mx.sym.Activation(data=policy_out, act_type=act_type, name='policy_act0')
        policy_out = mx.sym.Flatten(data=policy_out, name='policy_flatten0')
        policy_out = mx.sym.FullyConnected(data=policy_out, num_hidden=n_labels, name='policy_out')
        policy_out = mx.sym.SoftmaxOutput(data=policy_out, name='policy', grad_scale=grad_scale_policy)

    # for value output
    value_out = mx.sym.Convolution(data=body, num_filter=channels_value_head, kernel=(1, 1), pad=(0, 0),
                                   no_bias=True, name="value_conv0")
    value_out = mx.sym.BatchNorm(data=value_out, name='value_bn0')
    value_out = get_act(data=value_out, act_type=act_type, name='value_act0')
    value_out = mx.sym.Flatten(data=value_out, name='value_flatten0')
    value_out = mx.sym.FullyConnected(data=value_out, num_hidden=value_fc_size, name='value_fc0')
    value_out = get_act(data=value_out, act_type=act_type, name='value_act1')
    value_out = mx.sym.FullyConnected(data=value_out, num_hidden=1, name='value_fc1')
    value_out = get_act(data=value_out, act_type='tanh', name='value_out')
    value_out = mx.sym.LinearRegressionOutput(data=value_out, name='value', grad_scale=grad_scale_value)

    # group value_out and policy_out together
    sym = mx.symbol.Group([value_out, policy_out])

    return sym


def preact_resnet_symbol(channels=256, channels_value_head=8,
                   channels_policy_head=81, value_fc_size=256, res_blocks=19, act_type='relu',
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
                            (They used 1.0 for default and 0.01 in the supervised setting)
    :param grad_scale_policy: Constant scalar which the gradient for the policy outputs are being scaled width.
                            (They used 1.0 for default and 0.99 in the supervised setting)
    :return: mxnet symbol of the model
    """
    # get the input data
    data = mx.sym.Variable(name='data')

    # first initial convolution layer followed by batchnormalization
    body = mx.sym.Convolution(data=data, num_filter=channels, kernel=(3, 3), pad=(1, 1),
                              no_bias=True, name="stem_conv0")
    body = mx.sym.BatchNorm(data=body, name='stem_bn0')
    body = get_act(data=body, act_type=act_type, name='stem_act0')

    for idx in range(res_blocks):
        body = preact_residual_block(body, channels, name='res_block%d' % idx, kernel=3,
                              act_type=act_type)

    body = mx.sym.BatchNorm(data=body, name='stem_bn1')
    body = get_act(data=body, act_type=act_type, name='stem_act1')

    # for policy output
    if select_policy_from_plane:
        policy_out = mx.sym.Convolution(data=body, num_filter=channels, kernel=(3, 3), pad=(1, 1),
                                        no_bias=True, name="policy_conv0")
        policy_out = mx.sym.BatchNorm(data=policy_out, name='policy_bn0')
        policy_out = get_act(data=policy_out, act_type=act_type, name='policy_act0')

        policy_out = mx.sym.Convolution(data=policy_out, num_filter=channels_policy_head, kernel=(3, 3), pad=(1, 1),
                                        no_bias=True, name="policy_conv1")
        policy_out = mx.sym.flatten(data=policy_out, name='policy_out')
        policy_out = mx.sym.SoftmaxOutput(data=policy_out, name='policy', grad_scale=grad_scale_policy)
    else:
        policy_out = mx.sym.Convolution(data=body, num_filter=channels_policy_head, kernel=(1, 1), pad=(0, 0),
                                        no_bias=True, name="policy_conv0")
        policy_out = mx.sym.BatchNorm(data=policy_out, fix_gamma=False, name='policy_bn0')
        policy_out = mx.sym.Activation(data=policy_out, act_type=act_type, name='policy_act0')
        policy_out = mx.sym.Flatten(data=policy_out, name='policy_flatten0')
        policy_out = mx.sym.FullyConnected(data=policy_out, num_hidden=n_labels, name='policy_out')
        policy_out = mx.sym.SoftmaxOutput(data=policy_out, name='policy', grad_scale=grad_scale_policy)

    # for value output
    value_out = mx.sym.Convolution(data=body, num_filter=channels_value_head, kernel=(1, 1), pad=(0, 0),
                                   no_bias=True, name="value_conv0")
    value_out = mx.sym.BatchNorm(data=value_out, name='value_bn0')
    value_out = get_act(data=value_out, act_type=act_type, name='value_act0')
    value_out = mx.sym.Flatten(data=value_out, name='value_flatten0')
    value_out = mx.sym.FullyConnected(data=value_out, num_hidden=value_fc_size, name='value_fc0')
    value_out = get_act(data=value_out, act_type=act_type, name='value_act1')
    value_out = mx.sym.FullyConnected(data=value_out, num_hidden=1, name='value_fc1')
    value_out = get_act(data=value_out, act_type='tanh', name='value_out')
    value_out = mx.sym.LinearRegressionOutput(data=value_out, name='value', grad_scale=grad_scale_value)

    # group value_out and policy_out together
    sym = mx.symbol.Group([value_out, policy_out])

    return sym
