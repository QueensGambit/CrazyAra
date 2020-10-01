"""
@file: builder_util_symbol.py
Created on 06.12.19
@project: CrazyAra
@author: queensgambit

Utility methods for building the neural network using MXNet symbol API
"""

import mxnet as mx


def get_act(data, act_type, name):
    """Wrapper method for different non linear activation functions"""
    if act_type in ["relu", "sigmoid", "softrelu", "softsign", "tanh"]:
        return mx.sym.Activation(data=data, act_type=act_type, name=name)
    if act_type == "lrelu":
        return mx.sym.LeakyReLU(data=data, slope=0.2, act_type='leaky', name=name)
    if act_type == "hard_sigmoid":
        return mx.sym.clip(data=data + 3.0, a_min=0.0, a_max=6.0, name=name) / 6.0
    if act_type == "hard_swish":
        return data * (mx.sym.clip(data + 3, 0, 6, name=name) / 6.0)

    raise NotImplementedError


def channel_squeeze_excitation(data, channels, name, ratio=16, act_type="relu", use_hard_sigmoid=False):
    """
    Squeeze excitation block - Hu et al. - https://arxiv.org/abs/1709.01507
    :param data:
    :param channels: Number of filters
    :param name: Prefix name of the block
    :param ratio: Ration for the number of neurons to use.
    :param act_type: Activation function to use
    :param use_hard_sigmoid: Whether to use the linearized form of sigmoid:
     MobileNetv3: https://arxiv.org/pdf/1905.02244.pdf
    :return: mxnet symbol
    """
    return channel_attention_module(data, channels, name, ratio, act_type, use_hard_sigmoid, pool_type="avg")


def get_stem(data, channels, act_type):
    """
    Creates the convolution stem before the residual head
    :param data: Input data
    :param channels: Number of channels for the stem
    :param act_type: Activation function
    :return: symbol
    """
    body = mx.sym.Convolution(data=data, num_filter=channels, kernel=(3, 3), pad=(1, 1),
                              no_bias=True, name="stem_conv0")
    body = mx.sym.BatchNorm(data=body, name='stem_bn0')
    body = get_act(data=body, act_type=act_type, name='stem_act0')

    return body


def get_depthwise_stem(data, channels, act_type):
    """
    Sames as get_stem() but with group depthwise convolutions
    """
    conv1 = mx.sym.Convolution(data=data, num_filter=channels, kernel=(1, 1), pad=(0, 0), no_bias=True, name="stem_conv0")
    bn2 = mx.sym.BatchNorm(data=conv1, name='stem_bn0')
    act1 = get_act(data=bn2, act_type=act_type, name='stem_act0')
    conv2 = mx.sym.Convolution(data=act1, num_filter=channels, num_group=channels, kernel=(3, 3), pad=(1, 1),
                               no_bias=True, name="stem_conv1")
    bn3 = mx.sym.BatchNorm(data=conv2, name='stem_bn1')
    out = get_act(data=bn3, act_type=act_type, name='stem_act1')
    out = mx.sym.Convolution(data=out, num_filter=channels, kernel=(1, 1),
                               pad=(0, 0), no_bias=True, name='stem_conv2')
    return out


def value_head(data, channels_value_head=4, value_kernelsize=1, act_type='relu', value_fc_size=256,
               grad_scale_value=0.01, use_se=False, use_mix_conv=False, orig_data=None, use_avg_features=False,
               use_raw_features=False, value_nb_hidden=0, value_fc_size_hidden=256, use_batchnorm=False, dropout_rate=0.0,
               use_conv_features=True):
    """
    Value head of the network which outputs the value evaluation. A floating point number in the range [-1,+1].
    :param data: Input data
    :param channels_value_head Number of channels for the value head
    :param value_kernelsize Kernel size to use for the convolutional layer
    :param act_type: Activation function to use
    :param value_fc_size Number of units of the fully connected layer
    :param grad_scale_value: Optional re-weighting of gradient
    :param use_se: Indicates if a squeeze excitation layer shall be used
    :param use_mix_conv: True, if an additional mix convolutional layer shall be used
    :param orig_data: Original data input of the network
    :param use_avg_features: If true average features are extracted from the original input
    :param value_nb_hidden: Number of hidden layers
    :param value_fc_size_hidden: Number of units in the hidden layers
    :param use_batchnorm: If true batchnormalization is used
    :param dropout_rate: If > 0, dropout is enabled after the last fully connected layer
    :param use_conv_features: If true, features of the shared network are used
    """
    # for value output
    value_flatten = None
    if use_conv_features:
        value_out = mx.sym.Convolution(data=data, num_filter=channels_value_head,
                                       kernel=(value_kernelsize, value_kernelsize),
                                       pad=(value_kernelsize//2, value_kernelsize//2),
                                       no_bias=True, name="value_conv0")
        value_out = mx.sym.BatchNorm(data=value_out, name='value_bn0')
        value_out = get_act(data=value_out, act_type=act_type, name='value_act0')
        if use_mix_conv:
            mix_conv(value_out, channels=channels_value_head, kernels=[3, 5, 7, 9], name='value_mix_conv0')
            value_out = mx.sym.BatchNorm(data=value_out, name='value_mix_bn1')
            value_out = get_act(data=value_out, act_type=act_type, name='value_mix_act1')

        value_flatten = mx.sym.Flatten(data=value_out, name='value_flatten1')
        if use_se:
            avg_pool = mx.sym.Pooling(data=data, kernel=(8, 8), pool_type='avg', name='value_pool0')
            pool_flatten = mx.symbol.Flatten(data=avg_pool, name='value_flatten0')
            value_flatten = mx.sym.Concat(*[value_flatten, pool_flatten], name='value_concat')

    features = mx.sym.slice_axis(orig_data, axis=1, begin=0, end=13)

    if orig_data is not None and use_avg_features:
        avg_pool = mx.sym.Pooling(data=orig_data, kernel=(8, 8), pool_type='avg', name='value_avg_pool0')
        pool_flatten = mx.symbol.Flatten(data=avg_pool, name='value_pool_flatten')

        avg_pool1 = mx.sym.Pooling(data=features, kernel=(4, 4), stride=(1, 1), pool_type='avg', name='value_avg_pool1')
        pool_flatten1 = mx.symbol.Flatten(data=avg_pool1, name='value_pool_flatten1')

        avg_pool2 = mx.sym.Pooling(data=features, kernel=(2, 2), stride=(1, 1), pool_type='avg', name='value_avg_pool2')
        pool_flatten2 = mx.symbol.Flatten(data=avg_pool2, name='value_pool_flatten2')

        avg_mean = mx.sym.mean(data=features, axis=1)
        mean_flatten = mx.symbol.Flatten(data=avg_mean, name='value_mean_flatten0')

        w_pieces = mx.sym.slice_axis(features, axis=1, begin=0, end=6)
        b_pieces = mx.sym.slice_axis(features, axis=1, begin=6, end=12)

        avg_mean = mx.sym.mean(data=w_pieces, axis=1)
        w_features_mean_flatten = mx.symbol.Flatten(data=avg_mean, name='value_w_features_mean_flatten0')
        avg_mean = mx.sym.mean(data=b_pieces, axis=1)
        b_features_mean_flatten = mx.symbol.Flatten(data=avg_mean, name='value_b_features_mean_flatten0')

        if value_flatten is None:
            value_flatten = mx.sym.Concat(*[pool_flatten, pool_flatten1, pool_flatten2, mean_flatten, w_features_mean_flatten, b_features_mean_flatten], name='value_concat')
        else:
            value_flatten = mx.sym.Concat(*[value_flatten, pool_flatten, pool_flatten1, pool_flatten2, mean_flatten, w_features_mean_flatten, b_features_mean_flatten], name='value_concat')

    if orig_data is not None and use_raw_features:
        raw_flatten = mx.symbol.Flatten(data=features, name='value_flatten_raw')
        if value_flatten is None:
            value_flatten = raw_flatten
        else:
            value_flatten = mx.sym.Concat(*[value_flatten, raw_flatten], name='value_concat_raw')

    value_out = mx.sym.FullyConnected(data=value_flatten, num_hidden=value_fc_size, name='value_fc0')
    if use_batchnorm:
        value_out = mx.sym.BatchNorm(data=value_out, name='value_bn1')

    value_out = get_act(data=value_out, act_type=act_type, name='value_act1')
    for i in range(value_nb_hidden):
        value_out = mx.sym.FullyConnected(data=value_out, num_hidden=value_fc_size_hidden, name=f'value_fc{i + 1}')
        if use_batchnorm:
            value_out = mx.sym.BatchNorm(data=value_out, name=f'value_bn{i+2}')
        value_out = get_act(data=value_out, act_type=act_type, name=f'value_act{i+2}')

    if dropout_rate != 0:
        value_out = mx.sym.Dropout(value_out, p=dropout_rate)

    value_out = mx.sym.FullyConnected(data=value_out, num_hidden=1, name='value_fc_final')
    value_out = get_act(data=value_out, act_type='tanh', name='value_out')
    value_out = mx.sym.LinearRegressionOutput(data=value_out, name='value', grad_scale=grad_scale_value)
    return value_out


def policy_head(data, channels, act_type, channels_policy_head, select_policy_from_plane, n_labels,
                grad_scale_policy=1.0, use_se=False, no_bias=False):
    """
    Policy head of the network which outputs the policy distribution for a given position
    :param data: Input data
    :param channels_policy_head:
    :param act_type: Activation function to use
    :param select_policy_from_plane: True for policy head move representation
    :param n_labels: Number of possible move targets
    :param grad_scale_policy: Optional re-weighting of gradient
    :param use_se: Indicates if a squeeze excitation layer shall be used
    :param no_bias: If no bias shall be used for the last conv layer before softmax (backward compability)
    """
    # for policy output
    kernel = 3
    policy_out = mx.sym.Convolution(data=data, num_filter=channels, kernel=(kernel, kernel),
                                    pad=(kernel // 2, kernel // 2), no_bias=True, name="policy_conv0")
    policy_out = mx.sym.BatchNorm(data=policy_out, name='policy_bn0')
    policy_out = get_act(data=policy_out, act_type=act_type, name='policy_act0')
    if use_se:
        policy_out = channel_squeeze_excitation(policy_out, channels, name='policy_se', ratio=4, act_type=act_type,
                                                use_hard_sigmoid=True)
    if select_policy_from_plane:
        policy_out = mx.sym.Convolution(data=policy_out, num_filter=channels_policy_head, kernel=(3, 3), pad=(1, 1),
                                        no_bias=no_bias, name="policy_conv1")
        policy_out = mx.sym.flatten(data=policy_out, name='policy_out')
        policy_out = mx.sym.SoftmaxOutput(data=policy_out, name='policy', grad_scale=grad_scale_policy)
    else:
        policy_out = mx.sym.Flatten(data=policy_out, name='policy_flatten0')
        policy_out = mx.sym.FullyConnected(data=policy_out, num_hidden=n_labels, name='policy_out')
        policy_out = mx.sym.SoftmaxOutput(data=policy_out, name='policy', grad_scale=grad_scale_policy)

    return policy_out


def policy_head_depthwise(data, channels, act_type, channels_policy_head, select_policy_from_plane, n_labels,
                grad_scale_policy=1.0, no_bias=False):
    """
    Policy head of the network which outputs the policy distribution for a given position
    :param data: Input data
    :param channels_policy_head:
    :param act_type: Activation function to use
    :param select_policy_from_plane: True for policy head move representation
    :param n_labels: Number of possible move targets
    :param grad_scale_policy: Optional re-weighting of gradient
    :param use_se: Indicates if a squeeze excitation layer shall be used
    :param no_bias: If no bias shall be used for the last conv layer before softmax (backward compability)
    """

    policy_out = mx.sym.Convolution(data=data, num_filter=channels*2, kernel=(1, 1), pad=(0, 0), no_bias=True, name="policy_conv0")
    policy_out = mx.sym.BatchNorm(data=policy_out, name='policy_bn0')
    policy_out = get_act(data=policy_out, act_type=act_type, name='policy_act0')
    policy_out = mx.sym.Convolution(data=policy_out, num_filter=channels*2, num_group=channels, kernel=(3, 3), pad=(1, 1),
                               no_bias=True, name="policy_conv1")
    policy_out = mx.sym.BatchNorm(data=policy_out, name='policy_bn1')
    policy_out = get_act(data=policy_out, act_type=act_type, name='policy_act1')
    policy_out = mx.sym.Convolution(data=policy_out, num_filter=channels_policy_head, kernel=(1, 1),
                               pad=(0, 0), no_bias=no_bias, name='policy_conv2')

    if select_policy_from_plane:
        # policy_out = mx.sym.Convolution(data=policy_out, num_filter=channels_policy_head, kernel=(3, 3), pad=(1, 1),
        #                                 no_bias=no_bias, name="policy_conv1")
        policy_out = mx.sym.flatten(data=policy_out, name='policy_out')
        policy_out = mx.sym.SoftmaxOutput(data=policy_out, name='policy', grad_scale=grad_scale_policy)
    else:
        policy_out = mx.sym.Flatten(data=policy_out, name='policy_flatten0')
        policy_out = mx.sym.FullyConnected(data=policy_out, num_hidden=n_labels, name='policy_out')
        policy_out = mx.sym.SoftmaxOutput(data=policy_out, name='policy', grad_scale=grad_scale_policy)

    return policy_out


def mix_conv(data, name, channels, kernels):
    """
    Mix depth-wise convolution layers
    :param data: Input data
    :param name: Name of the block
    :param channels: Number of convolutional channels
    :param kernels: List of kernel sizes to use
    :return: symbol
    """
    num_splits = len(kernels)
    conv_layers = []

    if num_splits == 1:
        kernel = kernels[0]
        return mx.sym.Convolution(data=data, num_filter=channels, kernel=(kernel, kernel),
                                  pad=(kernel//2, kernel//2), no_bias=True,
                                  num_group=channels, name=name + '_conv3_k%d' % kernel)

    for xi, kernel in zip(mx.sym.split(data, axis=1, num_outputs=num_splits, name=name + '_split'), kernels):
        conv_layers.append(mx.sym.Convolution(data=xi, num_filter=channels//num_splits, kernel=(kernel, kernel),
                                              pad=(kernel//2, kernel//2), no_bias=True, num_group=channels//num_splits,
                                              name=name + '_conv3_k%d' % kernel))
    return mx.sym.Concat(*conv_layers, name=name + '_concat')


def channel_attention_module(data, channels, name, ratio=16, act_type="relu", use_hard_sigmoid=False, pool_type="both"):
    """
    Channel Attention Module of (CBAM) - Woo et al. - https://arxiv.org/pdf/1807.06521.pdf
    modified: Input to the shared fully connected layer gets concatenated rather than
              shared network with later addition
    :param data: Input data
    :param channels: Number of input channels
    :param name: Layer name
    :param ratio: Reduction ratio
    :param act_type: Activation type for hidden layer in MLP
    :param use_hard_sigmoid: Whether to use the linearized form of sigmoid:
     MobileNetv3: https://arxiv.org/pdf/1905.02244.pdf
    :param pool_type: Pooling type to use. If "both" are used, then the features will be concatenated.
    Available options are: ["both", "avg", "max"]
     """
    if pool_type == "both":
        avg_pool = mx.sym.Pooling(data=data, kernel=(8, 8), pool_type='avg', name=name + '_avg_pool0')
        max_pool = mx.sym.Pooling(data=data, kernel=(8, 8), pool_type='max', name=name + '_max_pool0')
        merge = mx.sym.Concat(avg_pool, max_pool, dim=1, name=name + '_concat_0')
    elif pool_type == "avg":
        merge = mx.sym.Pooling(data=data, kernel=(8, 8), pool_type='avg', name=name + '_avg_pool0')
    elif pool_type == "max":
        merge = mx.sym.Pooling(data=data, kernel=(8, 8), pool_type='max', name=name + '_max_pool0')
    else:
        raise Exception(f"Invalid value for pool_type given: {pool_type}")

    flatten = mx.symbol.Flatten(data=merge, name=name + '_flatten0')
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=channels // ratio, name=name + '_fc0')
    act1 = get_act(data=fc1, act_type=act_type, name=name + '_act0')
    fc2 = mx.symbol.FullyConnected(data=act1, num_hidden=channels, name=name + '_fc1')
    if use_hard_sigmoid:
        act_type = 'hard_sigmoid'
    else:
        act_type = 'sigmoid'
    act2 = get_act(data=fc2, act_type=act_type, name=name + '_act1')
    return mx.symbol.broadcast_mul(data, mx.symbol.reshape(data=act2, shape=(-1, channels, 1, 1)))


def spatial_attention_module(data, name, use_hard_sigmoid=False, pool_type="both"):
    """
    Spatial Attention Modul of (CBA) - Woo et al. - https://arxiv.org/pdf/1807.06521.pdf
    :param data: Input data
    :param name: Layer name
    :param use_hard_sigmoid: Whether to use the linearized form of sigmoid:
     MobileNetv3: https://arxiv.org/pdf/1905.02244.pdf
    :param pool_type: Pooling type to use. If "both" are used, then the features will be concatenated.
    Available options are: ["both", "avg", "max"]
     """
    if pool_type == "both":
        avg_spatial = mx.symbol.mean(data=data, axis=1, keepdims=True, name=name + '_avg_spatial0')
        max_spatial = mx.symbol.max(data=data, axis=1, keepdims=True, name=name + '_max_spatial0')
        merge = mx.sym.Concat(avg_spatial, max_spatial, dim=1, name=name + '_concat_0')
    elif pool_type == "avg":
        merge = mx.symbol.mean(data=data, axis=1, keepdims=True, name=name + '_avg_spatial0')
    elif pool_type == "max":
        merge = mx.symbol.max(data=data, axis=1, keepdims=True, name=name + '_avg_spatial0')
    else:
        raise Exception(f"Invalid value for pool_type given: {pool_type}")

    conv0 = mx.sym.Convolution(data=merge, num_filter=1, kernel=(7, 7),
                             pad=(3, 3), no_bias=False,
                             num_group=1, name=name + '_conv0')
    if use_hard_sigmoid:
        act_type = 'hard_sigmoid'
    else:
        act_type = 'sigmoid'
    act0 = get_act(data=conv0, act_type=act_type, name=name + '_act0')
    return mx.symbol.broadcast_mul(act0, data)


def convolution_block_attention_module(data, channels, name, ratio=16, act_type="relu", use_hard_sigmoid=False):
    """
    Convolutional Block Attention Module (CBAM) - Woo et al. - https://arxiv.org/pdf/1807.06521.pdf
    First applies channel attention and later spatial attention
        :param data: Input data
    :param channels: Number of input channels
    :param name: Layer name
    :param ratio: Reduction ratio
    :param act_type: Activation type for hidden layer in MLP
    :param use_hard_sigmoid: Whether to use the linearized form of sigmoid:
     MobileNetv3: https://arxiv.org/pdf/1905.02244.pdf
    """
    data = channel_attention_module(data, channels, name + '_channel', ratio, act_type, use_hard_sigmoid)
    return spatial_attention_module(data, name + '_spatial', use_hard_sigmoid)


def ca_se(data, channels, name, ratio=16, act_type="relu", use_hard_sigmoid=False):
    """
    Channel-Average-Squeeze-Excitation (caSE)
    Alias function for channel_attention_module() with average pooling
    """
    return channel_attention_module(data, channels, name, ratio, act_type, use_hard_sigmoid, pool_type="avg")


def cm_se(data, channels, name, ratio=16, act_type="relu", use_hard_sigmoid=False):
    """
    Channel-Max-Squeeze-Excitation (cmSE)
    Alias function for channel_attention_module() with max pooling
    """
    return channel_attention_module(data, channels, name, ratio, act_type, use_hard_sigmoid, pool_type="max")


def sa_se(data, name, use_hard_sigmoid=False):
    """
    Spatial-Average-Squeeze-Excitation (smSE)
    Alias function for spatial_attention_module() with average pooling
    """
    return spatial_attention_module(data, name, use_hard_sigmoid, pool_type="avg")

def sm_se(data, name, use_hard_sigmoid=False):
    """
    Spatial-Max-Squeeze-Excitation (smSE)
    Alias function for spatial_attention_module() with max pooling
    """
    return spatial_attention_module(data, name, use_hard_sigmoid, pool_type="max")
