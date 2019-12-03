"""
@file: rise_mobile_v3.py
Created on 21.11.19
@project: CrazyAra
@author: queensgambit

Upgrade of RISE architecture using mixed depthwise convolutions, preactivation residuals and dropout
 proposed by Johannes Czech

Influenced by the following papers:
    * MixConv: Mixed Depthwise Convolutional Kernels, Mingxing Tan, Quoc V. Le, https://arxiv.org/abs/1907.09595
    * ProxylessNas: Direct Neural Architecture Search on Target Task and Hardware, Han Cai, Ligeng Zhu, Song Han.
     https://arxiv.org/abs/1812.00332
    * MnasNet: Platform-Aware Neural Architecture Search for Mobile,
     Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan, Mark Sandler, Andrew Howard, Quoc V. Le
     http://openaccess.thecvf.com/content_CVPR_2019/html/Tan_MnasNet_Platform-Aware_Neural_Architecture_Search_for_Mobile_CVPR_2019_paper.html
    * FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search,
    Bichen Wu, Xiaoliang Dai, Peizhao Zhang, Yanghan Wang, Fei Sun, Yiming Wu, Yuandong Tian, Peter Vajda, Yangqing Jia, Kurt Keutzer,
    http://openaccess.thecvf.com/content_CVPR_2019/html/Wu_FBNet_Hardware-Aware_Efficient_ConvNet_Design_via_Differentiable_Neural_Architecture_Search_CVPR_2019_paper.html
    * MobileNetV3: Searching for MobileNetV3,
    Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam.
    https://arxiv.org/abs/1905.02244
"""
import mxnet as mx
from DeepCrazyhouse.src.domain.neural_net.architectures.rise_mobile_symbol import get_act, channel_squeeze_excitation


def mix_conv(data, name, channels, kernels):
    """
    Mix depth-wise convolution layers
    :param data:
    :param name:
    :param channels:
    :param kernels:
    :return:
    """
    num_splits = len(kernels)
    conv_layers = []

    for xi, kernel in zip(mx.sym.split(data, axis=1, num_outputs=num_splits), kernels):
        conv_layers.append(mx.sym.Convolution(data=xi, num_filter=channels//num_splits, kernel=(kernel, kernel),
                                              pad=(kernel//2, kernel//2), no_bias=True,
                                              name=name + '_conv3_k%d' % kernel))
    return mx.sym.Concat(*conv_layers, name=name + '_concat')


def preact_residual_dmixconv_block(data, channels, channels_operating, name, kernels=None, act_type='relu', use_se=True):
    """
    Returns a residual block without any max pooling operation
    :param data: Input data
    :param channels: Number of filters for all CNN-layers
    :param name: Name for the residual block
    :param act_type: Activation function to use
    :return:
    """
    bn1 = mx.sym.BatchNorm(data=data, name=name + '_bn1')
    conv1 = mx.sym.Convolution(data=bn1, num_filter=channels_operating, kernel=(1, 1), pad=(0, 0), no_bias=True,
                               name=name + '_conv1')
    bn2 = mx.sym.BatchNorm(data=conv1, name=name + '_bn2')
    act1 = get_act(data=bn2, act_type=act_type, name=name + '_act1')
    conv2 = mix_conv(data=act1, channels=channels_operating, kernels=kernels, name=name + 'conv2')
    bn3 = mx.sym.BatchNorm(data=conv2, name=name + '_bn3')
    out = get_act(data=bn3, act_type=act_type, name=name + '_act2')
    if use_se:
        out = channel_squeeze_excitation(out, channels_operating, name=name + '_se', ratio=4, act_type=act_type,
                                         use_hard_sigmoid=True)
    conv3 = mx.sym.Convolution(data=out, num_filter=channels, kernel=(1, 1),
                               pad=(0, 0), no_bias=True, name=name + '_conv3')
    bn4 = mx.sym.BatchNorm(data=conv3, name=name + '_bn4')
    sum = mx.sym.broadcast_add(data, bn4, name=name + '_add')

    return sum


def get_stem(data, channels, act_type):
    """
    Creates the convolution stem before the residual head
    :param data:
    :param channels:
    :param act_type:
    :return:
    """
    body = mx.sym.Convolution(data=data, num_filter=channels, kernel=(3, 3), pad=(1, 1),
                              no_bias=True, name="stem_conv0")
    body = mx.sym.BatchNorm(data=body, name='stem_bn0')
    body = get_act(data=body, act_type=act_type, name='stem_act0')

    return body


def value_head(data, channels_value_head=256, value_kernelsize=1, act_type='relu', value_fc_size=256,
               grad_scale_value=0.01, use_se=False):
    # for value output
    value_out = mx.sym.Convolution(data=data, num_filter=channels_value_head,
                                   kernel=(value_kernelsize, value_kernelsize),
                                   pad=(value_kernelsize//2, value_kernelsize//2),
                                   no_bias=True, name="value_conv0")
    value_out = mx.sym.BatchNorm(data=value_out, name='value_bn0')
    value_out = get_act(data=value_out, act_type=act_type, name='value_act0')
    if use_se:
        value_out = channel_squeeze_excitation(value_out, channels_value_head, name='value_se', ratio=1,
                                               act_type=act_type, use_hard_sigmoid=True)
    value_out = mx.sym.Flatten(data=value_out, name='value_flatten0')
    value_out = mx.sym.FullyConnected(data=value_out, num_hidden=value_fc_size, name='value_fc0')
    value_out = get_act(data=value_out, act_type=act_type, name='value_act1')
    value_out = mx.sym.FullyConnected(data=value_out, num_hidden=1, name='value_fc1')
    value_out = get_act(data=value_out, act_type='tanh', name='value_out')
    value_out = mx.sym.LinearRegressionOutput(data=value_out, name='value', grad_scale=grad_scale_value)
    return value_out


def policy_head(data, channels, act_type, channels_policy_head, select_policy_from_plane, n_labels,
                grad_scale_policy=1.0, use_se=False):
    # for policy output
    if select_policy_from_plane:
        kernel = 3
        policy_out = mx.sym.Convolution(data=data, num_filter=channels, kernel=(kernel, kernel), pad=(kernel//2, kernel//2),
                                        no_bias=True, name="policy_conv0")
        policy_out = mx.sym.BatchNorm(data=policy_out, name='policy_bn0')
        policy_out = get_act(data=policy_out, act_type=act_type, name='policy_act0')
        if use_se:
            policy_out = channel_squeeze_excitation(policy_out, channels, name='policy_se', ratio=4, act_type=act_type,
                                                    use_hard_sigmoid=True)
        policy_out = mx.sym.Convolution(data=policy_out, num_filter=channels_policy_head, kernel=(3, 3), pad=(1, 1),
                                        no_bias=True, name="policy_conv1")
        policy_out = mx.sym.flatten(data=policy_out, name='policy_out')
        policy_out = mx.sym.SoftmaxOutput(data=policy_out, name='policy', grad_scale=grad_scale_policy)
    else:
        kernel = 3
        policy_out = mx.sym.Convolution(data=data, num_filter=channels_policy_head, kernel=(kernel, kernel), pad=(kernel//2, kernel//2),
                                        no_bias=True, name="policy_conv0")
        policy_out = mx.sym.BatchNorm(data=policy_out, name='policy_bn0')
        policy_out = mx.sym.Activation(data=policy_out, act_type=act_type, name='policy_act0')
        if use_se:
            policy_out = channel_squeeze_excitation(policy_out, channels, name='policy_se', ratio=4, act_type=act_type,
                                                    use_hard_sigmoid=True)
        policy_out = mx.sym.Flatten(data=policy_out, name='policy_flatten0')
        policy_out = mx.sym.FullyConnected(data=policy_out, num_hidden=n_labels, name='policy_out')
        policy_out = mx.sym.SoftmaxOutput(data=policy_out, name='policy', grad_scale=grad_scale_policy)

    return policy_out


def rise_mobile_v3_symbol(channels=256, channels_operating_init=128, channel_expansion=64, act_type='relu',
                          channels_policy_head=81, dropout_rate=0.15, select_policy_from_plane=True, use_se=True,
                          res_blocks=13, n_labels=4992):
    """
    RISEv3 architecture
    :param channels:
    :param channels_operating_init:
    :param channel_expansion:
    :param act_type:
    :param channels_policy_head:
    :param dropout_rate:
    :param select_policy_from_plane:
    :param use_se:
    :param res_blocks:
    :return:
    """
    # get the input data
    data = mx.sym.Variable(name='data')

    data = get_stem(data=data, channels=channels, act_type=act_type)

    cur_channels = channels_operating_init

    kernels = [
        [3],  # 0
        [3],  # 1
        [3],  # 2
        [3],  # 6
        [3],  # 7
        [3, 5, 7, 9],  # 11
        [3, 5],  # 3
        [3, 5],  # 4
        [3, 5],  # 5
        [3, 5],  # 9
        [3, 5],  # 10
        [3, 5, 7, 9],  # 11
        [3, 5],  # 12
    ]
    for idx in range(res_blocks):

        cur_kernels = kernels[idx]
        if idx > 5:
            use_se = True
        else:
            use_se = False
        data = preact_residual_dmixconv_block(data=data, channels=channels, channels_operating=cur_channels,
                                              kernels=cur_kernels, name='dconv_%d' % idx, use_se=True)
        cur_channels += channel_expansion
    # return data
    data = mx.sym.BatchNorm(data=data, name='stem_bn1')
    data = get_act(data=data, act_type=act_type, name='stem_act1')

    if dropout_rate != 0:
        data = mx.sym.Dropout(data, p=dropout_rate)

    value_out = value_head(data=data, act_type=act_type, use_se=use_se, channels_value_head=4)
    policy_out = policy_head(data=data, act_type=act_type, channels_policy_head=channels_policy_head, n_labels=n_labels,
                             select_policy_from_plane=select_policy_from_plane, use_se=use_se, channels=channels)
    # group value_out and policy_out together
    sym = mx.symbol.Group([value_out, policy_out])

    return sym
