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
from DeepCrazyhouse.src.domain.neural_net.architectures.builder_util_symbol import get_act, channel_squeeze_excitation, \
    mix_conv, get_stem, value_head, policy_head


def preact_residual_dmixconv_block(data, channels, channels_operating, name, kernels=None, act_type='relu', use_se=True):
    """
    Returns a residual block without any max pooling operation
    :param data: Input data
    :param channels: Number of filters for all CNN-layers
    :param name: Name for the residual block
    :param act_type: Activation function to use
    :return: symbol
    """
    bn1 = mx.sym.BatchNorm(data=data, name=name + '_bn1')
    conv1 = mx.sym.Convolution(data=bn1, num_filter=channels_operating, kernel=(1, 1), pad=(0, 0), no_bias=True,
                               name=name + '_conv1')
    bn2 = mx.sym.BatchNorm(data=conv1, name=name + '_bn2')
    act1 = get_act(data=bn2, act_type=act_type, name=name + '_act1')
    conv2 = mix_conv(data=act1, channels=channels_operating, kernels=kernels, name=name + 'conv2')
    bn3 = mx.sym.BatchNorm(data=conv2, name=name + '_bn3')
    act2 = get_act(data=bn3, act_type=act_type, name=name + '_act2')
    out = mx.sym.Convolution(data=act2, num_filter=channels, kernel=(1, 1),
                               pad=(0, 0), no_bias=True, name=name + '_conv3')
    # out = mx.sym.BatchNorm(data=conv3, name=name + '_bn4')
    if use_se:
        out = channel_squeeze_excitation(out, channels, name=name + '_se', ratio=4, act_type=act_type,
                                         use_hard_sigmoid=True)
    out_sum = mx.sym.broadcast_add(data, out, name=name + '_add')

    return out_sum


def rise_mobile_v3_symbol(channels=256, channels_operating_init=128, channel_expansion=64, act_type='relu',
                          channels_value_head=32, channels_policy_head=81, value_fc_size=128, dropout_rate=0.15,
                          select_policy_from_plane=True, use_se=True, kernels=None, n_labels=4992):
    """
    RISEv3 architecture
    :param channels: Main number of channels
    :param channels_operating_init: Initial number of channels at the start of the net for the depthwise convolution
    :param channel_expansion: Number of channels to add after each residual block
    :param act_type: Activation type to use
    :param channels_value_head: Number of channels for the value head
    :param value_fc_size: Number of units in the fully connected layer of the value head
    :param channels_policy_head: Number of channels for the policy head
    :param dropout_rate: Droput factor to use. If 0, no dropout will be applied. Value must be in [0,1]
    :param select_policy_from_plane: True, if policy head type shall be used
    :param use_se: Indicates if a squeeze excitation layer shall be used
    :param res_blocks: Number of residual blocks
    :param n_labels: Number of policy target labels (used for select_policy_from_plane=False)
    :return: symbol
    """
    # get the input data
    data = mx.sym.Variable(name='data')

    data = get_stem(data=data, channels=channels, act_type=act_type)

    if kernels is None:
        kernels = [3] * 13

    cur_channels = channels_operating_init

    for idx, cur_kernels in enumerate(kernels):

        cur_kernels = kernels[idx]
        if idx == 4 or idx >= 9:
            use_se = True
        else:
            use_se = False
        data = preact_residual_dmixconv_block(data=data, channels=channels, channels_operating=cur_channels,
                                              kernels=cur_kernels, name='dconv_%d' % idx, use_se=use_se)
        cur_channels += channel_expansion
    # return data
    data = mx.sym.BatchNorm(data=data, name='stem_bn1')
    data = get_act(data=data, act_type=act_type, name='stem_act1')

    if dropout_rate != 0:
        data = mx.sym.Dropout(data, p=dropout_rate)

    value_out = value_head(data=data, act_type=act_type, use_se=use_se, channels_value_head=channels_value_head,
                           value_fc_size=value_fc_size, use_mix_conv=True)
    policy_out = policy_head(data=data, act_type=act_type, channels_policy_head=channels_policy_head, n_labels=n_labels,
                             select_policy_from_plane=select_policy_from_plane, use_se=False, channels=channels)
    # group value_out and policy_out together
    sym = mx.symbol.Group([value_out, policy_out])

    return sym
