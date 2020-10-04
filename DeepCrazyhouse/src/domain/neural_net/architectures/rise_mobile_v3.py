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
    * Convolutional Block Attention Module (CBAM),
    Sanghyun Woo, Jongchan Park, Joon-Young Lee, In So Kweon
    https://arxiv.org/pdf/1807.06521.pdf
"""
import mxnet as mx
from DeepCrazyhouse.src.domain.neural_net.architectures.builder_util_symbol import get_act, channel_squeeze_excitation, \
    mix_conv, get_depthwise_stem, value_head, policy_head_depthwise, convolution_block_attention_module,\
    ca_se, sa_se, cm_se, sm_se, eca_se, policy_head


def preact_residual_dmixconv_block(data, channels, channels_operating, name, kernels=None, act_type='relu',
                                   se_ratio=4, se_type="se"):
    """
    Returns a residual block without any max pooling operation
    :param data: Input data
    :param channels: Number of filters for all CNN-layers
    :param name: Name for the residual block
    :param act_type: Activation function to use
    :param se_ratio: Squeeze excitation ratio
    :param use_se: Boolean if a squeeze excitation module will be used
    :param se_type: Squeeze excitation module type. Available [None, "se", "cbam", "ca_se", "eca_se", cm_se", "sa_se", "sm_se"]
    :return: symbol
    """
    out = mx.sym.BatchNorm(data=data, name=name + '_bn1')
    if se_type is not None:
        if se_type == "se":
           out = channel_squeeze_excitation(out, channels, name=name+'_se', ratio=se_ratio, act_type=act_type,
                                            use_hard_sigmoid=True)
        elif se_type == "cbam":
            out = convolution_block_attention_module(out, channels, name=name+'_se', ratio=se_ratio,
                                                     act_type=act_type,
                                                     use_hard_sigmoid=True)
        elif se_type == "ca_se":
            out = ca_se(out, channels, name=name+'_ca_se', ratio=se_ratio, act_type=act_type, use_hard_sigmoid=True)
        elif se_type == "eca_se":
            out = eca_se(out, channels, name=name+'_eca_se', use_hard_sigmoid=True)
        elif se_type == "cm_se":
            out = cm_se(out, channels, name=name+'_cm_se', ratio=se_ratio, act_type=act_type, use_hard_sigmoid=True)
        elif se_type == "sa_se":
            out = sa_se(out, name=name+'sa_se', use_hard_sigmoid=True)
        elif se_type == "sm_se":
            out = sm_se(out, name=name+'sm_se', use_hard_sigmoid=True)
        else:
            raise Exception(f'Unsupported se_type "{se_type}"')
    conv1 = mx.sym.Convolution(data=out, num_filter=channels_operating, kernel=(1, 1), pad=(0, 0), no_bias=True,
                               name=name + '_conv1')
    bn2 = mx.sym.BatchNorm(data=conv1, name=name + '_bn2')
    act1 = get_act(data=bn2, act_type=act_type, name=name + '_act1')
    conv2 = mix_conv(data=act1, channels=channels_operating, kernels=kernels, name=name + 'conv2')
    bn3 = mx.sym.BatchNorm(data=conv2, name=name + '_bn3')
    out = get_act(data=bn3, act_type=act_type, name=name + '_act2')
    out = mx.sym.Convolution(data=out, num_filter=channels, kernel=(1, 1),
                               pad=(0, 0), no_bias=True, name=name + '_conv3')
    out_sum = mx.sym.broadcast_add(data, out, name=name + '_add')

    return out_sum


def rise_mobile_v3_symbol(channels=256, channels_operating_init=128, channel_expansion=64, act_type='relu',
                          channels_value_head=8, channels_policy_head=81, value_fc_size=256, dropout_rate=0.15,
                          grad_scale_value=0.01, grad_scale_policy=0.99,
                          select_policy_from_plane=True, kernels=None, n_labels=4992, se_ratio=4,
                          se_types="se", use_avg_features=False, use_raw_features=False, value_nb_hidden=7,
                          value_fc_size_hidden=256, value_dropout=0.15, use_more_features=False):
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
    :param grad_scale_value: Constant scalar which the gradient for the value outputs are being scaled width.
                            (0.01 is recommended for supervised learning with little data)
    :param grad_scale_policy: Constant scalar which the gradient for the policy outputs are being scaled width.
    :param select_policy_from_plane: True, if policy head type shall be used
    :param kernels: List of kernel sizes used for the residual blocks. The length of the list corresponds to the number
    of residual blocks.
    :param n_labels: Number of policy target labels (used for select_policy_from_plane=False)
    :param se_ratio: Reduction ration used in the squeeze excitation module
    :param se_types: List of squeeze exciation modules to use for each residual layer.
     The length of this list must be the same as len(kernels). Available types:
    - "se": Squeeze excitation block - Hu et al. - https://arxiv.org/abs/1709.01507
    - "cbam": Convolutional Block Attention Module (CBAM) - Woo et al. - https://arxiv.org/pdf/1807.06521.pdf
    - "ca_se": Same as "se"
    - "cm_se": Squeeze excitation with max operator
    - "sa_se": Spatial excitation with average operator
    - "sm_se": Spatial excitation with max operator
    :param use_avg_features: If true the value head receives the avg of the each channel of the original input
    :param use_raw_features: If true the value receives the raw features of the pieces positions one hot encoded
    :param value_nb_hidden: Number of hidden layers of the vlaue head
    :return: symbol
    """
    if len(kernels) != len(se_types):
        raise Exception(f'The length of "kernels": {len(kernels)} must be the same as'
                        f' the length of "se_types": {len(se_types)}')

    valid_se_types = [None, "se", "cbam", "eca_se", "ca_se", "cm_se", "sa_se", "sm_se"]
    for se_type in se_types:
        if se_type not in valid_se_types:
            raise Exception(f"Unavailable se_type: {se_type}. Available se_types include {se_types}")

    # get the input data
    orig_data = mx.sym.Variable(name='data')

    if use_more_features:
        w_pieces = mx.sym.slice_axis(orig_data, axis=1, begin=0, end=6)
        b_pieces = mx.sym.slice_axis(orig_data, axis=1, begin=6, end=12)

        w_pieces_mask = mx.sym.max(data=w_pieces, axis=1, keepdims=True)
        b_pieces_mask = mx.sym.max(data=b_pieces, axis=1, keepdims=True)
        pieces_mean = w_pieces_mask + b_pieces_mask

        orig_data = mx.sym.Concat(*[orig_data, pieces_mean, w_pieces_mask, b_pieces_mask], name='feature_concat_raw')

    data = get_depthwise_stem(data=orig_data, channels=channels, act_type=act_type)

    if kernels is None:
        kernels = [3] * 13

    cur_channels = channels_operating_init

    for idx, cur_kernels in enumerate(kernels):
        if cur_kernels[0] == 5:
            temp_channels = cur_channels // 2
        else:
            temp_channels = cur_channels
        data = preact_residual_dmixconv_block(data=data, channels=channels, channels_operating=temp_channels,
                                              kernels=cur_kernels, name='dconv_%d' % idx, act_type=act_type,
                                              se_ratio=se_ratio, se_type=se_types[idx])
        cur_channels += channel_expansion

    data = mx.sym.BatchNorm(data=data, name='stem_bn_final')
    # data = mx.sym.Convolution(data=data, num_filter=channels*4, kernel=(1, 1),
    #                            pad=(0, 0), no_bias=True, name='stem_conv1x1_final')
    data = get_act(data=data, act_type=act_type, name='stem_act_final')

    if dropout_rate != 0:
        data = mx.sym.Dropout(data, p=dropout_rate)

    value_out = value_head(data=data, act_type=act_type, use_se=False, channels_value_head=channels_value_head,
                           value_fc_size=value_fc_size, use_mix_conv=False, grad_scale_value=grad_scale_value,
                           orig_data=orig_data, use_avg_features=use_avg_features,
                           use_raw_features=use_raw_features, value_nb_hidden=value_nb_hidden, dropout_rate=value_dropout,
                           use_batchnorm=True, value_fc_size_hidden=value_fc_size_hidden)
    policy_out = policy_head(data=data, act_type=act_type, channels_policy_head=channels_policy_head, n_labels=n_labels,
                             select_policy_from_plane=select_policy_from_plane, channels=channels,
                             grad_scale_policy=grad_scale_policy)
    # group value_out and policy_out together
    sym = mx.symbol.Group([value_out, policy_out])

    return sym


def value_network(act_type='relu', channels_value_head=8, channels_policy_head=81, value_fc_size=256, dropout_rate=0,
                  grad_scale_value=1, grad_scale_policy=0, use_avg_features=False, use_raw_features=False,
                  value_nb_hidden=0, value_fc_size_hidden=32, use_batchnorm=False, use_conv_features=False):

    # get the input data
    orig_data = mx.sym.Variable(name='data')

    value_out = value_head(data=orig_data, act_type=act_type, use_se=False, channels_value_head=channels_value_head,
                           value_fc_size=value_fc_size, use_mix_conv=False, grad_scale_value=grad_scale_value,
                           orig_data=orig_data, use_avg_features=use_avg_features,
                           use_raw_features=use_raw_features, value_nb_hidden=value_nb_hidden,
                           value_fc_size_hidden=value_fc_size_hidden, use_batchnorm=use_batchnorm, dropout_rate=dropout_rate,
                           use_conv_features=use_conv_features)

    policy_out = mx.sym.Convolution(data=orig_data, num_filter=channels_policy_head, kernel=(1, 1), pad=(0, 0),
                                    no_bias=True, name="policy_conv1")
    policy_out = mx.sym.flatten(data=policy_out, name='policy_out')
    policy_out = mx.sym.SoftmaxOutput(data=policy_out, name='policy', grad_scale=grad_scale_policy)

    # group value_out and policy_out together
    sym = mx.symbol.Group([value_out, policy_out])

    return sym


def policy_network(channels=256, act_type='relu',
                          channels_policy_head=81,
                          grad_scale_policy=0.99,
                          select_policy_from_plane=True, n_labels=4992,
                          use_avg_features=False, use_raw_features=False, use_batchnorm=False, policy_nb_hidden=7,
                          policy_fc_size_hidden=256, policy_dropout=0.15, grad_scale_value=0):

    # get the input data
    orig_data = mx.sym.Variable(name='data')

    policy_out = policy_head_depthwise(data=orig_data, act_type=act_type, channels_policy_head=channels_policy_head,
                                       n_labels=n_labels, select_policy_from_plane=select_policy_from_plane,
                                       channels=channels, grad_scale_policy=grad_scale_policy,
                                       use_avg_features=use_avg_features, use_raw_features=use_raw_features,
                                       use_batchnorm=use_batchnorm, use_conv_features=False, orig_data=orig_data, dropout_rate=policy_dropout,
                                       policy_fc_size=policy_fc_size_hidden, policy_nb_hidden=policy_nb_hidden, policy_fc_size_hidden=policy_fc_size_hidden
                                       )

    orig_data = mx.sym.Flatten(orig_data)
    value_out = mx.sym.FullyConnected(data=orig_data, num_hidden=1)
    value_out = mx.sym.LinearRegressionOutput(data=value_out, name='value', grad_scale=grad_scale_value)

    # group value_out and policy_out together
    sym = mx.symbol.Group([value_out, policy_out])

    return sym
