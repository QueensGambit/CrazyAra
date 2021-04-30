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
    mix_conv, get_stem, value_head, policy_head, convolution_block_attention_module, ca_se, sa_se, cm_se, sm_se, eca_se, get_norm_layer, get_se_layer


def bottleneck_residual_block_v2(data, channels, channels_operating, name, kernel, act_type='hard_swish', norm_type="ic", use_se=False):
    """
    Returns a residual block without any max pooling operation
    :param data: Input data
    :param channels: Number of filters for all CNN-layers
    :param name: Name for the residual block
    :param act_type: Activation function to use
    :param se_ratio: Squeeze excitation ratio
    :param use_se: Boolean if a squeeze excitation module will be used
    :param se_type: Squeeze excitation module type. Available [None, "se", "cbam", "ca_se", "cm_se", "sa_se", "sm_se"]
    :return: symbol
    """
    if use_se:
        se = eca_se(data, channels, name=name + '_se', use_hard_sigmoid=True)
        conv1 = mx.sym.Convolution(data=se, num_filter=channels_operating, kernel=(1, 1), pad=(0, 0),
                                   no_bias=True, name=name + '_conv1')
    else:
        conv1 = mx.sym.Convolution(data=data, num_filter=channels_operating, kernel=(1, 1), pad=(0, 0),
                                   no_bias=True, name=name + '_conv1')
    bn1 = get_norm_layer(data=conv1, norm_type=norm_type, name=name + '_bn1')
    act1 = get_act(data=bn1, act_type=act_type, name=name + '_act1')
    conv2 = mx.sym.Convolution(data=act1, num_filter=channels_operating, kernel=(kernel, kernel), stride=(1, 1),
                               num_group=channels_operating, pad=(kernel // 2, kernel // 2), no_bias=True,
                               name=name + '_conv2')
    bn2 = get_norm_layer(data=conv2, norm_type=norm_type, name=name + '_bn2')
    act2 = get_act(data=bn2, act_type=act_type, name=name + '_act2')
    conv3 = mx.sym.Convolution(data=act2, num_filter=channels, kernel=(1, 1), pad=(0, 0),
                               no_bias=True, name=name + '_conv3')
    bn3 = get_norm_layer(data=conv3, norm_type=norm_type, name=name + '_bn3')
    sum = mx.sym.broadcast_add(bn3, data, name=name+'_add')

    return sum


def sandglass_block(data, channels, channels_reduced, name, kernel, act_type='relu', norm_type="bn", se_type="eca_se"):
    """
    Rethinking Bottleneck Structure for EfficientMobile Network Design, D. Zhou and Q. Hou et al.
    """
    if se_type:
        data_input = get_se_layer(data, channels, se_type, name=name + '_se', use_hard_sigmoid=True)
    else:
        data_input = data
    conv1 = mx.sym.Convolution(data=data_input, num_filter=channels,  kernel=(kernel, kernel),
                               pad=(kernel // 2, kernel // 2), num_group=channels,
                               no_bias=True, name=name + '_conv1')
    bn1 = get_norm_layer(data=conv1, norm_type=norm_type, name=name + '_bn1')
    act1 = get_act(data=bn1, act_type=act_type, name=name + '_act1')
    conv2 = mx.sym.Convolution(data=act1, num_filter=channels_reduced, kernel=(1, 1), pad=(0, 0),
                               no_bias=False, name=name + '_conv2')
    conv3 = mx.sym.Convolution(data=conv2, num_filter=channels, kernel=(1, 1), pad=(0, 0),
                               no_bias=True, name=name + '_conv3')
    bn2 = get_norm_layer(data=conv3, norm_type=norm_type, name=name + '_bn2')
    act2 = get_act(data=bn2, act_type=act_type, name=name + '_act2')
    conv4 = mx.sym.Convolution(data=act2, num_filter=channels,  kernel=(kernel, kernel),
                               pad=(kernel // 2, kernel // 2), num_group=channels,
                               no_bias=False, name=name + '_conv4')
    sum = mx.sym.broadcast_add(conv4, data, name=name+'_add')
    return sum


def rise_mobile_v3_symbol(channels=256, channels_reduced=128, act_type='relu',
                          channels_value_head=8, channels_policy_head=81, value_fc_size=256, dropout_rate=0.15,
                          grad_scale_value=0.01, grad_scale_policy=0.99,
                          select_policy_from_plane=True, kernels=None, n_labels=4992, norm_type="bn",
                          se_types=None, use_avg_features=False):
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

    data = get_stem(data=orig_data, channels=channels, act_type=act_type, kernel=1)

    if kernels is None:
        kernels = [3] * 13

    for idx, cur_kernel in enumerate(kernels):
        #cur_channels = channels_operating
        #if idx < len(kernels) - 5:
        #    use_squeeze_excitation = False
        #else:
        #    use_squeeze_excitation = use_se
        #if cur_kernel == 5:
        #    cur_channels *= 0.75
        #    cur_channels = int(cur_channels)

        #if idx >= len(kernels) / 2:
        #    cur_act_type = act_type
        #else:
        #    cur_act_type = "relu"

        data = sandglass_block(data=data, channels=channels, channels_reduced=channels_reduced, name='sandglass_%d' % idx,
                               act_type=act_type, norm_type=norm_type, se_type=se_types[idx], kernel=cur_kernel)
        #data = bottleneck_residual_block_v2(data=data, channels=channels, channels_operating=cur_channels,
        #                                      kernel=cur_kernel, name='dconv_%d' % idx, act_type=cur_act_type,
        #                                      norm_type=norm_type, use_se=use_squeeze_excitation)

    if dropout_rate != 0:
        data = mx.sym.Dropout(data, p=dropout_rate)

    channels_policy_head = channels # 256
    value_out = value_head(data=data, act_type=act_type, use_se=False, channels_value_head=channels_value_head,
                           value_fc_size=value_fc_size, use_mix_conv=False, grad_scale_value=grad_scale_value,
                           orig_data=orig_data, use_avg_features=use_avg_features)
    policy_out = policy_head(data=data, act_type=act_type, channels_policy_head=channels_policy_head, n_labels=n_labels,
                             select_policy_from_plane=select_policy_from_plane, use_se=False, channels=channels_policy_head,
                             grad_scale_policy=grad_scale_policy)
    # group value_out and policy_out together
    sym = mx.symbol.Group([value_out, policy_out])

    return sym
