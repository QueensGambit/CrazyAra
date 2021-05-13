"""
@file: preact_resnet_se.py
Created on 01.05.21
@project: CrazyAra
@author: queensgambit

Descirption of a preactivation resnet with se layers
"""
import mxnet as mx
from DeepCrazyhouse.src.domain.neural_net.architectures.builder_util_symbol import value_head, policy_head, get_se_layer


def preact_residual_block(data, channels, name, act_type='relu', se_type="eca_se", kernel=3):
    """
    Returns a residual block without any max pooling operation
    :param data: Input data
    :param channels: Number of filters for all CNN-layers
    :param name: Name for the residual block
    :param act_type: Activation function to use
    :return:
    """
    bn1 = mx.sym.BatchNorm(data=data, name=name + '_bn1')
    if se_type:
        next_input = get_se_layer(bn1, channels, se_type, name=name + '_se', use_hard_sigmoid=True)
    else:
        next_input = bn1
    conv1 = mx.sym.Convolution(data=next_input, num_filter=channels, kernel=(kernel,kernel), pad=(kernel//2,kernel//2),
                               no_bias=True, name=name + '_conv1')
    bn1 = mx.sym.BatchNorm(data=conv1, name=name + '_bn2')
    act1 = mx.sym.Activation(data=bn1, act_type=act_type, name=name + '_act1')

    last_kernel = 3
    conv2 = mx.sym.Convolution(data=act1, num_filter=channels, kernel=(last_kernel,last_kernel), pad=(last_kernel//2,last_kernel//2),
                               no_bias=False, name=name + '_conv2')
    shortcut = data
    sum_out = mx.sym.broadcast_add(conv2, shortcut, name=name+'_add')

    return sum_out


def preact_resnet_se(channels=256, act_type='relu',
                     channels_value_head=8, channels_policy_head=81, value_fc_size=256, dropout_rate=0.15,
                     grad_scale_value=0.01, grad_scale_policy=0.99,
                     select_policy_from_plane=True, kernels=None, n_labels=4992, norm_type="bn",
                     se_types=None, use_avg_features=False, use_raw_features=True):
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
    data = mx.sym.Convolution(data=orig_data, num_filter=channels, kernel=(3, 3), pad=(3//2, 3//2),
                              no_bias=True, name="stem_conv0")

    if kernels is None:
        kernels = [3] * 13

    for idx, cur_kernel in enumerate(kernels):
        data = preact_residual_block(data=data, channels=channels, name='preact_res_%d' % idx,
                               act_type=act_type, se_type=se_types[idx], kernel=cur_kernel)

    if dropout_rate != 0:
        data = mx.sym.Dropout(data, p=dropout_rate)

    channels_policy_input = channels
    value_out = value_head(data=data, act_type=act_type, use_se=False, channels_value_head=channels_value_head,
                           value_fc_size=value_fc_size, use_mix_conv=False, grad_scale_value=grad_scale_value,
                           orig_data=orig_data, use_avg_features=use_avg_features, use_raw_features=use_raw_features,
                           use_bn=True)
    policy_out = policy_head(data=data, act_type=act_type, channels_policy_head=channels_policy_head, n_labels=n_labels,
                             select_policy_from_plane=select_policy_from_plane, use_se=False, channels=channels_policy_input,
                             grad_scale_policy=grad_scale_policy)
    # group value_out and policy_out together
    sym = mx.symbol.Group([value_out, policy_out])

    return sym

