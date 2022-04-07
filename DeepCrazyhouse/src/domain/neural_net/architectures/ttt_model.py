import mxnet as mx
from DeepCrazyhouse.src.domain.neural_net.architectures.builder_util_symbol import get_act, channel_squeeze_excitation,\
    get_stem, policy_head, value_head
from DeepCrazyhouse.src.domain.neural_net.architectures.rise_mobile_v2 import bottleneck_residual_block

def get_ttt_symbol():
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
    kernel = 3
    channels = 32
    bc_res_blocks = [3, 3]
    use_se = False
    use_act = True
    act_type = "relu"
    data_variant = None
    channels_operating_init = 32
    channel_expansion = 0
    # first initial convolution layer followed by batchnormalization
    body = mx.sym.Convolution(data=data, num_filter=channels, kernel=(kernel, kernel), pad=(kernel // 2, kernel // 2),
                              no_bias=True, name="stem_conv0")
    body = mx.sym.BatchNorm(data=body, name='stem_bn0')
    if use_act:
        body = get_act(data=body, act_type=act_type, name='stem_act0')
    channels_operating = channels_operating_init

    # build residual towerFalse
    for idx, kernel in enumerate(bc_res_blocks):
        use_squeeze_excitation = use_se

        if idx < len(bc_res_blocks) - 5:
            use_squeeze_excitation = False
        body = bottleneck_residual_block(body, channels, channels_operating, name='bc_res_block%d' % idx, kernel=kernel,
                                         use_se=use_squeeze_excitation, act_type=act_type, data_variant=data_variant)
        channels_operating += channel_expansion

    # for idx, kernel in enumerate(res_blocks):
    #     if idx < len(res_blocks) - 5:
    #         use_squeeze_excitation = False
    #     else:
    #         use_squeeze_excitation = use_se
    #
    #     body = residual_block(body, channels, name='res_block%d' % idx, kernel=kernel,
    #                           use_se=use_squeeze_excitation, act_type=act_type)

    # for policy output
    policy_out = policy_head(data=body, channels=channels, act_type=act_type, channels_policy_head=1,
                             select_policy_from_plane=False, n_labels=9,
                             grad_scale_policy=1, use_se=False, no_bias=True)

    # for value output
    value_out = value_head(data=body, channels_value_head=2, value_kernelsize=1, act_type=act_type,
                           value_fc_size=16, grad_scale_value=1, use_se=False,
                           use_mix_conv=False)

    # group value_out and policy_out together
    sym = mx.symbol.Group([value_out, policy_out])

    return sym