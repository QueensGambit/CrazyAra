"""
Upgrade of RISE architecture using mixed depthwise convolutions, preactivation residuals and dropout
 proposed by Johannes Czech:

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
import torch
from torch.nn import Sequential, Conv2d, BatchNorm2d, Module
from timm.models.layers import DropPath

from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.builder_util import get_act, _ValueHead, _PolicyHead,\
    _Stem, get_se, process_value_policy_head, _BottlekneckResidualBlock, ClassicalResidualBlock, round_to_next_multiple_of_32
from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.next_vit_official_modules import NCB
from DeepCrazyhouse.configs.train_config import TrainConfig
from DeepCrazyhouse.src.domain.variants.constants import NB_POLICY_MAP_CHANNELS, NB_LABELS
from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.next_vit_official_modules import NTB


def _get_res_blocks(act_types, channels, channels_operating_init, channel_expansion, kernels, se_types, use_transformers, path_dropout_rates, conv_block, kernel_5_channel_ratio, round_channels_to_next_32):
    """Helper function which generates the residual blocks for Risev3"""

    channels_operating = channels_operating_init
    res_blocks = []

    for idx, kernel in enumerate(kernels):
        if kernel == 5:
            if kernel_5_channel_ratio is None:
                channels_operating_active = channels_operating - 32 * (idx // 2)
            else:
                channels_operating_active = int(channels_operating * kernel_5_channel_ratio + 0.5) # 0.68 95 #- 32 * (idx // 2)
        else:
            channels_operating_active = channels_operating

        if round_channels_to_next_32:
            channels_operating_active = round_to_next_multiple_of_32(channels_operating_active)
            channels = round_to_next_multiple_of_32(channels)

        if use_transformers[idx]:
            res_blocks.append(NTB(channels, channels, path_dropout=path_dropout_rates[idx]))
        elif conv_block == "mobile_bottlekneck_res_block":
            res_blocks.append(_BottlekneckResidualBlock(channels=channels,
                                                        channels_operating=channels_operating_active,
                                                        use_depthwise_conv=True,
                                                        kernel=kernel, act_type=act_types[idx],
                                                        se_type=se_types[idx],
                                                        path_dropout=path_dropout_rates[idx]))
        elif conv_block == "bottlekneck_res_block":
            res_blocks.append(_BottlekneckResidualBlock(channels=channels,
                                                        channels_operating=channels_operating_active,
                                                        use_depthwise_conv=False,
                                                        kernel=kernel, act_type=act_types[idx],
                                                        se_type=se_types[idx],
                                                        path_dropout=path_dropout_rates[idx]))
        elif conv_block == "classical_res_block":
            res_blocks.append(ClassicalResidualBlock(channels, act_types[idx], se_type=se_types[idx], path_dropout=path_dropout_rates[idx]))
        elif conv_block == "next_conv_block":
            res_blocks.append(NCB(channels, channels, stride=1, se_type=se_types[idx], path_dropout=path_dropout_rates[idx]))

        channels_operating += channel_expansion

    return res_blocks


class RiseV3(Module):

    def __init__(self, nb_input_channels, board_height, board_width,
                 channels=256, channels_operating_init=224, channel_expansion=32, act_types=None,
                 channels_value_head=8, channels_policy_head=81, value_fc_size=256, dropout_rate=0.15,
                 select_policy_from_plane=True, kernels=None, n_labels=4992,
                 se_types=None, use_avg_features=False, use_wdl=False, use_plys_to_end=False,
                 use_mlp_wdl_ply=False,
                 use_transformers=None, path_dropout=0, conv_block="mobile_bottlekneck_res_block",
                 kernel_5_channel_ratio=None, round_channels_to_next_32=False,
                 ):
        """
        RISEv3 architecture
        :param channels: Main number of channels
        :param channels_operating: Initial number of channels at the start of the net for the depthwise convolution
        :param channel_expansion: Number of channels to add after each residual block
        :param act_types: Activation type to use as a list of layers.
        :param channels_value_head: Number of channels for the value head
        :param value_fc_size: Number of units in the fully connected layer of the value head
        :param channels_policy_head: Number of channels for the policy head
        :param dropout_rate: Droput factor to use. If 0, no dropout will be applied. Value must be in [0,1]
        :param select_policy_from_plane: True, if policy head type shall be used
        :param kernels: List of kernel sizes used for the residual blocks. The length of the list corresponds to the number
        of residual blocks.
        :param n_labels: Number of policy target labels (used for select_policy_from_plane=False)
        :param se_types: List of squeeze exciation modules to use for each residual layer.
         The length of this list must be the same as len(kernels). Available types:
        - "se": Squeeze excitation block - Hu et al. - https://arxiv.org/abs/1709.01507
        - "cbam": Convolutional Block Attention Module (CBAM) - Woo et al. - https://arxiv.org/pdf/1807.06521.pdf
        - "ca_se": Same as "se"
        - "cm_se": Squeeze excitation with max operator
        - "sa_se": Spatial excitation with average operator
        - "sm_se": Spatial excitation with max operator
         the spatial dimensionality and the number of channels will be doubled.
        Later the spatial and scalar embeddings will be merged again.
        :param use_wdl: If a win draw loss head shall be used
        :param use_plys_to_end: If a plys to end prediction head shall be used
        :param use_mlp_wdl_ply: If a small mlp with value output for the wdl and ply head shall be used
        :param path_dropout: Path dropout for stochastic depth
        :param conv_block: Base convolutional block ["mobile_bottlekneck_res_block", "bottlekneck_res_block", "classical_res_block", "next_conv_block"]
        :param kernel_5_channel_ratio: Downscale factor for channels_operating in case of 5x5 kernels
        :param round_channels_to_next_32: Rounds all number of channels within the network to the closest multiple of 32
        :return: symbol
        """
        super(RiseV3, self).__init__()
        self.nb_input_channels = nb_input_channels
        self.use_plys_to_end = use_plys_to_end
        self.use_wdl = use_wdl

        if round_channels_to_next_32:
            channels = round_to_next_multiple_of_32(channels)

        if se_types is None:
            se_types = [None] * len(kernels)
        if use_transformers is None:
            use_transformers = [None] * len(kernels)
        if act_types is None:
            act_types = ['relu'] * len(kernels)

        if len(kernels) != len(se_types):
            raise Exception(f'The length of "kernels": {len(kernels)} must be the same as'
                            f' the length of "se_types": {len(se_types)}')

        valid_se_types = [None, "se", "cbam", "eca_se", "ca_se", "cm_se", "sa_se", "sm_se"]
        for se_type in se_types:
            if se_type not in valid_se_types:
                raise Exception(f"Unavailable se_type: {se_type}. Available se_types include {se_types}")

        path_dropout_rates = [x.item() for x in torch.linspace(0, path_dropout, len(kernels))]  # stochastic depth decay rule
        res_blocks = _get_res_blocks(act_types, channels, channels_operating_init, channel_expansion, kernels, se_types, use_transformers, path_dropout_rates, conv_block, kernel_5_channel_ratio, round_channels_to_next_32)

        self.body_spatial = Sequential(
            _Stem(channels=channels, act_type=act_types[0], nb_input_channels=nb_input_channels),
            *res_blocks,
        )
        self.nb_body_spatial_out = channels * board_height * board_width

        # create the two heads which will be used in the hybrid fwd pass
        self.value_head = _ValueHead(board_height, board_width, channels, channels_value_head, value_fc_size,
                                     act_types[-1], False, nb_input_channels,
                                     use_wdl, use_plys_to_end, use_mlp_wdl_ply)
        self.policy_head = _PolicyHead(board_height, board_width, channels, channels_policy_head, n_labels,
                                       act_types[-1], select_policy_from_plane)

    def forward(self, x):
        """
        Implementation of the forward pass of the full network
        Uses a broadcast add operation for the shortcut and the output of the residual block
        :param x: Input to the ResidualBlock
        :return: Value & Policy Output
        """
        out = self.body_spatial(x)

        return process_value_policy_head(out, self.value_head, self.policy_head, self.use_plys_to_end, self.use_wdl)


def get_rise_v33_model(args):
    """
    Wrapper definition for RISEv3.3.
    :return: pytorch model object
    """
    kernels = [3] * 15
    kernels[7] = 5
    kernels[11] = 5
    kernels[12] = 5
    kernels[13] = 5

    se_types = [None] * len(kernels)
    se_types[5] = "eca_se"
    se_types[8] = "eca_se"
    se_types[12] = "eca_se"
    se_types[13] = "eca_se"
    se_types[14] = "eca_se"

    act_types = ['relu'] * len(kernels)

    model = RiseV3(nb_input_channels=args.input_shape[0], board_height=args.input_shape[1], board_width=args.input_shape[2],
                   channels=256, channels_operating_init=224, channel_expansion=32, act_types=act_types,
                   channels_value_head=8, value_fc_size=256,
                   channels_policy_head=args.channels_policy_head,
                   dropout_rate=0, select_policy_from_plane=args.select_policy_from_plane,
                   kernels=kernels, se_types=se_types, use_avg_features=False, n_labels=args.n_labels,
                   use_wdl=args.use_wdl, use_plys_to_end=args.use_plys_to_end, use_mlp_wdl_ply=args.use_mlp_wdl_ply,
                   )
    return model


def get_rise_v2_model(args):
    """
    Wrapper definition for RISEv2.0
    :return: pytorch model object
    """
    kernels = [3] * 13

    se_types = [None] * len(kernels)
    se_types[8] = "ca_se"
    se_types[9] = "ca_se"
    se_types[10] = "ca_se"
    se_types[11] = "ca_se"
    se_types[12] = "ca_se"

    act_types = ['relu'] * len(kernels)

    model = RiseV3(nb_input_channels=args.input_shape[0], board_height=args.input_shape[1], board_width=args.input_shape[2],
                   channels=256, channels_operating_init=128, channel_expansion=64, act_types=act_types,
                   channels_value_head=8, value_fc_size=256,
                   channels_policy_head=args.channels_policy_head,
                   dropout_rate=0, select_policy_from_plane=args.select_policy_from_plane,
                   kernels=kernels, se_types=se_types, use_avg_features=False, n_labels=args.n_labels,
                   use_wdl=args.use_wdl, use_plys_to_end=args.use_plys_to_end, use_mlp_wdl_ply=args.use_mlp_wdl_ply,
                   )
    return model


def get_rise_v2_model_by_train_config(input_shape, tc: TrainConfig):
    args = create_args_by_train_config(input_shape, tc)
    model = get_rise_v2_model(args)
    return model


def get_rise_v33_model_by_train_config(input_shape, tc: TrainConfig):
    args = create_args_by_train_config(input_shape, tc)
    model = get_rise_v33_model(args)
    return model


def create_args_by_train_config(input_shape, tc):
    class Args:
        pass

    args = Args()
    args.input_shape = input_shape
    args.channels_policy_head = NB_POLICY_MAP_CHANNELS
    args.n_labels = NB_LABELS
    args.select_policy_from_plane = tc.select_policy_from_plane
    args.use_wdl = tc.use_wdl
    args.use_plys_to_end = tc.use_plys_to_end
    args.use_mlp_wdl_ply = tc.use_mlp_wdl_ply
    return args
