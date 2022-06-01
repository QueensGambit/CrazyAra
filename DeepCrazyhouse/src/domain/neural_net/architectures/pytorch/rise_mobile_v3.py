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
from torch.nn import Sequential, Conv2d, BatchNorm2d, Module
from torch import Tensor
from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.builder_util import get_act, _ValueHead, _PolicyHead, _Stem, get_se


class _BottlekneckResidualBlock(Module):

    def __init__(self, channels, channels_operating, kernel=3, act_type='relu', se_type=None, bn_mom=0.9):
        """
        Returns a residual block without any max pooling operation
        :param channels: Number of filters for all CNN-layers
        :param name: Name for the residual block
        :param act_type: Activation function to use
        :param se_type: Squeeze excitation module that will be used
        :return: symbol
        """
        super(_BottlekneckResidualBlock, self).__init__()

        self.se_type = se_type
        if se_type:
            self.se = get_se(se_type=se_type, channels=channels, use_hard_sigmoid=False)
        self.body = Sequential(Conv2d(in_channels=channels, out_channels=channels_operating, kernel_size=(1, 1), bias=False),
                               BatchNorm2d(momentum=bn_mom, num_features=channels_operating),
                               get_act(act_type),
                               Conv2d(in_channels=channels_operating, out_channels=channels_operating, kernel_size=(kernel, kernel), padding=(kernel // 2, kernel // 2), bias=False, groups=channels_operating),
                               BatchNorm2d(momentum=bn_mom, num_features=channels_operating),
                               get_act(act_type),
                               Conv2d(in_channels=channels_operating, out_channels=channels, kernel_size=(1, 1), bias=False),
                               BatchNorm2d(momentum=bn_mom, num_features=channels))

    def forward(self, x):
        """
        Compute forward pass
        :param x: Input data to the block
        :return: Activation maps of the block
        """
        if self.se_type:
            return x + self.body(self.se(x))
        return x + self.body(x)


def _get_res_blocks(act_type, bn_mom, channels, channels_operating_init, channel_expansion, kernels, se_types):
    """Helper function which generates the residual blocks for Risev3"""

    channels_operating = channels_operating_init
    res_blocks = []

    for idx, kernel in enumerate(kernels):
        if kernel == 5:
            channels_operating_active = channels_operating - 32 * (idx // 2)
        else:
            channels_operating_active = channels_operating

        res_blocks.append(_BottlekneckResidualBlock(channels=channels,
                                                    channels_operating=channels_operating_active,
                                                    kernel=kernel, act_type=act_type,
                                                    se_type=se_types[idx], bn_mom=bn_mom))
        channels_operating += channel_expansion

    return res_blocks


class RiseV3():

    def __init__(self, nb_input_channels, board_height, board_width,
                  channels=256, channels_operating_init=224, channel_expansion=32, act_type='relu',
                  channels_value_head=8, channels_policy_head=81, value_fc_size=256, dropout_rate=0.15,
                  grad_scale_value=0.01, grad_scale_policy=0.99, grad_scale_wdl=None, grad_scale_ply=None,
                  select_policy_from_plane=True, kernels=None, n_labels=4992,
                  se_types=None, use_avg_features=False, use_wdl=False, use_plys_to_end=False,
                  use_mlp_wdl_ply=False, bn_mom=0.9,
                 ):
        """
        RISEv3 architecture
        :param channels: Main number of channels
        :param channels_operating: Initial number of channels at the start of the net for the depthwise convolution
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
        :param use_downsampling: If true in the middle of the network a strided convolution will be used to downsample
         the spatial dimensionality and the number of channels will be doubled.
        :param slice_scalars: If true, the scalar features will be sliced and processed in an independent MLP.
        Later the spatial and scalar embeddings will be merged again.
        :return: symbol
        """
        self.nb_input_channels = nb_input_channels

        if len(kernels) != len(se_types):
            raise Exception(f'The length of "kernels": {len(kernels)} must be the same as'
                            f' the length of "se_types": {len(se_types)}')

        valid_se_types = [None, "se", "cbam", "eca_se", "ca_se", "cm_se", "sa_se", "sm_se"]
        for se_type in se_types:
            if se_type not in valid_se_types:
                raise Exception(f"Unavailable se_type: {se_type}. Available se_types include {se_types}")

        res_blocks = _get_res_blocks(act_type, bn_mom, channels, channels_operating_init, channel_expansion, kernels, se_types)

        self.body_spatial = Sequential(
            _Stem(channels=channels, bn_mom=bn_mom, act_type=act_type, nb_input_channels=nb_input_channels),
            *res_blocks,
        )
        self.nb_body_spatial_out = channels * board_height * board_width

        # create the two heads which will be used in the hybrid fwd pass
        self.value_head = _ValueHead(board_height, board_width, channels, channels_value_head, value_fc_size, bn_mom, act_type)
        self.policy_head = _PolicyHead(board_height, board_width, channels, channels_policy_head, n_labels,
                                       bn_mom, act_type, select_policy_from_plane)

    def forward(self, x, hidden_state: Tensor = None):
        """
        Implementation of the forward pass of the full network
        Uses a broadcast add operation for the shortcut and the output of the residual block
        :param x: Input to the ResidualBlock
        :param hidden_state: Input to lstm
        :return: Value & Policy Output
        """

        out = self.body_spatial(x)
        out = out.view(*out.shape[:-3], self.nb_body_spatial_out)

        # use the output to create value/policy predictions
        value = self.value_head(out)
        policy = self.policy_head(out)

        return value, policy


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

    model = RiseV3(nb_input_channels=args.input_shape[0], board_height=args.input_shape[1], board_width=args.input_shape[2],
                    channels=256, channels_operating_init=224, channel_expansion=32, act_type='relu',
                    channels_value_head=8, value_fc_size=256,
                    channels_policy_head=args.channels_policy_head,
                    grad_scale_value=args.val_loss_factor, grad_scale_policy=args.policy_loss_factor,
                    dropout_rate=0, select_policy_from_plane=args.select_policy_from_plane,
                    kernels=kernels, se_types=se_types, use_avg_features=False, n_labels=args.n_labels)
    return model
