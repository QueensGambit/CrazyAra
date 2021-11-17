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
from typing import Optional

import torch
from torch import nn
from torch.nn import Sequential, Conv2d, BatchNorm2d, Hardsigmoid, Hardswish, Module, Linear, BatchNorm1d, LSTM
from torch import split, cat, Tensor
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


from DeepCrazyhouse.src.nn.builder_util import get_act, MixConv, _ValueHead, _PolicyHead, _Stem, get_se, _ValueHeadFlat, _PolicyHeadFlat
from DeepCrazyhouse.src.nn.PommerModel import PommerModel
from DeepCrazyhouse.src.nn.builder_util import get_act, MixConv, _ValueHead, _PolicyHead, _Stem, get_se, TimeDistributed


class _PreactResidualMixConvBlock(Module):

    def __init__(self, channels, channels_operating, kernels=None, act_type='relu', se_ratio=4, se_type="se", bn_mom=0.9):
        """
        Returns a residual block without any max pooling operation
        :param channels: Number of filters for all CNN-layers
        :param name: Name for the residual block
        :param act_type: Activation function to use
        :param se_ratio: Squeeze excitation ratio
        :param use_se: Boolean if a squeeze excitation module will be used
        :param se_type: Squeeze excitation module type. Available [None, "se", "cbam", "ca_se", "cm_se", "sa_se", "sm_se"]
        :return: symbol
        """
        super(_PreactResidualMixConvBlock, self).__init__()

        self.body = Sequential(BatchNorm2d(momentum=bn_mom, num_features=channels),
                               Conv2d(in_channels=channels, out_channels=channels_operating, kernel_size=(1, 1), bias=False),
                               BatchNorm2d(momentum=bn_mom, num_features=channels_operating),
                               get_act(act_type),
                               MixConv(in_channels=channels_operating, out_channels=channels_operating, kernels=kernels),
                               BatchNorm2d(momentum=bn_mom, num_features=channels_operating),
                               get_act(act_type),
                               Conv2d(in_channels=channels_operating, out_channels=channels, kernel_size=(1, 1), bias=False),
                               )

    def forward(self, x):
        """
        Compute forward pass
        :param x: Input data to the block
        :return: Activation maps of the block
        """
        return x + self.body(x)


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


class _StridedBottlekneckBlock(Module):

    def __init__(self, channels, channels_operating, kernel=3, act_type='relu', se_type=None, bn_mom=0.9):
        """
        Returns a residual block without any max pooling operation
        :param channels: Number of filters for all CNN-layers
        :param name: Name for the residual block
        :param act_type: Activation function to use
        :param se_type: Squeeze excitation module that will be used
        :return: symbol
        """
        super(_StridedBottlekneckBlock, self).__init__()
        self.use_se = se_type
        if se_type:
            self.se = get_se(se_type=se_type, channels=channels, use_hard_sigmoid=False)
        self.body = Sequential(Conv2d(in_channels=channels, out_channels=channels_operating, kernel_size=(1, 1), bias=False),
                               BatchNorm2d(momentum=bn_mom, num_features=channels_operating),
                               get_act(act_type),
                               Conv2d(in_channels=channels_operating, out_channels=channels_operating,
                                      kernel_size=(kernel, kernel), stride=(2, 2), padding=(kernel // 2, kernel // 2), bias=False, groups=channels_operating),
                               BatchNorm2d(momentum=bn_mom, num_features=channels_operating),
                               get_act(act_type),
                               Conv2d(in_channels=channels_operating, out_channels=channels*2, kernel_size=(1, 1), bias=False),
                               BatchNorm2d(momentum=bn_mom, num_features=channels*2))

    def forward(self, x):
        """
        Compute forward pass
        :param x: Input data to the block
        :return: Activation maps of the block
        """
        if self.use_se:
            return self.body(self.se(x))
        return self.body(x)


class MLPBlock(Module):
    def __init__(self, act_type='relu', bn_mom=0.9, in_features=256, out_features=256):
        """
        Simple multi layer perceptron block
        :param act_type: Activation type
        :param bn_mom: Batch momentum
        :param in_features: Number of input features
        :param out_features: Nubmer of output features
        """
        super(MLPBlock, self).__init__()

        self.body = Sequential(
            Linear(in_features=in_features, out_features=out_features),
            BatchNorm1d(momentum=bn_mom, num_features=out_features),
            get_act(act_type))

    def forward(self, x):
        """
        Compute forward pass
        :param x: Input data to the block
        :return: Activation maps of the block
        """

        return self.body(x)


def _get_res_blocks(act_type, bn_mom, channels, channels_operating, kernels,
                    se_types, use_downsampling):
    """Helper function which generates the residual blocks for Risev3"""
    res_blocks = []
    expansion_factor = 1
    for idx, cur_kernel in enumerate(kernels):
        if use_downsampling and idx >= len(kernels) / 2:
            use_downsampling = False
            res_blocks.append(_StridedBottlekneckBlock(channels=channels, channels_operating=channels_operating * 2,
                                                       kernel=cur_kernel, act_type=act_type,
                                                       se_type=se_types[idx], bn_mom=bn_mom))
            expansion_factor = 2
        else:
            res_blocks.append(_BottlekneckResidualBlock(channels=channels * expansion_factor,
                                                        channels_operating=channels_operating * expansion_factor,
                                                        kernel=cur_kernel, act_type=act_type,
                                                        se_type=se_types[idx], bn_mom=bn_mom))
    return res_blocks


class RiseV3(PommerModel):

    def __init__(self, channels=128, nb_input_channels=18, channels_operating=256, act_type='relu',
                 channels_value_head=1, channels_policy_head=81, value_fc_size=256,
                 select_policy_from_plane=False, kernels=None, n_labels=6, se_ratio=4,
                 se_types=None, use_avg_features=False, use_raw_features=False, value_nb_hidden=7,
                 value_dropout=0.15, use_more_features=False, bn_mom=0.9,
                 board_height=11, board_width=11, use_downsampling=True, slice_scalars=False, nb_scalar_features=4,
                 use_flat_core=True, core_hidden=512, value_hidden=128, policy_hidden=128, use_lstm=False, lstm_layers=1
                 ):
        """
        RISEv3 architecture
        :param channels: Main number of channels
        :param channels_operating: Initial number of channels at the start of the net for the depthwise convolution
        :param channel_expansion: Number of channels to add after each residual block
        :param act_type: Activation type to use
        :param channels_value_head: Number of channels for the value head
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
        super(RiseV3, self).__init__(nb_input_channels=nb_input_channels, board_height=board_height, board_width=board_height,
                                     is_stateful=use_lstm, state_batch_dim=2)

        self.use_raw_features = use_raw_features
        self.nb_scalar_features = nb_scalar_features
        self.slice_scalars = slice_scalars
        self.use_flat_core = use_flat_core
        self.use_lstm = use_lstm

        if len(kernels) != len(se_types):
            raise Exception(f'The length of "kernels": {len(kernels)} must be the same as'
                            f' the length of "se_types": {len(se_types)}')

        valid_se_types = [None, "se", "cbam", "eca_se", "ca_se", "cm_se", "sa_se", "sm_se"]
        for se_type in se_types:
            if se_type not in valid_se_types:
                raise Exception(f"Unavailable se_type: {se_type}. Available se_types include {se_types}")

        res_blocks = _get_res_blocks(act_type, bn_mom, channels, channels_operating,
                                          kernels, se_types, use_downsampling)
        expansion_factor = 2 if use_downsampling else 1
        if use_downsampling:
            out_board_width = round(board_width * 0.5)
            out_board_height = round(board_height * 0.5)
        else:
            out_board_width = board_width
            out_board_height = board_height

        if self.slice_scalars:
            nb_input_channels -= self.nb_scalar_features

        self.body_spatial = TimeDistributed(Sequential(
            _Stem(channels=channels, bn_mom=bn_mom, act_type=act_type, nb_input_channels=nb_input_channels),
            *res_blocks,
        ), 4)
        self.nb_body_spatial_out = channels * out_board_width * out_board_height * expansion_factor

        if slice_scalars:
            mlp_blocks = [MLPBlock(act_type=act_type, bn_mom=bn_mom, in_features=channels_operating,
                                   out_features=channels_operating)] * len(res_blocks)
            self.body_scalars = TimeDistributed(Sequential(
                MLPBlock(act_type=act_type, bn_mom=bn_mom, in_features=nb_scalar_features,
                         out_features=channels_operating),
                *mlp_blocks,
            ), 2)

        # create the two heads which will be used in the hybrid fwd pass
        if self.use_flat_core:
            nb_merged_features = self.nb_body_spatial_out + (channels_operating if self.slice_scalars else 0)
            self.body_merged = TimeDistributed(MLPBlock(act_type=act_type, bn_mom=bn_mom, in_features=nb_merged_features,
                                        out_features=core_hidden), 2)

            if self.use_lstm:
                self.lstm = LSTM(input_size=core_hidden, hidden_size=core_hidden, batch_first=True,
                                 num_layers=lstm_layers)

            self.value_head = TimeDistributed(_ValueHeadFlat(in_features=core_hidden, fc0=value_hidden, bn_mom=bn_mom, act_type=act_type), 2)
            self.policy_head = TimeDistributed(_PolicyHeadFlat(in_features=core_hidden, fc0=policy_hidden, bn_mom=bn_mom, act_type=act_type, n_labels=n_labels), 2)
        else:
            self.value_head = _ValueHead(out_board_height, out_board_width, channels*expansion_factor, channels_value_head, value_fc_size, bn_mom, act_type)
            self.policy_head = _PolicyHead(out_board_height, out_board_width, channels*expansion_factor, channels_policy_head, n_labels,
                                                bn_mom, act_type, select_policy_from_plane)

    def get_state_shape(self, batch_size: int):
        # expected size by LSTM: 2 x (num_layers * num_directions, batch, hidden_size)
        return 2, self.lstm.num_layers, batch_size, self.lstm.hidden_size

    def forward(self, flat_input):
        """
        Implementation of the forward pass of the full network

        Uses a broadcast add operation for the shortcut and the output of the residual block

        :param flat_input: Flattened input (is reshaped internally)
        :return: Value, policy and auxiliary output
        """

        # input shape processing
        x, state_bf = self.unflatten(flat_input)

        # actual forward pass
        next_hidden_state_bf = None
        if self.use_flat_core:
            # shape: (batch[, sequence], channels, y, x) => (batch[, sequence], flat)
            if self.slice_scalars:
                spatial_features, scalar_features = split(x, self.nb_input_channels - self.nb_scalar_features, dim=-3)

                out_spatial = self.body_spatial(spatial_features)
                out_spatial = out_spatial.view(*out_spatial.shape[:-3], self.nb_body_spatial_out)

                # slices the first entry of each plane along channels
                if len(scalar_features.shape) == 4:
                    scalar_features = scalar_features[:, :, 0, 0]
                elif len(scalar_features.shape) == 5:
                    scalar_features = scalar_features[:, :, :, 0, 0]
                out_scalar = self.body_scalars(scalar_features)

                out = cat((out_spatial, out_scalar), dim=-1)
            else:
                out = self.body_spatial(x)
                out = out.view(*out.shape[:-3], self.nb_body_spatial_out)

            # merged body
            out = self.body_merged(out)

            # optional: lstm
            if self.use_lstm:
                # first ensure that we always have a sequence dimension for the lstm
                # batch without seq: 2 (batch, features) vs 3 (batch, sequence, features)
                no_sequence_dimension = len(out.shape) == 2
                if no_sequence_dimension:
                    # unsqueeze single input (add sequence dimension) => process a batch of 1-element sequences
                    out = out.unsqueeze(1)

                if state_bf is None:
                    out, next_hidden_state_pair = self.lstm(out)
                else:
                    # transpose batch_first state to expected input
                    state = self.transpose_state(state_bf)
                    out, next_hidden_state_pair = self.lstm(out, (state[0], state[1]))

                next_h = next_hidden_state_pair[0].unsqueeze(0)
                next_c = next_hidden_state_pair[1].unsqueeze(0)
                next_hidden_state = cat((next_h, next_c), dim=0)

                # transpose output to batch first
                next_hidden_state_bf = self.transpose_state(next_hidden_state)

                # return to original input dimension
                if no_sequence_dimension:
                    out = out.squeeze(1)
        else:
            out = self.body_spatial(x)

        # use the output to create value/policy predictions
        if self.use_raw_features:
            value = self.value_head(out, x)
        else:
            value = self.value_head(out)

        policy = self.policy_head(out)

        if self.use_lstm:
            # additional outputs which will be processed/decoded manually
            next_hidden_state_bf_flat = next_hidden_state_bf.view(next_hidden_state_bf.shape[0], -1)
            return value, policy, next_hidden_state_bf_flat
        else:
            return value, policy


# https://gist.github.com/kriventsov/7d710c316d611e855c590bdf48fdce63

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class ViT(PommerModel):
    def __init__(self, *, nb_input_channels, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3):
        # super().__init__()
        super(ViT, self).__init__(nb_input_channels=nb_input_channels, board_height=8, board_width=8,
                                     is_stateful=False, state_batch_dim=2)

        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, dim, mlp_dim)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes)
        )
        self.mlp_value_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, 1)
        )

    def forward(self, img, mask=None):

        img, state_bf = self.unflatten(img)

        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.transformer(x) #, mask)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_value_head(x), self.mlp_head(x)

    def get_state_shape(self, batch_size: int): #-> Tuple[int]:
        raise NotImplementedError

# from vit_pytorch import ViT
#
# v = ViT(
#     image_size = 256,
#     patch_size = 32,
#     num_classes = 1000,
#     dim = 1024,
#     depth = 6,
#     heads = 16,
#     mlp_dim = 2048,
#     dropout = 0.1,
#     emb_dropout = 0.1
# )
#
# img = torch.randn(1, 3, 256, 256)
