"""
@file: trt_vit.py
Created on 07.07.22
@project: CrazyAra
@author: queensgambit

Please describe what the content of this file is about
Based on: TRT-ViT: TensorRT-oriented Vision Transformer, Xia et al.
https://arxiv.org/pdf/2205.09579.pdf

https://github.com/whai362/PVT/blob/v2/classification/pvt.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.builder_util import get_se, get_act, _ValueHead, _PolicyHead, process_value_policy_head
from DeepCrazyhouse.src.domain.variants.constants import NB_POLICY_MAP_CHANNELS, NB_LABELS
#from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.rise_mobile_v3 import _BottlekneckResidualBlock
from torch.nn import Sequential, Conv2d, BatchNorm2d, Module

class _BottlekneckResidualBlock(Module):

    def __init__(self, channels, channels_operating, kernel=3, act_type='relu', se_type=None, bn_mom=0.1):
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
            self.se = get_se(se_type=se_type, channels=channels, use_hard_sigmoid=True)
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


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, H, W, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.H = H
        self.W = W
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, self.H, self.W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class TransformerBlock(nn.Module):

    def __init__(self, dim, num_heads, H, W, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.path_embed = PatchEmbed(img_size=8, patch_size=2, in_chans=dim//4, embed_dim=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, H, W,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.path_embed(x)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        # assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
        #     f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        #B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        #H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x#, (H, W)


class ClassicResidualBlock(nn.Module):

    def __init__(self, channels, kernel=3, act_type='relu'):
        """
        Returns a residual block without any max pooling operation
        :param channels: Number of filters for all CNN-layers
        :param act_type: Activation function to use
        :return: symbol
        """
        super(ClassicResidualBlock, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(kernel, kernel), padding=(kernel // 2, kernel // 2), bias=False),
                                  nn.BatchNorm2d(num_features=channels),
                                  get_act(act_type),
                                  nn.Conv2d(in_channels=channels, out_channels=channels,
                                            kernel_size=(kernel, kernel), padding=(kernel // 2, kernel // 2),
                                            bias=False),
                                  nn.BatchNorm2d(num_features=channels))

    def forward(self, x):
        """
        Compute forward pass
        :param x: Input data to the block
        :return: Activation maps of the block
        """
        return x + self.body(x)


class ClassicBottleneckResidualBlock(nn.Module):

    def __init__(self, channels, channels_operating, kernel=3, act_type='relu'):
        """
        Returns a residual block without any max pooling operation
        :param channels: Number of filters for all CNN-layers
        :param act_type: Activation function to use
        :return: symbol
        """
        super(ClassicBottleneckResidualBlock, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=channels_operating, kernel_size=(1, 1), bias=False),
                                  nn.BatchNorm2d(num_features=channels_operating),
                                  get_act(act_type),
                                  nn.Conv2d(in_channels=channels_operating, out_channels=channels_operating, kernel_size=(kernel, kernel), padding=(kernel // 2, kernel // 2), bias=False),
                                  nn.BatchNorm2d(num_features=channels_operating),
                                  get_act(act_type),
                                  nn.Conv2d(in_channels=channels_operating, out_channels=channels, kernel_size=(1, 1), bias=False),
                                  nn.BatchNorm2d(num_features=channels))

    def forward(self, x):
        """
        Compute forward pass
        :param x: Input data to the block
        :return: Activation maps of the block
        """
        return x + self.body(x)


class MixBlockC(nn.Module):
    """
    As described in https://arxiv.org/pdf/2205.09579.pdf
    """
    def __init__(self, in_channels, num_heads, kernel, channels_operating, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.transformer_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, (1,1), stride=(1,1), padding=0),
            TransformerBlock(dim=in_channels*2, num_heads=num_heads, H=4, W=4),
        )

        #self.bottlekneck_block = ClassicBottleneckResidualBlock(channels=in_channels//2, channels_operating=in_channels//8, kernel=5)
        self.bottlekneck_block = _BottlekneckResidualBlock(channels=in_channels//2,
                                                    channels_operating=channels_operating,
                                                    kernel=kernel, act_type='relu',
                                                    se_type=None, bn_mom=0.1)


    def forward(self, x):
        x_1 = self.transformer_block(x)
        x_2 = self.bottlekneck_block(x_1)
        return x + torch.concat((x_1, x_2), dim=1)


class TrtViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        # num_classes,
        channels_policy_head,
        # dim,
        # depth,
        # heads,
        # mlp_mult,
        in_channels = 3,
        channels=320,
        stages = 3,
        dim_key = 32,
        dim_value = 64,
        dropout = 0.,
        use_wdl=False, use_plys_to_end=False,
        use_mlp_wdl_ply=False,
        select_policy_from_plane=True
    ):
        super().__init__()
        self.use_wdl = use_wdl
        self.use_plys_to_end = use_plys_to_end
        self.image_size = image_size

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, channels, (3,3), stride=(1,1), padding=1),
            # ClassicBottleneckResidualBlock(channels, channels//4),
            # ClassicBottleneckResidualBlock(channels, channels//4),
            # ClassicBottleneckResidualBlock(channels, channels//4),
            ClassicResidualBlock(channels),
            ClassicResidualBlock(channels),
            ClassicResidualBlock(channels),
            ClassicResidualBlock(channels),
            ClassicResidualBlock(channels),
            MixBlockC(channels, num_heads=4), #5),
            MixBlockC(channels, num_heads=4), #5),
            MixBlockC(channels, num_heads=4), #5),
        )

        self.value_head = _ValueHead(board_height=image_size, board_width=image_size, channels=channels, channels_value_head=8, fc0=256,
                                     nb_input_channels=256, use_wdl=use_wdl, use_plys_to_end=use_plys_to_end, use_mlp_wdl_ply=use_mlp_wdl_ply)
        self.policy_head = _PolicyHead(board_height=image_size, board_width=image_size, channels=channels, policy_channels=channels_policy_head, n_labels=NB_LABELS,
                                       select_policy_from_plane=select_policy_from_plane,)

    def forward(self, x):

        x = self.layers(x)

        return process_value_policy_head(x, self.value_head, self.policy_head, self.use_plys_to_end, self.use_wdl)
