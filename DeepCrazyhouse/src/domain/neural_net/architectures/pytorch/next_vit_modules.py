"""
@file: next_vit_modules.py
Created on 01.08.22
@project: CrazyAra
@author: queensgambit

https://raw.githubusercontent.com/wilile26811249/Next-ViT/main/model/module.py
"""

#from turtle import forward
from einops import rearrange
import torch
import torch.nn as nn


class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)

    def forward(self, x):
        return self.conv(x)


class MHCA(nn.Module):
    """
    Multi-Head Convolutional Attention (MHCA)
    Uniformly set head dim to 32 in all MHCA for fast inference
    speed with various date-type on TensorRT.

    Parameters
    ----------
    channel : int
        Number of channels.
    groups : int
        Number of groups.
    """
    def __init__(self, channel, groups = 32):
        super(MHCA, self).__init__()
        self.grouped_conv = nn.Conv2d(channel, channel, kernel_size = 3, padding = 1, groups = groups)
        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace = True)
        self.point_conv = nn.Conv2d(channel, channel, kernel_size = 1)

    def forward(self, x):
        return self.point_conv(self.relu(self.bn(self.grouped_conv(x))))


class Mlp(nn.Module):
    """
    Multi layer perceptron with dropout.
    Paper: https://arxiv.org/abs/2111.11418
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_features)
        self.relu = nn.ReLU(inplace = True)
        self.conv1 = nn.Conv2d(in_features, out_features, kernel_size = 1)
        self.conv2 = nn.Conv2d(out_features, out_features, kernel_size = 1)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(self.bn(x))))


class NCB(nn.Module):
    """
    Next-Convolution Block (NCB)
    Parameters
    ----------
    channel : int
        Number of channels.
    """
    def __init__(self, channel):
        super().__init__()
        self.mhca = MHCA(channel)
        self.mlp = Mlp(channel, channel)

    def forward(self, x):
        x = self.mhca(x) + x
        x = self.mlp(x) + x
        return x


class E_MHSA(nn.Module):
    """
    Effecient Multi-Head Self-Attention (E-MHSA)
    Parameters
    ----------
    dim : int
        Number of input channels.
    heads : int
        Number of heads.
    inner_dim : int
        Number of hidden channels for each head.
    dropout : float
        Dropout rate.
    stride : int
        Stride of the convolutional block.
    """
    def __init__(self, dim, heads = 8, inner_dim = 64 , dropout = 0., stride = 2):
        super().__init__()
        self.dim = dim
        self.inner_dim = inner_dim
        self.heads = heads
        self.scaled_factor = inner_dim ** -0.5

        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm2d(dim)
        self.avg_pool = nn.AvgPool2d(stride, stride = stride)

        self.fc_q = nn.Linear(dim, self.inner_dim * self.heads)
        self.fc_k = nn.Linear(dim, self.inner_dim * self.heads)
        self.fc_v = nn.Linear(dim, self.inner_dim * self.heads)
        self.fc_o = nn.Linear(self.inner_dim * self.heads, dim)


    def forward(self, x):
        b, c, h, w = x.shape
        x = self.bn(x)
        x_reshape = x.view(b, c, h * w).permute(0, 2, 1)  # [b, h * w, c]

         # Get q, k, v
        q = self.fc_q(x_reshape)
        # [b, heads, h * w, inner_dim]
        q = q.view(b, h * w, self.heads, self.inner_dim).permute(0, 2, 1, 3).contiguous()

        k = self.fc_k(x_reshape)
        k = k.view(b, self.heads * self.inner_dim, h, w)
        k = self.avg_pool(k)
        # [b, heads, h * w, inner_dim]
        k = rearrange(k, "b (head n) h w -> b head (h w) n", head = self.heads)

        v = self.fc_v(x_reshape)
        v = v.view(b, self.heads * self.inner_dim, h, w)
        v = self.avg_pool(v)
        # [b, heads, h * w, inner_dim]
        v = rearrange(v, "b (head n) h w -> b head (h w) n", head = self.heads)

        # Attention
        # replace the following einsum because "[Einsum]: ellipsis is not permitted in Einsum equation)" in TensorRT
        # attn = torch.einsum('... i d, ... j d -> ... i j', q, k) * self.scaled_factor
        attn = (q @ k.transpose(-2, -1)) * self.scaled_factor
        attn = torch.softmax(attn, dim = -1) # [b, heads, h * w, s_h * s_w], s_h = s_h // stride

        result = torch.matmul(attn, v).permute(0, 2, 1, 3)
        result = result.contiguous().view(b, h * w, self.heads * self.inner_dim)
        result = self.fc_o(result).view(b, self.dim, h, w)
        result = result + x
        return result


class NTB(nn.Module):
    """
    Next-Transposed Convolution Block (NTB)
    Parameters
    ----------
    in_channel : int
        Number of input channels.
    out_channel : int
        Number of output channels.
    shrink_ratio: int
        Shrink ratio of the channel rection.
    """
    def __init__(self, in_channel, out_channel, shrink_ratio = 0.75):
        super().__init__()
        first_part_dim = int(out_channel * shrink_ratio)
        second_part_dim = out_channel - first_part_dim
        self.point_conv1 = nn.Conv2d(in_channel, first_part_dim, kernel_size = 1)
        self.point_conv2 = nn.Conv2d(first_part_dim, second_part_dim, kernel_size = 1)

        self.e_mhsa = E_MHSA(first_part_dim)
        self.mhca = MHCA(second_part_dim)
        self.mlp = Mlp(out_channel, out_channel)

    def forward(self, x):
        x = self.point_conv1(x)
        first_part = self.e_mhsa(x) + x

        seconf_part = self.point_conv2(first_part)
        seconf_part = self.mhca(seconf_part) + seconf_part

        result = torch.cat([first_part, seconf_part], dim = 1)
        result = self.mlp(result) + result
        return result


class PatchEmbed(nn.Module):
    """
    Patch Embedding (PE)
    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super(PatchEmbed, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size = 2, stride = 2)
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(self.avgpool(x))


class Stem(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Stem, self).__init__()
        self.conv1 = Conv3x3(in_channels, 64, stride = 2)
        self.conv2 = Conv3x3(64, 32, stride = 1)
        self.conv3 = Conv3x3(32, 64, stride = 1)
        self.conv4 = Conv3x3(64, out_channels, stride = 2)

    def forward(self, x):
        return self.conv4(self.conv3(self.conv2(self.conv1(x))))


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, ncb_layers, nct_layers, repeat):
        super(Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ncb_layers = ncb_layers
        self.nct_layers = nct_layers
        self.repeat = repeat

        block = []
        for num in range(repeat):
            if num != repeat - 1:
                block += self._make_layer(self.in_channels, self.in_channels,
                                          self.ncb_layers, self.nct_layers)
            else:
                block += self._make_layer(self.in_channels, self.out_channels,
                                          self.ncb_layers, self.nct_layers)
        self.block = nn.Sequential(*block)

    def _make_layer(self, in_channels, out_channels, ncb_layers, nct_layers):
        self.sub_layers = []
        for _ in range(ncb_layers):
            self.sub_layers +=  [NCB(in_channels)]
        for _ in range(nct_layers):
            self.sub_layers +=  [NTB(in_channels, out_channels)]
        return nn.Sequential(*self.sub_layers)

    def forward(self, x):
        return self.block(x)


if __name__ == "__main__":
    x = torch.randn(1, 256, 16, 16)
    x_input = torch.randn(1, 3, 64, 64)

    stem = Stem(3, 64)
    patch_embed = PatchEmbed(256, 384)
    ncb = NCB(256)
    ntb = NTB(256, 384)
    block = Block(256, 384, 4, 1, 6)

    stem_x = stem(x_input)
    patch_x = patch_embed(x)
    ncb_x = ncb(x)
    ntb_x = ntb(ncb_x)
    block_x = block(x)

    print(f"stem_x: {stem_x.shape}")  # [1, 64, 16, 16]
    print(f"patch_x.shape: {patch_x.shape}")  # [1, 384, 32, 32]
    print(f"ncb_x.shape: {ncb_x.shape}")  # [1, 256, 64, 64]
    print(f"ntb_x.shape: {ntb_x.shape}")  # [1, 256, 64, 64]
    print(f"block_x.shape: {block_x.shape}")  # [1, 384, 16, 16]

