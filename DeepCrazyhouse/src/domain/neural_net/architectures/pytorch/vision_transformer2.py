"""
@file: vision_transformer2.py
Created on 06.07.22
@project: CrazyAra
@author: queensgambit

Please describe what the content of this file is about
"""

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from vit_pytorch.vit import pair, Transformer
from DeepCrazyhouse.src.domain.neural_net.architectures.pytorch.builder_util import _ValueHead

input_shape = (52, 8, 8)


class ChessViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.,
                 use_wdl=False, use_plys_to_end=False, use_mlp_wdl_ply=False):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.use_wdl = use_wdl
        self.use_plys_to_end = use_plys_to_end

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_value_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
            nn.Tanh()
        )
        self.value_head = _ValueHead(board_height=image_size, board_width=image_size, channels=0, channels_value_head=0, fc0=256,
                                     nb_input_channels=256, use_wdl=use_wdl, use_plys_to_end=use_plys_to_end, use_mlp_wdl_ply=use_mlp_wdl_ply,
                                     use_flat_inputs=True, in_features=dim)
        self.mlp_policy_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        value_head_out = self.value_head(x)
        logits = self.mlp_policy_head(x)

        if self.use_plys_to_end and self.use_wdl:
            value_out, wdl_out, plys_to_end_out = value_head_out
            auxiliary_out = torch.cat((wdl_out, plys_to_end_out), dim=1)
            return value_out, logits, auxiliary_out, wdl_out, plys_to_end_out
        else:
            return value_head_out, logits
