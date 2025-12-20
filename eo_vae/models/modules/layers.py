# Apache-2.0 License

# Copyright (c) https://github.com/black-forest-labs/flux2

# Based on code: https://github.com/black-forest-labs/flux2/blob/main/src/flux2/autoencoder.py


import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


def Normalize(in_channels: int, num_groups: int = 32) -> nn.Module:
    """Group normalization with default of 32 groups."""
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=2, padding=0
        )

    def forward(self, x: Tensor):
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode='constant', value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode='nearest')
        x = self.conv(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int = None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.cond_dim = cond_dim

        self.norm1 = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        # --- NEW: Optional AdaIN Projection ---
        if self.cond_dim is not None:
            # Project embedding to [scale, shift] for out_channels
            self.emb_proj = nn.Linear(cond_dim, out_channels * 2)
            # Initialize to identity (scale=1, shift=0)
            nn.init.zeros_(self.emb_proj.bias)
            self.emb_proj.weight.data.zero_()
            # Initialize scale part of bias to 1
            self.emb_proj.bias.data[:out_channels] = 1.0

        self.norm2 = nn.GroupNorm(
            num_groups=32, num_channels=out_channels, eps=1e-6, affine=True
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x, emb=None):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        # --- NEW: Apply AdaIN if embedding is provided ---
        if self.cond_dim is not None and emb is not None:
            # emb: [B, cond_dim] -> [B, 2*out_channels]
            style = self.emb_proj(emb)
            # Reshape for broadcasting: [B, 2*C, 1, 1]
            style = style.unsqueeze(-1).unsqueeze(-1)
            scale, shift = style.chunk(2, dim=1)

            h = self.norm2(h)
            h = h * scale + shift  # AdaIN modulation
        else:
            h = self.norm2(h)

        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        h_ = self.norm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)

        # Reshape for SDPA: B C H W -> B 1 (H W) C
        b, c, h, w = q.shape
        q = rearrange(q, 'b c h w -> b 1 (h w) c').contiguous()
        k = rearrange(k, 'b c h w -> b 1 (h w) c').contiguous()
        v = rearrange(v, 'b c h w -> b 1 (h w) c').contiguous()

        # Flash Attention / SDPA
        h_ = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        h_ = rearrange(h_, 'b 1 (h w) c -> b c h w', h=h, w=w)
        return x + self.proj_out(h_)
