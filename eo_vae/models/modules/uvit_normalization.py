# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def get_norm_layer(
    norm_type: str,
    dim: int,
    *,
    heads: int = 1,
    num_groups: int = 32,
    eps: float = 1e-5,
    elementwise_affine: bool = True,
):
    if norm_type in [None, False]:
        return None
    elif norm_type == 'layer_norm':
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
    elif norm_type == 'ln_l2':
        return nn.LayerNorm(dim, elementwise_affine=False, bias=False)
    elif norm_type == 'layer_norm_across_heads':
        return nn.LayerNorm(dim * heads, eps=eps)
    elif norm_type == 'rms_norm':
        return RMSNorm(dim, eps=eps)
    elif norm_type == 'rms_norm_across_heads':
        return RMSNorm(dim * heads, eps=eps)
    elif norm_type == 'l2':
        return LpNorm(p=2, dim=-1, eps=eps)
    elif norm_type == 'group_norm':
        return nn.GroupNorm(
            num_channels=dim, num_groups=num_groups, eps=eps, affine=elementwise_affine
        )
    else:
        raise ValueError(f'unknown norm_type: {norm_type}.')


class RMSNorm(nn.Module):
    # From https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/normalization.py
    def __init__(
        self, dim, eps: float, elementwise_affine: bool = True, bias: bool = False
    ):
        super().__init__()

        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if isinstance(dim, numbers.Integral):
            dim = (dim,)

        self.dim = torch.Size(dim)

        self.weight = None
        self.bias = None

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
            if bias:
                self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        if self.weight is not None:
            # convert into half-precision if necessary
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)
            hidden_states = hidden_states * self.weight
            if self.bias is not None:
                hidden_states = hidden_states + self.bias
        else:
            hidden_states = hidden_states.to(input_dtype)

        return hidden_states


class LpNorm(nn.Module):
    def __init__(self, p: int = 2, dim: int = -1, eps: float = 1e-12):
        super().__init__()

        self.p = p
        self.dim = dim
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.normalize(hidden_states, p=self.p, dim=self.dim, eps=self.eps)


class RunningNormalizer(torch.nn.Module):
    def __init__(self, channels, momentum=0.01, epsilon=1e-5):
        super().__init__()
        self.register_buffer('mean', torch.zeros(channels))
        self.register_buffer('var', torch.ones(channels))
        self.momentum = momentum
        self.epsilon = epsilon

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        reduce_dims = tuple(range(x.dim()))[:-1]
        batch_mean = x.mean(dim=reduce_dims)
        batch_var = x.var(dim=reduce_dims, unbiased=False)

        self.mean.data.mul_(1 - self.momentum).add_(self.momentum * batch_mean)
        self.var.data.mul_(1 - self.momentum).add_(self.momentum * batch_var)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        std = (self.var + self.epsilon).sqrt()
        return (x - self.mean) / std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.update(x)
        return self.normalize(x)


class SequenceBatchNorm1d(nn.BatchNorm1d):
    def forward(self, x):
        x = rearrange(x, 'b l c -> b c l')
        x = super().forward(x)
        x = rearrange(x, 'b c l -> b l c')
        return x


class AdaGroupNorm2D(nn.Module):
    r"""GroupNorm layer modified to incorporate timestep embeddings from a 2D features map.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
        num_groups (`int`): The number of groups to separate the channels into.
        eps (`float`, *optional*, defaults to `1e-5`): The epsilon value to use for numerical stability.
    """

    def __init__(
        self, embedding_dim: int, out_dim: int, num_groups: int, eps: float = 1e-5
    ):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

        self.ctx_proj = nn.Conv2d(
            embedding_dim, out_dim * 2, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: torch.Tensor, ctx_emb: torch.Tensor) -> torch.Tensor:
        assert ctx_emb is not None
        ctx_emb = self.ctx_proj(ctx_emb)
        ctx_emb = torch.nn.functional.interpolate(
            ctx_emb, size=(x.shape[-2], x.shape[-1]), mode='nearest'
        )
        scale, shift = ctx_emb.chunk(2, dim=1)

        x = F.group_norm(x, self.num_groups, eps=self.eps)
        x = x * (1 + scale) + shift
        return x


class AdaLayerNorm(nn.Module):
    r"""Norm layer modified to incorporate context embeddings from 0D, 1D or 2D feature maps

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`, *optional*): The size of the embeddings dictionary.
        output_dim (`int`, *optional*):
        norm_elementwise_affine (`bool`, defaults to `False):
        norm_eps (`bool`, defaults to `False`):
        chunk_dim (`int`, defaults to `0`):
    """

    def __init__(self, embedding_dim: int, output_dim: int, eps: float = 1e-5):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim * 2)
        self.norm = nn.LayerNorm(output_dim, eps, False)

    def forward(self, x: torch.Tensor, ctx_emb: torch.Tensor) -> torch.Tensor:
        _, x_L, _ = x.shape

        # Regroup and project patches
        if ctx_emb.dim() == 4:  # 4D tensor (B, C, H, W)
            _, _, H, W = ctx_emb.shape
            ctx_emb = rearrange(ctx_emb, 'B C H W -> B H W C')
            ctx_emb = self.linear(self.silu(ctx_emb))

            # Upsample features
            r = max(1, int((x_L / H / W) ** 0.5))
            ctx_emb = repeat(ctx_emb, 'B W H C -> B (W R1 H R2) C', R1=r, R2=r)
        elif ctx_emb.dim() == 3:  # 3D tensor (B, L, C)
            ctx_emb = self.linear(self.silu(ctx_emb))
            ctx_emb = repeat(
                ctx_emb, 'B L C -> B (L R) C', R=max(1, x_L // ctx_emb.shape[1])
            )
        elif ctx_emb.dim() == 2:  # 2D tensor (B, C)
            ctx_emb = self.linear(self.silu(ctx_emb))
            ctx_emb = repeat(ctx_emb, 'B C -> B L C', L=x_L)
        else:
            raise ValueError(
                f'Unsuported ctx_emb shape: {ctx_emb.shape}. Expected 2D tensor (B, C), 3D tensor (B, L, C) or 4D tensor (B, C, H, W).'
            )

        shift, scale = ctx_emb.chunk(2, dim=2)
        assert shift.shape == x.shape
        x = self.norm(x) * (1 + scale) + shift
        return x
