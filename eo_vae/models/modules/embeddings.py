# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
from einops import rearrange, repeat

from .uvit_torch_utils import ACTIVATIONS

##########################
### Timestep Embedding ###
##########################


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """From https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/embeddings.py

    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args:
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, 'Timesteps should be a 1d-array'

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Timesteps(nn.Module):
    # From https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/embeddings.py
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        scale: int = 1,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb


class TimestepEmbedding(nn.Module):
    # From https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/embeddings.py
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = 'silu',
        out_dim: int = None,
        post_act_fn: str | None = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = ACTIVATIONS[act_fn]()

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = ACTIVATIONS[post_act_fn]()

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


#####################################
### Relative Positional Embedding ###
#####################################


class RelativePositionBias(nn.Module):
    def __init__(self, window_size, num_heads, compatibility=False):
        super().__init__()
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        self.window_size = window_size  # (H, W)
        self.num_heads = num_heads
        self.relative_position_index_cache = {}

        # Total number of relative positions = (2H - 1) * (2W - 1)
        win_H, win_W = (2 * window_size[0] - 1), (2 * window_size[1] - 1)
        self.win_H = win_H
        self.win_W = win_W

        # Learnable bias table
        if compatibility:
            self.relative_bias_table = nn.Parameter(
                torch.zeros(win_H * win_W, num_heads)
            )
        else:
            self.relative_bias_table = nn.Parameter(
                torch.zeros(win_H, win_W, num_heads)
            )

        self._cached_mask = None

    def compute_relative_position_index(self, window_size):
        if window_size not in self.relative_position_index_cache:
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(
                torch.meshgrid([coords_h, coords_w], indexing='ij')
            )  # (2, H, W)
            coords_flatten = coords.view(2, -1)  # (2, H*W)

            relative_coords = (
                coords_flatten[:, :, None] - coords_flatten[:, None, :]
            )  # (2, HW, HW)
            relative_coords = relative_coords.permute(
                1, 2, 0
            ).contiguous()  # (HW, HW, 2)
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to >= 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # (HW, HW)
            self.relative_position_index_cache[window_size] = relative_position_index
        return self.relative_position_index_cache[window_size]

    def get_full_relative_bias_table(self, window_size):
        H, W = window_size
        relative_bias_table = self.relative_bias_table.reshape(
            (self.win_H, self.win_W, self.num_heads)
        )

        assert H >= self.window_size[0] and W >= self.window_size[1], (
            'Not supported yet'
        )
        pad_H = H - self.window_size[0]
        pad_W = W - self.window_size[1]
        relative_bias_table = torch.nn.functional.pad(
            relative_bias_table,
            (0, 0, pad_W, pad_W, pad_H, pad_H),
            mode='constant',
            value=-(10**7),
        )
        assert relative_bias_table.shape == (H * 2 - 1, W * 2 - 1, self.num_heads)
        return relative_bias_table

    def forward(self, grid_shape):
        """Output: (num_heads, HW, HW)
        Can be added to attention logits.
        """
        B, H, W = grid_shape

        # Get the relative position index and (padded) bias table
        relative_position_index = self.compute_relative_position_index((H, W))
        relative_bias_table = self.get_full_relative_bias_table((H, W))
        relative_bias_table = relative_bias_table.reshape(
            -1, relative_bias_table.shape[-1]
        )  # (H*W, num_heads)

        # Get the relative bias for the current grid size
        relative_bias = relative_bias_table[
            relative_position_index.view(-1)
        ]  # (HW*HW, num_heads)
        relative_bias = relative_bias.view(H * W, H * W, self.num_heads)
        relative_bias = rearrange(
            relative_bias, 'N M H -> H N M'
        )  # (num_heads, HW, HW)
        mask = repeat(
            relative_bias, 'H N M -> (B H) N M', B=B
        )  # (batch_size*num_heads, HW, HW)
        return mask


class RelativeBiasAttentionWrapper(nn.Module):
    def __init__(self, attention, grid_size, compatibility=False):
        super().__init__()
        self.processor = attention.processor
        self.relative_position_bias = RelativePositionBias(
            (grid_size, grid_size), attention.heads, compatibility=compatibility
        )
        self.num_heads = attention.heads

    def forward(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        assert attention_mask is None
        B, N, D = hidden_states.shape
        grid_size = int(N**0.5)
        attention_mask = self.relative_position_bias(
            (hidden_states.shape[0], grid_size, grid_size)
        )

        assert attention_mask.shape == (B * self.num_heads, N, N), (
            f'Expected attention_mask shape {(B * self.num_heads, N, N)}, got {attention_mask.shape}'
        )

        return self.processor(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
            *args,
            **kwargs,
        )


####################################
### Learned Positional Embedding ###
####################################


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, embeds_shape):
        super().__init__()
        self.embeds = nn.Parameter(torch.zeros(embeds_shape))
        nn.init.normal_(self.embeds, mean=0.0, std=0.02)

    def forward(self, hidden_states: torch.Tensor):
        embeds = self.embeds
        nhidden_dims = hidden_states.ndim - 1

        if embeds.ndim == 3:
            if nhidden_dims == 2:
                embeds = rearrange(embeds, 'C H W -> (H W) C')

        assert hidden_states.shape[1:] == embeds.shape
        return hidden_states + embeds.unsqueeze(0)
