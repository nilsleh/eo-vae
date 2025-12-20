# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
from einops import rearrange

from .embeddings import LearnedPositionalEmbedding, RelativePositionBias
from .uvit_activations import get_linear_activation
from .uvit_attention import Attention
from .uvit_normalization import AdaLayerNorm

#########################
### Transformer parts ###
#########################


class FeedForward(nn.Module):
    r"""A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = 'geglu',
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        self.proj_in_act = get_linear_activation(
            activation_fn, dim, inner_dim=inner_dim, bias=bias
        )
        self.drop = nn.Dropout(dropout)
        self.proj_out = nn.Linear(inner_dim, dim_out, bias=bias)

        self.final_dropout = None
        if final_dropout:
            self.final_dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        hidden_states = self.proj_in_act(hidden_states)
        hidden_states = self.drop(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        if self.final_dropout is not None:
            hidden_states = self.final_dropout(hidden_states)
        return hidden_states


class ResidualBlock(FeedForward):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = nn.LayerNorm(self.dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return super().forward(self.norm(hidden_states)) + hidden_states


class ResidualMLP(nn.Module):
    def __init__(self, dim, in_dim=None, out_dim=None, n_layers=0, bias=True):
        super().__init__()

        if in_dim is None:
            in_dim = dim
        if out_dim is None:
            out_dim = dim

        self.layers = nn.Sequential()
        if n_layers == 0:
            self.append(nn.Linear(in_dim, out_dim, bias=bias))
        else:
            if in_dim != dim:
                self.layers.append(nn.Linear(in_dim, dim, bias=bias))
            for _ in range(n_layers):
                self.layers.append(ResidualBlock(dim))

            self.out_norm = nn.LayerNorm(dim)
            self.out_proj = nn.Linear(dim, out_dim, bias=bias)

    def forward(self, hidden_states: torch.Tensor, return_dict=False) -> torch.Tensor:
        """Forward pass of the residual MLP."""
        # Apply the MLP layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.out_norm(hidden_states)
        out_states = self.out_proj(hidden_states)

        if return_dict:
            return {'hidden_states': hidden_states, 'out_states': out_states}

        return out_states


##########################
### Transformer blocks ###
##########################


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        activation_fn: str = 'geglu',
        attention_bias: bool = False,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        ff_inner_dim: int | None = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        ada_norm: bool = False,
        ada_emb_dim: int | None = None,
        relative_bias=False,
        attn_window: int | None = None,  # Required for relative positional bias
        rope_theta=10000.0,
        attn_norm=None,
    ):
        super().__init__()

        # 1. Self-Attn
        self.ada_norm = ada_norm
        if ada_norm:
            self.norm1 = AdaLayerNorm(ada_emb_dim, dim, eps=norm_eps)
        else:
            self.norm1 = nn.LayerNorm(
                dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps
            )

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=None,
            out_bias=attention_out_bias,
            qk_norm=attn_norm,
        )

        # 2. Feed-forward
        self.norm2 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        # Positional bias
        self.relative_position_bias = None
        if relative_bias:
            self.relative_position_bias = RelativePositionBias(
                attn_window, num_attention_heads
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        ctx_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1. Self-Attention
        if self.ada_norm:
            norm_hidden_states = self.norm1(hidden_states, ctx_emb)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        if self.relative_position_bias is not None:
            assert attention_mask is None
            B, N, D = hidden_states.shape
            grid_size = int(N**0.5)
            attention_mask = self.relative_position_bias((B, grid_size, grid_size))

        attn_output = self.attn1(norm_hidden_states, attention_mask=attention_mask)
        hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm2(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states

        return hidden_states


################################
### Transformer architecture ###
################################


class VisionTransformer(nn.Module):
    # Based on code from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_2d.py
    """A Vision Transformer (ViT) block for processing 2D images or feature maps."""

    def __init__(
        self,
        inner_dim: int,
        patch_size: int = 1,
        in_channels: int | None = None,
        out_channels: int | None = None,
        dropout: float = 0.0,
        num_layers: int = 12,
        attn_norm: str | None = None,
        norm_elementwise_affine=True,
        num_attention_heads: int | None = None,
        max_auto_attention_heads: int = 16,
        act_fn: str = 'gelu',
        attention_bias: bool = False,
        ada_norm: bool = False,
        ada_emb_dim: int | None = None,  # Context embedding dimension for AdaNorm
        learned_pos_embed=False,
        sample_size: tuple[int]
        | None = None,  # Only for absolute positional embeddings
        relative_pos_embed=False,
        attn_window: int
        | None = None,  # Window size for Swin Transformer or Relative positional bias
        rope_theta: float = 100.0,  # 100 for 2D
        eps: float = 1e-5,
        out_norm: bool = True,
    ):
        ### Config ###
        self.inner_dim = inner_dim
        self.in_channels = in_channels = in_channels or inner_dim
        self.out_channels = out_channels = out_channels or inner_dim
        self.patch_size = patch_size

        super().__init__()

        ### Initialize the transformer blocks ###
        self.proj_in = torch.nn.Linear(in_channels, inner_dim)

        self.pos_embeddings = None
        if learned_pos_embed:
            assert sample_size is not None, (
                'sample_size must be provided for learned positional embeddings.'
            )
            H, W = sample_size
            self.pos_embeddings = LearnedPositionalEmbedding(
                (inner_dim, H // patch_size, W // patch_size)
            )

        # If num_attention_heads is not specified, use inner_dim // 64
        if num_attention_heads is None:
            num_attention_heads = min(max_auto_attention_heads, max(4, inner_dim // 64))

        transformer_blocks = [
            TransformerBlock(
                dim=inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=inner_dim // num_attention_heads,
                dropout=dropout,
                activation_fn=act_fn,
                attention_bias=attention_bias,
                norm_elementwise_affine=norm_elementwise_affine,
                norm_eps=eps,
                ada_norm=ada_norm,
                ada_emb_dim=ada_emb_dim,
                relative_bias=relative_pos_embed,
                attn_window=attn_window,
                rope_theta=rope_theta,
                attn_norm=attn_norm,
            )
            for _ in range(num_layers)
        ]
        self.transformer_blocks = nn.ModuleList(transformer_blocks)

        self.norm_out = None
        if out_norm:
            self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=eps)
        self.proj_out = torch.nn.Linear(inner_dim, out_channels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        ctx_emb: torch.Tensor | None = None,
        out_2d_map=False,
        return_dict=False,
    ) -> torch.Tensor:
        # 1. Input
        if hidden_states.dim() == 4:
            _, _, H, W = hidden_states.shape
            H, W = H // self.patch_size, W // self.patch_size

            hidden_states = rearrange(
                hidden_states,
                'b c (h p1) (w p2) -> b (h w) (c p1 p2)',
                p1=self.patch_size,
                p2=self.patch_size,
            )
        else:
            assert hidden_states.dim() == 3
            assert self.patch_size == 1, 'Patch size only works with 4D input.'
            assert not out_2d_map, 'out_2d_map is only supported for 4D input.'

        hidden_states = self.proj_in(hidden_states)

        # 2. Pos embedding
        if self.pos_embeddings is not None:
            hidden_states = self.pos_embeddings(hidden_states)

        # 3. Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states, attention_mask=attention_mask, ctx_emb=ctx_emb
            )

        # 4. Output
        if self.norm_out is not None:
            hidden_states = self.norm_out(hidden_states)
        out_states = self.proj_out(hidden_states)
        if out_2d_map:
            out_states = rearrange(out_states, 'b (h w) c -> b c h w', h=H, w=W)

        if return_dict:
            return {'hidden_states': hidden_states, 'out_states': out_states}
        return out_states
