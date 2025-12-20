# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections.abc import Mapping

import torch
import torch.nn as nn

from ..modules.ssdd_dyn_conv import SSDD_InputConv, SSDD_OutputConv
from .embeddings import LearnedPositionalEmbedding, TimestepEmbedding, Timesteps
from .uvit_blocks import TransformerBlock, VisionTransformer
from .uvit_conv import DownBlock2D, ResnetBlock2D, UpBlock2D
from .uvit_torch_utils import ACTIVATIONS
from .uvit_train_utils import init_weights, init_zero


class UViTDecoder(nn.Module):
    SIZES = {
        'XS': {
            'channels': 32,
            'num_attention_heads': 4,
            'mid_nlayers': 8,
            'ch_mult': (1, 2, 3, 3),
        },
        'S': {
            'channels': 48,
            'num_attention_heads': 4,
            'mid_nlayers': 8,
            'ch_mult': (1, 2, 3, 3),
        },
        'B': {'channels': 64, 'mid_nlayers': 10, 'ch_mult': (1, 2, 3, 3)},
        'M': {'channels': 96, 'mid_nlayers': 12, 'ch_mult': (1, 2, 3, 3)},
        'L': {'channels': 96, 'mid_nlayers': 16, 'ch_mult': (1, 2, 4, 4)},
        'XL': {'channels': 128, 'mid_nlayers': 16, 'ch_mult': (1, 2, 4, 4)},
        'H': {'channels': 192, 'mid_nlayers': 16, 'ch_mult': (1, 2, 4, 4)},
        'XH': {'channels': 256, 'mid_nlayers': 16, 'ch_mult': (1, 2, 4, 4)},
        'G': {'channels': 384, 'mid_nlayers': 16, 'ch_mult': (1, 2, 4, 4)},
    }

    # Based on code from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_2d_condition.py
    def __init__(
        self,
        in_channels=3,
        z_dim=4,
        channels=128,
        ch_mult=(1, 2, 4, 4),
        act_fn: str = 'silu',
        vit_act_fn: str = 'geglu',
        layers_per_block=2,
        num_attention_heads: int | None = None,
        dropout=0.0,
        norm_num_groups=32,
        time_scale_shift=True,
        mid_nlayers=12,
        mid_theta=100.0,
        attn_window=8,
        eps=1e-5,
        ada_norm=True,
        learned_pos_embed=False,
        image_size=None,
        relative_pos_embed=True,
        init: Mapping | None = None,
    ):
        ### Config ###
        self.out_dim = in_channels
        self.ada_norm = ada_norm

        # Compute appropriate number of channels for each level, adjust for GroupNorm
        self.ch_level = [
            math.ceil(channels * ch_f / norm_num_groups) * norm_num_groups
            for ch_f in ch_mult
        ]
        channels = self.ch_level[0]  # The first channel is the input channel
        self.mid_dim = self.ch_level[-1]

        if isinstance(dropout, (int, float)):
            dropout = [dropout] * len(self.ch_level)

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(self.ch_level)

        super().__init__()

        ### Time ###
        time_embed_dim = channels * 4
        self.time_proj = Timesteps(
            channels, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedding = TimestepEmbedding(channels, time_embed_dim, act_fn=act_fn)

        self.wvs_embedding = TimestepEmbedding(channels, time_embed_dim, act_fn=act_fn)

        # add time embed to dynamic input layer
        self.conv_in = SSDD_InputConv(
            latent_channels=z_dim,
            embed_dim=channels,
            time_embed_dim=time_embed_dim,
            kernel_size=3,
        )

        ### AdaNorm Embedding ###
        if ada_norm:
            self.ada_ctx_proj = torch.nn.Sequential(
                torch.nn.Conv2d(z_dim, channels, kernel_size=3, stride=1, padding=1),
                torch.nn.SiLU(),
                torch.nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            )

        ### Down blocks ###
        self.down_blocks = nn.ModuleList([])
        output_channel = channels
        for i_level, ch in enumerate(self.ch_level):
            input_channel = output_channel
            output_channel = ch
            is_final_block = i_level == len(self.ch_level) - 1

            if layers_per_block[i_level] != 0:
                self.down_blocks.append(
                    DownBlock2D(
                        num_layers=layers_per_block[i_level],
                        in_channels=input_channel,
                        out_channels=output_channel,
                        temb_channels=time_embed_dim,
                        dropout=dropout[i_level],
                        add_downsample=not is_final_block,
                        resnet_act_fn=act_fn,
                        resnet_groups=norm_num_groups,
                        time_scale_shift=time_scale_shift,
                        resnet_eps=eps,
                        ada_norm=ada_norm,
                        ada_emb_dim=channels,
                    )
                )

        # Mid block ###
        down_scale = (2 ** (len(self.ch_level) - 1),)
        self.mid_block = UViTMiddleTransformer(
            inner_dim=output_channel,
            dropout=dropout[-1],
            num_layers=mid_nlayers,
            norm_num_groups=norm_num_groups,
            num_attention_heads=num_attention_heads,
            rope_theta=mid_theta,
            attn_window=attn_window,
            eps=eps,
            ada_norm=ada_norm,
            ada_emb_dim=channels,
            learned_pos_embed=learned_pos_embed,
            sample_size=(image_size[0] // down_scale[0], image_size[1] // down_scale[0])
            if learned_pos_embed
            else None,
            relative_pos_embed=relative_pos_embed,
            act_fn=vit_act_fn,
        )

        ### Up blocks ###
        self.up_blocks = nn.ModuleList([])

        for i_level, ch in enumerate(reversed(self.ch_level)):
            input_channel = (
                self.ch_level[-i_level - 2]
                if i_level < len(self.ch_level) - 1
                else self.ch_level[0]
            )
            prev_output_channel = output_channel
            output_channel = ch

            is_final_block = i_level == len(self.ch_level) - 1

            if layers_per_block[-i_level - 1] != 0:
                self.up_blocks.append(
                    UpBlock2D(
                        num_layers=layers_per_block[-i_level - 1] + 1,
                        in_channels=input_channel,
                        out_channels=output_channel,
                        prev_output_channel=prev_output_channel,
                        temb_channels=time_embed_dim,
                        resolution_idx=i_level,
                        dropout=dropout[-i_level - 1],
                        add_upsample=(not is_final_block),
                        resnet_act_fn=act_fn,
                        resnet_groups=norm_num_groups,
                        time_scale_shift=time_scale_shift,
                        resnet_eps=eps,
                        ada_norm=ada_norm,
                        ada_emb_dim=channels,
                    )
                )

        ### Output ###
        self.conv_norm_out = nn.GroupNorm(
            num_channels=channels, num_groups=norm_num_groups, eps=eps
        )
        self.conv_out_act = ACTIVATIONS[act_fn]()
        # self.conv_out = nn.Conv2d(channels, in_channels, kernel_size=3, padding=1)

        self.conv_out = SSDD_OutputConv(
            embed_dim=channels, time_embed_dim=time_embed_dim, kernel_size=3
        )

        ### Weights init ###
        self.init_weights(**(init or {}))

    @classmethod
    def make(cls, size=None, **kwargs):
        if size is not None:
            if size in cls.SIZES:
                kwargs = {**cls.SIZES[size], **kwargs}
            else:
                raise ValueError(
                    f"Unknown size '{size}' for UViTDecoder. Available sizes: {list(cls.SIZES.keys())}"
                )
        return cls(**kwargs)

    def init_weights(self, method='xavier_uniform', ckpt_module='decoder', **kwargs):
        for m in self.modules():
            if isinstance(m, ResnetBlock2D):
                init_zero(m.conv2)
            elif isinstance(m, TransformerBlock):
                init_zero(m.ff.proj_out)

        init_weights(self, method=method, ckpt_module=ckpt_module, **kwargs)

    def get_time_embed(
        self, sample: torch.Tensor, timestep: torch.Tensor | float
    ) -> torch.Tensor:
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                [timesteps], dtype=torch.float64, device=sample.device
            )
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)
        return t_emb

    def forward(self, x, t, z, wvs):
        """x:   (B, N, H, W)  -> Full multispectral cube (Sentinel-2 N=13, or S1 N=2, etc)
        t:   (B,)          -> Diffusion time
        z:   (B, Cz, h, w) -> Shared spectral blueprint
        wvs: (N)        -> Wavelengths for each band
        """
        # 1. Time embedding
        # Computed early because it is needed for the dynamic input convolution
        t_emb = self.get_time_embed(sample=x, timestep=t)
        t_emb = self.time_embedding(t_emb)

        z_expanded = torch.nn.functional.interpolate(
            z, size=(x.shape[-2], x.shape[-1]), mode='nearest'
        )

        # 2. Context embedding (z)
        ctx_emb = None
        if self.ada_norm:
            ctx_emb = self.ada_ctx_proj(z)

        # 3. Input Projection (Dynamic)
        # Note: We skip concatenating z to x here because SSDD_InputConv handles spectral bands x
        # and expects wvs to match x channels. z is injected via AdaNorm (ctx_emb).
        x = self.conv_in(x, wvs, t_emb, z_expanded)

        # 4. Down blocks
        down_block_res = [x]
        for downsample_block in self.down_blocks:
            x, res_samples = downsample_block(
                hidden_states=x, temb=t_emb, ctx_emb=ctx_emb
            )
            down_block_res.extend(res_samples)

        # 5. Mid block
        x = self.mid_block(x, ctx_emb=ctx_emb)

        # 6. Up blocks
        for upsample_block in self.up_blocks:
            res_samples = down_block_res[-len(upsample_block.resnets) :]
            down_block_res = down_block_res[: -len(upsample_block.resnets)]
            x = upsample_block(
                hidden_states=x,
                temb=t_emb,
                res_hidden_states_tuple=res_samples,
                ctx_emb=ctx_emb,
            )

        if self.conv_norm_out:
            x = self.conv_norm_out(x)
        x = self.conv_out_act(x)
        x = self.conv_out(x, wvs, t_emb)
        return x


class UViTMiddleTransformer(VisionTransformer):
    def __init__(
        self,
        *args,
        sample_size=None,
        act_fn: str = 'geglu',
        learned_pos_embed=True,
        norm_num_groups: int = 32,
        out_norm: bool = False,
        **kwargs,
    ):
        super().__init__(
            *args,
            learned_pos_embed=False,
            sample_size=sample_size,
            act_fn=act_fn,
            out_norm=out_norm,
            **kwargs,
        )

        self.norm = torch.nn.GroupNorm(
            num_groups=norm_num_groups,
            num_channels=self.in_channels,
            eps=1e-6,
            affine=True,
        )

        self.pre_pos_embeddings = None
        if learned_pos_embed:
            self.pre_pos_embeddings = LearnedPositionalEmbedding(
                (self.inner_dim, *sample_size)
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        ctx_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        if self.pre_pos_embeddings is not None:
            hidden_states = self.pre_pos_embeddings(hidden_states)

        hidden_states = self.norm(hidden_states)

        hidden_states = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            ctx_emb=ctx_emb,
            out_2d_map=True,
        )
        output = hidden_states + residual
        return output
