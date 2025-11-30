# Apache-2.0 License
# Copyright (c) https://github.com/black-forest-labs/flux2
# Modified for Unified EO-VAE / Flux Support

import torch
import torch.nn as nn
from torch import Tensor

from .modules.dynamic_conv import DynamicConv, DynamicConv_decoder
from .modules.layers import AttnBlock, Downsample, ResnetBlock, Upsample


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
        use_dynamic_ops: bool = False,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.z_channels = z_channels
        self.use_dynamic_ops = use_dynamic_ops

        if self.use_dynamic_ops:
            self.conv_in = DynamicConv(
                wv_planes=128,
                inter_dim=128,
                kernel_size=3,
                stride=1,
                padding=1,
                embed_dim=self.ch,
            )
        else:
            self.conv_in = nn.Conv2d(
                in_channels, self.ch, kernel_size=3, stride=1, padding=1
            )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        self.norm_out = nn.GroupNorm(
            num_groups=32, num_channels=block_in, eps=1e-6, affine=True
        )
        self.conv_out = nn.Conv2d(
            block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1
        )
        self.quant_conv = torch.nn.Conv2d(2 * z_channels, 2 * z_channels, 1)

    def forward(self, x: Tensor, wvs: Tensor = None) -> Tensor:
        if self.use_dynamic_ops:
            assert wvs is not None, 'wvs must be provided for Dynamic Encoder'
            hs = [self.conv_in(x, wvs)]
        else:
            hs = [self.conv_in(x)]

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        h = self.quant_conv(h)
        return h

    def load_flux_weights(self, state_dict, strict=True):
        """Smart loader that handles both exact matches (Original Flux)
        and Backbone-only loading (EO-Distillation).
        """
        own_state = self.state_dict()

        # If we are in Dynamic mode, we EXPECT mismatches in conv_in
        ignore_layers = ['conv_in'] if self.use_dynamic_ops else []

        for name, param in state_dict.items():
            # If dynamic, skip loading the static conv_in weights
            if self.use_dynamic_ops and any(x in name for x in ignore_layers):
                print(
                    f'Distillation Mode: Skipping {name} (Static weight not needed for Dynamic Layer)'
                )
                continue

            if name not in own_state:
                if strict:
                    raise KeyError(f'Unexpected key {name} in state_dict')
                continue

            if isinstance(param, torch.nn.Parameter):
                param = param.data

            try:
                own_state[name].copy_(param)
            except RuntimeError as e:
                print(f'Error loading {name}: {e}')

        print(f'Weights loaded. Dynamic Mode: {self.use_dynamic_ops}')


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        resolution: int,
        z_channels: int,
        use_dynamic_ops: bool = False,
    ):
        super().__init__()
        self.post_quant_conv = torch.nn.Conv2d(z_channels, z_channels, 1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.z_channels = z_channels
        self.resolution = resolution
        self.use_dynamic_ops = use_dynamic_ops

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        self.conv_in = nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        self.norm_out = nn.GroupNorm(
            num_groups=32, num_channels=block_in, eps=1e-6, affine=True
        )

        if self.use_dynamic_ops:
            # EO Mode: Dynamic Output
            self.conv_out = DynamicConv_decoder(
                wv_planes=128,
                inter_dim=128,
                kernel_size=3,
                stride=1,
                padding=1,
                embed_dim=block_in,
            )
        else:
            # Flux Mode: Standard Static Output
            self.conv_out = nn.Conv2d(
                block_in, out_ch, kernel_size=3, stride=1, padding=1
            )

    def forward(self, z: Tensor, wvs: Tensor = None) -> Tensor:
        z = self.post_quant_conv(z)
        upscale_dtype = next(self.up.parameters()).dtype

        h = self.conv_in(z)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        h = h.to(upscale_dtype)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = swish(h)

        if self.use_dynamic_ops:
            assert wvs is not None, 'wvs must be provided for Dynamic Decoder'
            h = self.conv_out(h, wvs)
        else:
            h = self.conv_out(h)

        return h

    def load_flux_weights(self, state_dict, strict=True):
        own_state = self.state_dict()
        ignore_layers = ['conv_out'] if self.use_dynamic_ops else []

        for name, param in state_dict.items():
            if self.use_dynamic_ops and any(x in name for x in ignore_layers):
                print(
                    f'Distillation Mode: Skipping {name} (Static weight not needed for Dynamic Layer)'
                )
                continue

            if name not in own_state:
                if strict:
                    # Only raise error if we expected an exact match
                    pass
                continue

            if isinstance(param, torch.nn.Parameter):
                param = param.data

            try:
                own_state[name].copy_(param)
            except RuntimeError as e:
                print(f'Error loading {name}: {e}')

        print(f'Weights loaded. Dynamic Mode: {self.use_dynamic_ops}')
