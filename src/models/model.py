# MIT License

# Copyright (c) 2022 Machine Vision and Learning Group, LMU Munich

# Based on code: https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/encoders/modules.py

import math
import pdb
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from .dynamic_conv import DynamicConv, DynamicConv_decoder
from typing import List, Tuple


from .modules.layers import (
    Normalize,
    nonlinearity,
    ResnetBlock,
    Upsample,
    Downsample,
    make_attn,
)


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult: Tuple[int] = (1, 2, 4, 8),
        num_res_blocks: int,
        attn_resolutions: List[int],
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int,
        resolution: int,
        z_channels: int,
        double_z: bool = True,
        **ignore_kwargs,
    ):
        """
        Encoder module that downsamples input images to latent representations.

        Args:
            ch: Base channel count
            out_ch: Output channels
            ch_mult: Channel multiplier for each level
            num_res_blocks: Number of ResNet blocks per level
            attn_resolutions: Resolutions at which to apply attention
            dropout: Dropout rate
            resamp_with_conv: Whether to use convolution for resampling
            in_channels: Number of input channels
            resolution: Input resolution
            z_channels: Number of channels in latent space
            double_z: Whether to double the latent channels
        """
        super().__init__()

        attn_type = 'vanilla'
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.z_channels = z_channels

        # downsampling
        # self.conv_in = torch.nn.Conv2d(
        #    in_channels, self.ch, kernel_size=3, stride=1, padding=1
        # )
        #'''
        self.conv_in = DynamicConv(
            wv_planes=128,
            inter_dim=128,
            kernel_size=3,
            stride=1,
            padding=1,
            embed_dim=self.ch,
        )

        # TODO: if training
        wg_weights = torch.load(
            '/home/xshadow/Datasets/eo-vae/src/models/encoder_dconv_weight_generator_init_0.01_er50k.pt'
        )
        self.conv_in.weight_generator.load_state_dict(wg_weights['weight_generator'])
        self.conv_in.fclayer.load_state_dict(wg_weights['fclayer'])
        #'''

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x, wvs):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x, wvs)]
        # hs = [self.conv_in(x)]

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult: Tuple[int] = (1, 2, 4, 8),
        num_res_blocks: int,
        attn_resolutions: List[int],
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int,
        resolution: int,
        z_channels: int,
        give_pre_end: bool = False,
        tanh_out: bool = False,
        **ignorekwargs,
    ):
        """
        Decoder module that upsamples latent representations to images.

        Args:
            ch: Base channel count
            out_ch: Output channels
            ch_mult: Channel multiplier for each level
            num_res_blocks: Number of ResNet blocks per level
            attn_resolutions: Resolutions at which to apply attention
            dropout: Dropout rate
            resamp_with_conv: Whether to use convolution for resampling
            in_channels: Number of input channels
            resolution: Output resolution
            z_channels: Number of channels in latent space
            give_pre_end: Whether to return features before final convolution
            tanh_out: Whether to apply tanh to output
        """
        super().__init__()

        attn_type = 'vanilla'

        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print(
            'Working with z of shape {} = {} dimensions.'.format(
                self.z_shape, np.prod(self.z_shape)
            )
        )

        # z to block_in
        self.conv_in = torch.nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        # self.conv_out = torch.nn.Conv2d(
        #    block_in, out_ch, kernel_size=3, stride=1, padding=1
        # )
        self.conv_out = DynamicConv_decoder(
            wv_planes=128,
            inter_dim=128,
            kernel_size=3,
            stride=1,
            padding=1,
            embed_dim=block_in,
        )
        wg_weights = torch.load(
            '/home/xshadow/Datasets/eo-vae/src/models/decoder_dconv_weight_generator_init_0.01_er50k.pt'
        )
        self.conv_out.weight_generator.load_state_dict(wg_weights['weight_generator'])
        self.conv_out.fclayer.load_state_dict(wg_weights['fclayer'])

    def forward(self, z, wvs):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h, wvs)
        # h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h
