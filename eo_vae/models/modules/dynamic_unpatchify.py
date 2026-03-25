"""Wavelength-conditioned decoder output layer for EO-UNITE.

Replaces UNITE's fixed 3-channel unpatchify with dynamic variants that can
output any number of spectral bands conditioned on target wavelengths.

Two implementations matching the patch embedding variants:
- PanopticonUnpatchify: inverse cross-attention, symmetric with PanopticonPatchEmbed
- DynamicHypernetworkUnpatchify: wavelength-conditioned transposed conv
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from eo_vae.models.modules.dynamic_conv import (
    DynamicConv_decoder,
    get_1d_sincos_pos_embed_from_grid_torch,
)
from eo_vae.models.modules.eo_unite_patch_embed import _ChnEmb, _CrossAttn


class PanopticonUnpatchify(nn.Module):
    """Inverse of PanopticonPatchEmbed: decoder tokens → multi-spectral image.

    Each target wavelength gets a learned query (conditioned on its wavelength
    embedding) that cross-attends over the spatial patch tokens to produce
    per-band pixel values. Then patches are reassembled into a full image.

    Args:
        dec_dim: Decoder output token dimension (input to this module).
        attn_dim: Internal cross-attention dimension.
        patch_size: Spatial patch size P (must match encoder).
        num_heads: Number of attention heads.
    """

    def __init__(
        self,
        dec_dim: int,
        attn_dim: int,
        patch_size: int,
        num_heads: int = 8,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.attn_dim = attn_dim

        # Project decoder tokens to attn_dim
        self.in_proj = nn.Linear(dec_dim, attn_dim)

        # Channel embeddings: per-wavelength query conditioning
        self.chnemb = _ChnEmb(embed_dim=attn_dim)

        # Cross-attention: per-channel query × all spatial tokens
        self.xattn = _CrossAttn(dim=attn_dim, num_heads=num_heads)

        # Per-channel output: attn_dim → P² pixel values
        self.out_proj = nn.Linear(attn_dim, patch_size * patch_size)

    def forward(self, tokens: Tensor, wvs: Tensor) -> Tensor:
        """Decode patch tokens to a multi-spectral image.

        Args:
            tokens: Decoder output [B, L, dec_dim] where L = Hp*Wp.
            wvs: Target wavelengths in µm [C].

        Returns:
            Reconstructed image [B, C, H, W].
        """
        B, L, _ = tokens.shape
        P = self.patch_size
        Hp = Wp = int(math.sqrt(L))
        C = wvs.shape[0]

        # Project decoder tokens to attn_dim
        kv = self.in_proj(tokens)  # [B, L, attn_dim]

        # Build per-channel queries from wavelength embeddings
        chn_ids = (wvs * 1000).unsqueeze(0)  # [1, C] in nm
        chn_embs = self.chnemb(chn_ids)       # [1, C, attn_dim]
        queries = chn_embs.expand(B, -1, -1)  # [B, C, attn_dim]

        # Decode each spatial position independently so spatial structure is preserved.
        # Reshape kv: [B, L, D] → [B*L, 1, D]  (each patch as its own single-token context)
        # Expand queries: [B, C, D] → [B*L, C, D]
        kv_flat = kv.reshape(B * L, 1, self.attn_dim)
        queries_flat = queries.unsqueeze(2).expand(-1, -1, L, -1)   # [B, C, L, D]
        queries_flat = queries_flat.permute(0, 2, 1, 3).reshape(B * L, C, self.attn_dim)

        # Cross-attend: each channel query attends its own spatial token
        channel_feats = self.xattn(queries_flat, kv_flat, kv_flat)  # [B*L, C, attn_dim]

        # Decode to pixel patches and reassemble: [B*L, C, P²] → [B, C, H, W]
        patches = self.out_proj(channel_feats)           # [B*L, C, P²]
        patches = patches.reshape(B, Hp, Wp, C, P, P)
        out = patches.permute(0, 3, 1, 4, 2, 5).reshape(B, C, Hp * P, Wp * P)
        return out


class DynamicHypernetworkUnpatchify(nn.Module):
    """Inverse of DynamicHypernetworkPatchEmbed: decoder tokens → multi-spectral image.

    Reshapes decoder tokens to a spatial feature map and applies a wavelength-
    conditioned transposed convolution (via DynamicConv_decoder) to produce
    the output image at the original resolution.

    Args:
        dec_dim: Decoder output token dimension.
        patch_size: Spatial patch size P (must match encoder).
        wv_planes: Sinusoidal embedding dimension for wavelengths.
        num_layers: Hypernetwork transformer layers.
    """

    def __init__(
        self,
        dec_dim: int,
        patch_size: int,
        wv_planes: int = 256,
        num_layers: int = 4,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.dec_dim = dec_dim

        # Project decoder token dim → patch pixel space (dec_dim → P² features)
        # Then DynamicConv_decoder maps the feature channels to output bands
        self.pixel_proj = nn.Linear(dec_dim, dec_dim)

        # DynamicConv_decoder: [B, dec_dim, Hp, Wp] → [B, C, Hp, Wp]
        # Then ConvTranspose2d upsample to full resolution
        self.dyn_conv = DynamicConv_decoder(
            wv_planes=wv_planes,
            embed_dim=dec_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            num_layers=num_layers,
        )
        # Transposed conv to go from [B, C, Hp, Wp] → [B, C, H, W]
        # This is a fixed per-channel op — share weights across channels using
        # group conv trick via a simple bicubic upsample for simplicity
        self.patch_size = patch_size

    def forward(self, tokens: Tensor, wvs: Tensor) -> Tensor:
        """Decode patch tokens to a multi-spectral image.

        Args:
            tokens: Decoder output [B, L, dec_dim].
            wvs: Target wavelengths in µm [C].

        Returns:
            Reconstructed image [B, C, H, W].
        """
        B, L, D = tokens.shape
        P = self.patch_size
        Hp = Wp = int(math.sqrt(L))

        # Reshape to spatial feature map: [B, dec_dim, Hp, Wp]
        feat = self.pixel_proj(tokens)
        feat = feat.transpose(1, 2).reshape(B, D, Hp, Wp)

        # Dynamic conv: [B, dec_dim, Hp, Wp] → [B, C, Hp, Wp]
        out = self.dyn_conv(feat, wvs)  # [B, C, Hp, Wp]

        # Upsample to original resolution
        if P > 1:
            out = F.interpolate(out, scale_factor=P, mode='bilinear', align_corners=False)

        return out


def build_unpatchify(patch_embed_type: str, cfg: dict) -> nn.Module:
    """Factory for unpatchify modules, symmetric with build_patch_embed.

    Args:
        patch_embed_type: 'panopticon' or 'dynamic'.
        cfg: Config dict forwarded to the constructor.

    Returns:
        Unpatchify module.
    """
    if patch_embed_type == 'panopticon':
        return PanopticonUnpatchify(**cfg)
    elif patch_embed_type == 'dynamic':
        return DynamicHypernetworkUnpatchify(**cfg)
    else:
        raise ValueError(f'Unknown patch_embed_type: {patch_embed_type!r}')
