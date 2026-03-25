"""Multi-spectral patch embedding for EO-UNITE.

Two implementations with identical interface:

- PanopticonPatchEmbed: Conv3D channel-wise patchification + cross-attention
  channel fusion (ported from Panopticon, Bai et al. 2025).
- DynamicHypernetworkPatchEmbed: Wavelength-conditioned hypernetwork generates
  convolutional patch embedding weights (adapted from EO-VAE DynamicConv).

Both output [B, num_patches, hidden_size] tokens ready for the ViT encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from eo_vae.models.modules.dynamic_conv import (
    TransformerWeightGenerator,
    get_1d_sincos_pos_embed_from_grid_torch,
)


# ---------------------------------------------------------------------------
# Panopticon-style patch embedding
# ---------------------------------------------------------------------------


class PanopticonPatchEmbed(nn.Module):
    """Multi-spectral patch embedding via Conv3D + cross-attention channel fusion.

    Handles variable numbers of input channels by:
    1. Independently patchifying each spectral band with a shared 3D conv
    2. Adding wavelength-based sinusoidal embeddings to each channel's tokens
    3. Cross-attending with a single learnable query to aggregate all channels
       into a fixed-size token per spatial patch

    Output is independent of input channel count C.

    Args:
        attn_dim: Internal attention dimension (after Conv3D, before final proj).
        embed_dim: Output embedding dimension (hidden_size of ViT encoder).
        patch_size: Spatial patch size P. Each patch covers P×P pixels.
        chnfus_cfg: Config dict passed to ChnAttn (num_heads, layer_norm, etc.).
    """

    def __init__(
        self,
        attn_dim: int,
        embed_dim: int,
        patch_size: int,
        chnfus_cfg: dict | None = None,
    ):
        super().__init__()
        self.conv3d = _Conv3dWrapper(patch_size=patch_size, embed_dim=attn_dim)
        self.chnfus = _ChnAttn(**(chnfus_cfg or {}), dim=attn_dim)
        self.proj = nn.Linear(attn_dim, embed_dim)

    def forward(self, x: Tensor, wvs: Tensor) -> Tensor:
        """Embed multi-spectral image into patch tokens.

        Args:
            x: Image tensor [B, C, H, W].
            wvs: Band wavelengths in micrometers [C].

        Returns:
            Patch tokens [B, num_patches, embed_dim].
        """
        # chn_ids shape: [B, C] with wavelength values in nm for ChnEmb
        B = x.shape[0]
        chn_ids = (wvs * 1000).unsqueeze(0).expand(B, -1)  # µm → nm, [B, C]
        x = self.conv3d(x)              # [B, C, L, attn_dim]
        x = self.chnfus(x, chn_ids)    # [B, L, attn_dim]
        return self.proj(x)             # [B, L, embed_dim]


class _Conv3dWrapper(nn.Module):
    """Channel-wise patchification: 1×P×P Conv3D applied independently per band."""

    def __init__(self, patch_size: int, embed_dim: int):
        super().__init__()
        P = patch_size
        self.conv3d = nn.Conv3d(1, embed_dim, kernel_size=(1, P, P), stride=(1, P, P))

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, H, W]
        x = self.conv3d(x.unsqueeze(1)).squeeze(1)  # [B, embed_dim, C, Hp, Wp]
        return x.flatten(-2).permute(0, 2, 3, 1)   # [B, C, L, embed_dim]


class _ChnAttn(nn.Module):
    """Cross-attention over spectral channels to produce a fixed-size token.

    A single learnable query attends over all channel tokens (keys/values),
    collapsing variable C channels into a fixed output per patch location.

    Args:
        dim: Attention dimension.
        num_heads: Number of attention heads.
        layer_norm: If True, apply LayerNorm to the output.
    """

    def __init__(self, dim: int, num_heads: int = 8, layer_norm: bool = False):
        super().__init__()
        self.chnemb = _ChnEmb(embed_dim=dim)
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.xattn = _CrossAttn(dim=dim, num_heads=num_heads)
        self.layer_norm = nn.LayerNorm(dim) if layer_norm else None

    def forward(self, x: Tensor, chn_ids: Tensor) -> Tensor:
        """
        Args:
            x: Channel tokens [B, C, L, D].
            chn_ids: Wavelengths in nm [B, C].

        Returns:
            Aggregated tokens [B, L, D].
        """
        B, C, L, D = x.shape
        chn_embs = self.chnemb(chn_ids)        # [B, C, D]
        x = x + chn_embs.unsqueeze(2)          # add wavelength emb per patch

        x = x.permute(0, 2, 1, 3).flatten(0, 1)  # [B*L, C, D]
        q = self.query.expand(x.shape[0], -1, -1)  # [B*L, 1, D]
        x = self.xattn(q, x, x)               # [B*L, 1, D]
        x = x.reshape(B, L, D)

        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return x


class _ChnEmb(nn.Module):
    """Wavelength sinusoidal embeddings for optical bands; learned for SAR.

    SAR channels are identified by wavelengths < 0 (negative sentinel values)
    following the Panopticon convention. For standard EO-VAE wvs (all > 0,
    in µm), everything is treated as optical.

    Args:
        embed_dim: Embedding dimension.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # Learned SAR embeddings (VV/VH ascending/descending, 8 categories)
        dim1 = embed_dim // 3
        dim2 = embed_dim - 2 * dim1
        self.embed_transmit = nn.Parameter(torch.zeros(2, dim1))
        self.embed_receive = nn.Parameter(torch.zeros(2, dim1))
        self.embed_orbit = nn.Parameter(torch.zeros(2, dim2))

    def forward(self, chn_ids: Tensor) -> Tensor:
        """
        Args:
            chn_ids: Wavelengths in nm [B, C]. Optical bands > 0; SAR < 0.

        Returns:
            Channel embeddings [B, C, embed_dim].
        """
        mus = chn_ids  # [B, C]
        device = mus.device
        dtype = self.embed_transmit.dtype

        sar_mask = mus < 0
        opt_mask = ~sar_mask

        embs = torch.zeros(*mus.shape, self.embed_dim, device=device, dtype=dtype)

        # Optical: sinusoidal over wavelength value
        if opt_mask.any():
            opt_vals = mus[opt_mask].float()
            embs[opt_mask] = get_1d_sincos_pos_embed_from_grid_torch(
                self.embed_dim, opt_vals
            ).to(dtype)

        # SAR: learned embeddings indexed by category (negative int index)
        if sar_mask.any():
            transmit = torch.cat(
                [self.embed_transmit[0].repeat(2, 1), self.embed_transmit[1].repeat(2, 1)],
                dim=0,
            ).repeat(3, 1)
            receive = torch.cat(
                [
                    self.embed_receive[0].unsqueeze(0),
                    self.embed_receive[1].repeat(2, 1),
                    self.embed_receive[0].unsqueeze(0),
                ],
                dim=0,
            ).repeat(3, 1)
            orbit = torch.stack(
                [
                    self.embed_orbit.mean(dim=0),
                    self.embed_orbit[0],
                    self.embed_orbit[1],
                ]
            ).repeat_interleave(4, dim=0)
            sar_embs = torch.cat([transmit, receive, orbit], dim=1)  # [12, embed_dim]
            idx = (-(mus[sar_mask] + 1)).long().clamp(0, sar_embs.shape[0] - 1)
            embs[sar_mask] = sar_embs[idx]

        return embs


class _CrossAttn(nn.Module):
    """Minimal multi-head cross-attention (no query projection)."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        assert dim % num_heads == 0
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        B, Nq, D = q.shape
        Nkv = k.shape[1]
        H, Hd = self.num_heads, self.head_dim

        q = q.reshape(B, Nq, H, Hd).permute(0, 2, 1, 3) * self.scale
        k = self.k_proj(k).reshape(B, Nkv, H, Hd).permute(0, 2, 1, 3)
        v = self.v_proj(v).reshape(B, Nkv, H, Hd).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        return (attn @ v).transpose(1, 2).reshape(B, Nq, D)


# ---------------------------------------------------------------------------
# Dynamic Hypernetwork patch embedding
# ---------------------------------------------------------------------------


class DynamicHypernetworkPatchEmbed(nn.Module):
    """Patch embedding via wavelength-conditioned hypernetwork.

    A transformer-based hypernetwork generates the weights of a patch
    convolution conditioned on the input wavelengths. Analogous to
    DynamicConv in the existing EO-VAE, but produces patch tokens
    [B, num_patches, embed_dim] instead of feature maps.

    Args:
        embed_dim: Output embedding dimension (hidden_size of ViT).
        patch_size: Spatial patch size P.
        wv_planes: Sinusoidal embedding dimension for wavelengths.
        num_layers: Transformer layers in the hypernetwork.
        inter_dim: Intermediate dimension in the hypernetwork.
    """

    def __init__(
        self,
        embed_dim: int,
        patch_size: int,
        wv_planes: int = 256,
        num_layers: int = 4,
        inter_dim: int = 256,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.wv_planes = wv_planes

        # Hypernetwork: wavelengths → conv patch embedding weights
        # Output weight shape: [embed_dim, C, P, P] — but we generate it as
        # [C, embed_dim * P * P] and reshape (C is variable, handled per call)
        # We generate one weight vector per input channel (len(wvs) vectors),
        # each of shape [embed_dim * P * P]
        self.weight_generator = TransformerWeightGenerator(
            input_dim=wv_planes,
            output_dim=embed_dim * patch_size * patch_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
        )
        self.scaler = 0.1
        self.fclayer = nn.Sequential(
            nn.Linear(wv_planes, wv_planes),
            nn.GELU(),
        )

    def forward(self, x: Tensor, wvs: Tensor) -> Tensor:
        """Patchify using hypernetwork-generated weights.

        Args:
            x: Image [B, C, H, W].
            wvs: Wavelengths in µm [C].

        Returns:
            Patch tokens [B, num_patches, embed_dim].
        """
        P = self.patch_size
        B, C, H, W = x.shape

        # Build wavelength embeddings
        wv_emb = get_1d_sincos_pos_embed_from_grid_torch(
            self.wv_planes, wvs * 1000
        )  # [C, wv_planes]
        wv_emb = self.fclayer(wv_emb)  # [C, wv_planes]

        # Generate patch conv weights + bias from wavelength-conditioned transformer.
        weights, bias = self.weight_generator(wv_emb)  # [C, embed_dim * P * P], [embed_dim]
        # Reshape to [embed_dim, C, P, P]
        weight = weights.view(C, self.embed_dim, P, P).permute(1, 0, 2, 3)
        bias = bias.view(self.embed_dim) * self.scaler if bias is not None else None

        # Apply as standard patch conv (stride=P, no padding)
        # x: [B, C, H, W] → [B, embed_dim, H/P, W/P]
        tokens = F.conv2d(x, weight * self.scaler, bias=bias, stride=P)
        # Flatten spatial dims: [B, embed_dim, Hp, Wp] → [B, Hp*Wp, embed_dim]
        return tokens.flatten(2).transpose(1, 2)


def build_patch_embed(patch_embed_type: str, cfg: dict) -> nn.Module:
    """Factory for patch embedding modules.

    Args:
        patch_embed_type: 'panopticon' or 'dynamic'.
        cfg: Config dict forwarded to the constructor.

    Returns:
        Patch embedding module.
    """
    if patch_embed_type == 'panopticon':
        return PanopticonPatchEmbed(**cfg)
    elif patch_embed_type == 'dynamic':
        return DynamicHypernetworkPatchEmbed(**cfg)
    else:
        raise ValueError(f'Unknown patch_embed_type: {patch_embed_type!r}')
