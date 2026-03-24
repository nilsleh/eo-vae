"""Modality conditioning for EO-UNITE generation arm.

Replaces UNITE's class-label embedding with a richer conditioning signal
combining spectral (wavelength), spatial (geolocation), and temporal (time)
information. Used as the conditioning input for the flow-matching denoiser.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from eo_vae.models.modules.dynamic_conv import get_1d_sincos_pos_embed_from_grid_torch


class ModalityConditioner(nn.Module):
    """Encode wavelengths, geolocation and time into a single conditioning vector.

    Produces a per-sample conditioning vector [B, cond_dim] for use as the
    generation conditioning signal in EO-UNITE (analogous to class-label
    embeddings in the original UNITE).

    All three inputs are optional — missing inputs fall back to zero embeddings
    so the module works with any dataset regardless of available metadata.

    Args:
        cond_dim: Output conditioning dimension.
        wv_planes: Sinusoidal embedding dimension for wavelengths.
        geo_planes: Sinusoidal embedding dimension per geo coordinate (lat/lon).
        time_planes: Sinusoidal embedding dimension for day-of-year.
    """

    def __init__(
        self,
        cond_dim: int = 256,
        wv_planes: int = 128,
        geo_planes: int = 64,
        time_planes: int = 64,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.wv_planes = wv_planes
        self.geo_planes = geo_planes
        self.time_planes = time_planes

        # Wavelength branch: mean(wvs) → sincos → MLP → [wv_planes]
        self.wv_mlp = nn.Sequential(
            nn.Linear(wv_planes, wv_planes),
            nn.SiLU(),
            nn.Linear(wv_planes, wv_planes),
        )

        # Geo branch: (lat sincos, lon sincos) → MLP → [geo_planes]
        # lat and lon each produce geo_planes/2 sincos dims, concatenated
        geo_in = geo_planes  # geo_planes/2 for lat + geo_planes/2 for lon
        self.geo_mlp = nn.Sequential(
            nn.Linear(geo_in, geo_planes),
            nn.SiLU(),
            nn.Linear(geo_planes, geo_planes),
        )

        # Time branch: day-of-year (0-365) → sincos → MLP → [time_planes]
        self.time_mlp = nn.Sequential(
            nn.Linear(time_planes, time_planes),
            nn.SiLU(),
            nn.Linear(time_planes, time_planes),
        )

        # Final projection: concat all branches → cond_dim
        total_in = wv_planes + geo_planes + time_planes
        self.proj = nn.Linear(total_in, cond_dim)

    def forward(
        self,
        wvs: Tensor,
        geo: Tensor | None = None,
        time: Tensor | None = None,
    ) -> Tensor:
        """Produce conditioning vector.

        Args:
            wvs: Band wavelengths in micrometers [C]. Batch-constant.
            geo: (lat, lon) in degrees [B, 2], or None.
            time: Day-of-year [B], or None.

        Returns:
            Conditioning vector [B, cond_dim].
        """
        # Infer batch size from geo/time; fallback to 1 if both absent
        if geo is not None:
            B = geo.shape[0]
            device = geo.device
            dtype = geo.dtype
        elif time is not None:
            B = time.shape[0]
            device = time.device
            dtype = time.dtype
        else:
            B = 1
            device = wvs.device
            dtype = wvs.dtype

        # --- Wavelength branch ---
        # Use mean wavelength as single representative value per batch [1]
        mean_wv = wvs.mean().unsqueeze(0)  # [1]
        wv_emb = get_1d_sincos_pos_embed_from_grid_torch(
            self.wv_planes, mean_wv * 1000  # convert µm → nm for better scale
        )  # [1, wv_planes]
        wv_emb = self.wv_mlp(wv_emb.to(dtype))  # [1, wv_planes]
        wv_emb = wv_emb.expand(B, -1)  # [B, wv_planes]

        # --- Geolocation branch ---
        if geo is not None:
            lat = geo[:, 0]  # [B]
            lon = geo[:, 1]  # [B]
            lat_emb = _sincos_1d(lat, self.geo_planes // 2)  # [B, geo_planes/2]
            lon_emb = _sincos_1d(lon, self.geo_planes // 2)  # [B, geo_planes/2]
            geo_emb = self.geo_mlp(torch.cat([lat_emb, lon_emb], dim=-1))  # [B, geo_planes]
        else:
            geo_emb = torch.zeros(B, self.geo_planes, device=device, dtype=dtype)

        # --- Time branch ---
        if time is not None:
            # Normalize day-of-year to [0, 2π] for sincos
            time_norm = time.float() / 365.0 * 2 * math.pi
            time_emb_raw = _sincos_1d(time_norm, self.time_planes)  # [B, time_planes]
            time_emb = self.time_mlp(time_emb_raw.to(dtype))  # [B, time_planes]
        else:
            time_emb = torch.zeros(B, self.time_planes, device=device, dtype=dtype)

        # --- Concatenate and project ---
        combined = torch.cat([wv_emb, geo_emb, time_emb], dim=-1)  # [B, total_in]
        return self.proj(combined)  # [B, cond_dim]


def _sincos_1d(pos: Tensor, embed_dim: int) -> Tensor:
    """Sinusoidal embedding for scalar positions.

    Args:
        pos: Positions [B].
        embed_dim: Output dimension (must be even).

    Returns:
        Embeddings [B, embed_dim].
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2)))  # [embed_dim/2]
    out = pos.unsqueeze(1).float() * omega.unsqueeze(0)  # [B, embed_dim/2]
    return torch.cat([torch.sin(out), torch.cos(out)], dim=-1)  # [B, embed_dim]
