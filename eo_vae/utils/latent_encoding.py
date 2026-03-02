"""Shared utilities for VAE latent encoding and decoding.

Reused by encode_latents.py (Sen2NAIP) and encode_latents_sen12ms.py (Sen12MS-CR).
"""

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from eo_vae.models.autoencoder_flux import FluxAutoencoderKL


# =============================================================================
# RUNNING STATISTICS
# =============================================================================


class RunningStatsButFast(torch.nn.Module):
    """Online per-channel mean/std tracker using the parallel algorithm.

    Reference:
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

    Example:
        rs = RunningStatsButFast((32,), [0, 2, 3])  # For latents [B, 32, H, W]
        for batch in dataloader:
            z = encode(batch)
            rs(z)
        print(rs.mean, rs.std)
    """

    def __init__(self, shape, dims):
        """Args:
        shape: Shape of resulting mean/variance (e.g., (32,) for 32 channels).
        dims: Dimensions to reduce over (e.g., [0, 2, 3] for batch, height, width).
        """
        super().__init__()
        self.register_buffer('mean', torch.zeros(shape))
        self.register_buffer('var', torch.ones(shape))
        self.register_buffer('std', torch.ones(shape))
        self.register_buffer('count', torch.zeros(1))
        self.register_buffer('min', torch.full(shape, float('inf')))
        self.register_buffer('max', torch.full(shape, float('-inf')))
        self.dims = dims

    def update(self, x):
        with torch.no_grad():
            batch_mean = torch.mean(x, dim=self.dims)
            batch_var = torch.var(x, dim=self.dims)
            batch_min = torch.amin(x, dim=self.dims)
            batch_max = torch.amax(x, dim=self.dims)

            batch_count = 1.0
            for d in self.dims:
                batch_count *= x.shape[d]
            batch_count = torch.tensor(batch_count, dtype=torch.float, device=x.device)

            n_ab = self.count + batch_count
            m_a = self.mean * self.count
            m_b = batch_mean * batch_count
            M2_a = self.var * self.count
            M2_b = batch_var * batch_count

            delta = batch_mean - self.mean

            self.mean = (m_a + m_b) / n_ab
            self.var = (
                M2_a + M2_b + delta**2 * self.count * batch_count / (n_ab + 1e-8)
            ) / n_ab
            self.count = n_ab
            self.std = torch.sqrt(self.var + 1e-8)

            self.min = torch.minimum(self.min, batch_min)
            self.max = torch.maximum(self.max, batch_max)

    def forward(self, x):
        self.update(x)
        return x

    def get_stats_dict(self):
        """Return statistics as a dictionary for saving."""
        return {
            'mean': self.mean.cpu(),
            'std': self.std.cpu(),
            'var': self.var.cpu(),
            'min': self.min.cpu(),
            'max': self.max.cpu(),
            'count': self.count.cpu(),
        }


# =============================================================================
# MODEL LOADING
# =============================================================================


def load_eo_vae(config_path, ckpt_path, device):
    """Load EO-VAE from config and checkpoint."""
    print(f'Loading EO-VAE from config: {config_path}')
    conf = OmegaConf.load(config_path)
    model = instantiate(conf.model)

    if ckpt_path:
        print(f'Loading EO-VAE checkpoint from {ckpt_path}')
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=False)

    model.to(device).eval()
    return model


# =============================================================================
# ENCODING FUNCTIONS
# =============================================================================


@torch.no_grad()
def encode_raw(model, img, wvs):
    """Encode image to RAW latent space (no shuffle, no BatchNorm).

    Args:
        model: EO-VAE model.
        img: Input image [B, C, H, W] (already normalized by dataloader).
        wvs: Wavelength vector.

    Returns:
        Raw latent [B, 32, H/8, W/8].
    """
    if isinstance(model, FluxAutoencoderKL) or hasattr(model, 'encoder'):
        moments = model.encoder(img, wvs)
        mean, _ = torch.chunk(moments, 2, dim=1)
        return mean

    raise ValueError(f'Unknown model type: {type(model)}')


@torch.no_grad()
def encode_spatial_norm(model, img, wvs):
    """Encode image to spatially normalised latent space (VAE internal BN)."""
    if hasattr(model, 'encode_spatial_normalized'):
        return model.encode_spatial_normalized(img, wvs)
    raise ValueError('Model does not support encode_spatial_normalized method')


@torch.no_grad()
def decode_raw(model, z, wvs):
    """Decode RAW latent to image (no unshuffle, no inverse BatchNorm).

    Args:
        z: Raw latent [B, 32, H/8, W/8].
        wvs: Wavelength vector.

    Returns:
        Reconstructed image [B, C, H, W].
    """
    if isinstance(model, FluxAutoencoderKL) or hasattr(model, 'decoder'):
        return model.decoder(z, wvs)

    raise ValueError(f'Unknown model type: {type(model)}')


@torch.no_grad()
def decode_spatial_norm(model, z, wvs):
    """Decode spatially normalised latent to image (VAE internal inverse BN)."""
    if hasattr(model, 'decode_spatial_normalized'):
        return model.decode_spatial_normalized(z, wvs)
    raise ValueError('Model does not support decode_spatial_normalized method')
