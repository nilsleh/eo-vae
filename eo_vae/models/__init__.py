from .autoencoder import AutoencoderKL
from .autoencoder_flux import FluxAutoencoderKL
from .model import Decoder, Encoder
from .new_autoencoder import EOFluxVAE

try:
    from .diffusers_vae import EOVAEDiffusersModel
except ImportError:
    EOVAEDiffusersModel = None

try:
    from .ssdd import EOSSDD
except ImportError:
    EOSSDD = None

__all__ = (
    'AutoencoderKL',
    'FluxAutoencoderKL',
    'Encoder',
    'Decoder',
    'EOFluxVAE',
    'EOVAEDiffusersModel',
    'EOSSDD',
)
