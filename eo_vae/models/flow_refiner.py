import torch
import torch.nn as nn
from azula.denoise import DiracPosterior
from torch import Tensor


class FlowRefinementDenoiser(nn.Module):
    """Azula-compliant wrapper for JiT Flow Matching.
    Logic: x_t = t * x + (1 - t) * z
    Where x = Ground Truth, z = VAE Reconstruction.
    """

    def __init__(self, backbone: nn.Module, schedule: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.schedule = schedule

    def forward(self, x_t: Tensor, t: Tensor, **kwargs) -> 'DiracPosterior':
        """Forward pass

        Args:
            x_t: current tensor
            t: current time step

        Returns:
            less noisy tensor
        """
        # Backbone predicts the clean target 'x' directly (JiT style)
        # No preconditioning (c_in, c_out) as requested.
        # wvs (wavelengths) are passed via kwargs.
        # if t has no dim, its for sampling and we need to cast to batch dimension
        if t.ndim == 0:
            t = t.view(1).expand(x_t.shape[0])
        x_pred = self.backbone(x_t, t, **kwargs)
        return DiracPosterior(mean=x_pred)

    def loss(self, x: Tensor, z: Tensor, t: Tensor, **kwargs) -> Tensor:
        """Standard Flow Matching / JiT objective using Azula notation.
        x: target (x_target), z: source (x_recon)
        """
        alpha_t, sigma_t = self.schedule(t)

        # Match dimensions for broadcasting
        while alpha_t.ndim < x.ndim:
            alpha_t, sigma_t = alpha_t[..., None], sigma_t[..., None]

        # 1. Create linear interpolation: x_t = t*x + (1-t)*z
        x_t = alpha_t * x + sigma_t * z

        # 2. Predict clean x
        # Note: Azula's forward returns a Posterior object; we take the mean.
        x_pred = self.forward(x_t, t, **kwargs).mean

        # 3. Simple MSE Loss: (pred_target - target)^2
        # This is equivalent to velocity matching on a rectified flow path.
        return torch.nn.functional.mse_loss(x_pred, x)
