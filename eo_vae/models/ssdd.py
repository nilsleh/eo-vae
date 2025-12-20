from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from azula.denoise import DiracPosterior
from lightning import LightningModule
from torch import Tensor

from .autoencoder_flux import get_cosine_schedule_with_warmup


class SSDDFlowModule(nn.Module):
    """Module that encapsulates the Flow Matching logic for SSDD.
    Flows from Noise (x_0) to Data (x_1) conditioned on Latent (z).

    This module is designed to be compatible with Azula samplers.
    """

    def __init__(self, backbone: nn.Module, schedule: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.schedule = schedule

    def forward(self, x_t: Tensor, t: Tensor, **kwargs) -> DiracPosterior:
        """Forward pass for sampling.
        Wraps the backbone prediction in a DiracPosterior.
        """
        # Ensure t has batch dimension for backbone if it's a scalar
        if t.ndim == 0:
            t = t.view(1).expand(x_t.shape[0])

        # Backbone signature: (x_t, t, z, wvs)
        # We assume kwargs contains 'z' and 'wvs'
        pred = self.backbone(x_t, t, **kwargs)
        return DiracPosterior(mean=pred)

    def loss(self, x: Tensor, t: Tensor, **kwargs) -> Tensor:
        """Computes the Flow Matching loss.
        x: Ground Truth Data (x_1)
        t: Time steps
        kwargs: Conditioning (z, wvs)
        """
        # 1. Sample noise (x_0)
        noise = torch.randn_like(x)

        # 2. Get schedule parameters
        alpha_t, sigma_t = self.schedule(t)

        # Match dimensions for broadcasting
        while alpha_t.ndim < x.ndim:
            alpha_t, sigma_t = alpha_t[..., None], sigma_t[..., None]

        # 3. Interpolate x_t = alpha * x + sigma * noise
        # For Rectified Flow: alpha=t, sigma=1-t (usually)
        x_t = alpha_t * x + sigma_t * noise

        # 4. Predict
        # The backbone predicts velocity v = x - noise
        pred = self.forward(x_t, t, **kwargs).mean

        # 5. Target Velocity
        v_target = x - noise

        # 6. MSE Loss
        return F.mse_loss(pred, v_target)


class TimeSamplerLogitNormal:
    def __init__(self, t_mean=0, t_std=1.0):
        self.t_std = t_std
        self.t_mean = t_mean

    def __call__(self, batch_size, device):
        t = torch.randn(batch_size, device=device) * self.t_std + self.t_mean
        return torch.sigmoid(t)


class EOSSDD(LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        denoiser: nn.Module,
        sampler: Any,  # Azula sampler
        # Optimizer & Scheduler Args
        lr: float = 1e-4,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        final_lr: float = 1e-6,
        weight_decay: float = 0.05,
        image_key: str = 'image',
    ):
        super().__init__()

        self.lr = lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.final_lr = final_lr
        self.weight_decay = weight_decay
        self.image_key = image_key

        # 1. Pretrained & Frozen Encoder
        self.encoder = encoder
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        # 2. SSDD Flow Module (wraps decoder and schedule)
        self.denoiser = denoiser

        # 3. Sampler
        self.sampler = sampler

        self.t_sampler = TimeSamplerLogitNormal(t_mean=0, t_std=1.0)

    def forward(self, x_input, wvs):
        """Inference: Encode -> Sample -> Decode (Implicitly via flow)"""
        B = x_input.shape[0]

        # 1. Get Condition
        with torch.no_grad():
            z = self.encoder(x_input, wvs)

        # 2. Start from Noise
        x_t = torch.randn_like(x_input)

        # 3. Sample using the external sampler
        # The sampler calls self.denoiser(x_t, t, z=z, wvs=wvs)
        sampler = self.sampler(self.denoiser, steps=25)
        x_recon = sampler(x=x_t, z=z, wvs=wvs)

        return x_recon

    def training_step(self, batch, batch_idx):
        x_gt = batch[self.image_key]
        wvs = batch['wvs']
        B = x_gt.shape[0]

        with torch.no_grad():
            z = self.encoder(x_gt, wvs)

        # Sample time uniform [0, 1]
        t = self.t_sampler(B, device=self.device)

        # Compute Loss via Denoiser
        loss = self.denoiser.loss(x_gt, t, z=z, wvs=wvs)

        self.log('train/loss', loss, prog_bar=True)
        # log lr
        lr = self.optimizers().param_groups[0]['lr']
        self.log('train/lr', lr, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x_gt = batch[self.image_key]
        wvs = batch['wvs']

        with torch.no_grad():
            z = self.encoder(x_gt, wvs)

        x_noise = torch.randn_like(x_gt)
        # import pdb; pdb.set_trace()
        recon = self.sample_decode(x_noise, z, wvs, steps=25)

        loss = F.mse_loss(recon, x_gt)

        self.log('val/loss', loss, prog_bar=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def sample_decode(
        self, x_noise: torch.Tensor, z: torch.Tensor, wvs: torch.Tensor, steps: int = 25
    ) -> torch.Tensor:
        """Sample from the model given starting noise, latent z and wavelengths wvs."""
        sampler = self.sampler(self.denoiser, steps=steps)
        return sampler(x=x_noise, z=z, wvs=wvs)

    def configure_optimizers(self):
        # Optimize the backbone within the denoiser
        params = list(self.denoiser.backbone.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.lr)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.max_steps,
            base_lr=self.lr,
            final_lr=self.final_lr,
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
