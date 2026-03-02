"""Clean EO-VAE Lightning Module for Finetuning.

This module handles:
- Standard VAE training with reconstruction + KL loss
- Optional discriminator training (GAN loss)
- EQ-VAE regularization (latent equivariance)
- Multi-modality support via wavelength conditioning

For weight distillation, see: distill.py
"""

import math
import os
import random

import torch
import torch.nn.functional as F
from einops import rearrange
from lightning import LightningModule
from safetensors import safe_open
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from .modules.distributions import DiagonalGaussianDistribution

# =============================================================================
# LEARNING RATE SCHEDULER
# =============================================================================


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    base_lr: float,
    final_lr: float,
    num_cycles: float = 0.5,
) -> LambdaLR:
    """Create a schedule with linear warmup and cosine decay."""

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        lr_scale = (base_lr - final_lr) * cosine_decay + final_lr
        return lr_scale / base_lr

    return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# EO-VAE LIGHTNING MODULE
# =============================================================================


class EOFluxVAE(LightningModule):
    """Earth Observation VAE based on Flux architecture.

    This module supports multi-spectral satellite imagery with dynamic
    wavelength-conditioned input/output layers.

    Training modes:
        - Standard reconstruction (MSE/perceptual + KL)
        - GAN training (with discriminator)
        - EQ-VAE regularization (latent equivariance)
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        loss_fn: torch.nn.Module,
        # Checkpoint
        ckpt_path: str | None = None,
        ignore_keys: list[str] | None = None,
        # Training config
        freeze_body: bool = True,
        base_lr: float = 1e-4,
        final_lr: float | None = None,
        warmup_epochs: int | None = None,
        decay_end_epoch: int | None = None,
        clip_grad: float | None = None,
        # EQ-VAE Hyperparameters
        p_prior: float = 0.0,
        p_prior_s: float = 0.0,
        anisotropic: bool = False,
        # optiional latent noise
        latent_noise_p: float = 0.0,
        noise_tau: float = 0.8,
        # Data
        image_key: str = 'image',
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = loss_fn

        self.image_key = image_key
        self.base_lr = base_lr
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.decay_end_epoch = decay_end_epoch
        self.clip_grad = clip_grad

        # EQ-VAE
        self.p_prior = p_prior
        self.p_prior_s = p_prior_s
        self.anisotropic = anisotropic

        self.latent_noise_p = latent_noise_p
        self.noise_tau = noise_tau

        # Flux latent processing
        self.ps = [2, 2]  # Patch size for latent shuffling
        self.bn_eps = 1e-4
        self.bn = torch.nn.BatchNorm2d(
            math.prod(self.ps) * encoder.z_channels,
            affine=False,
            track_running_stats=True,
        )

        # Manual optimization for GAN training
        self.automatic_optimization = False

        # Freeze body if requested
        self.freeze_body = freeze_body
        if self.freeze_body:
            self._freeze_body()

        # Load checkpoint
        if ckpt_path:
            self._load_checkpoint(ckpt_path, ignore_keys or [])

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def _freeze_body(self) -> None:
        """Freeze VAE body, keeping only dynamic layers trainable."""
        print('Freezing VAE body (dynamic layers remain trainable)')

        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False

        # Unfreeze dynamic input layer
        if self.encoder.use_dynamic_ops:
            for p in self.encoder.conv_in.parameters():
                p.requires_grad = True
            print('  -> Encoder conv_in: TRAINABLE')

        # Unfreeze dynamic output layer
        if self.decoder.use_dynamic_ops:
            for p in self.decoder.conv_out.parameters():
                p.requires_grad = True
            print('  -> Decoder conv_out: TRAINABLE')

        # Unfreeze adain conditioning (if present)
        if getattr(self.encoder, 'use_adain', False):
            for p in self.encoder.conditioner.parameters():
                p.requires_grad = True
            for m in self.encoder.modules():
                if hasattr(m, 'emb_proj'):
                    for p in m.emb_proj.parameters():
                        p.requires_grad = True
            print('  -> Encoder conditioner + emb_proj: TRAINABLE')

        if getattr(self.decoder, 'use_adain', False):
            for p in self.decoder.conditioner.parameters():
                p.requires_grad = True
            for m in self.decoder.modules():
                if hasattr(m, 'emb_proj'):
                    for p in m.emb_proj.parameters():
                        p.requires_grad = True
            print('  -> Decoder conditioner + emb_proj: TRAINABLE')

    def _load_checkpoint(self, path: str, ignore_keys: list[str]) -> None:
        """Load pretrained weights from checkpoint.

        Supports three checkpoint formats:
        1. Flux VAE checkpoint (.safetensors) - loads body weights, skips dynamic layers
        2. Distilled checkpoint (.pt with 'encoder_conv_in_state_dict') - loads dynamic layers
        3. Full EO-VAE checkpoint (.ckpt) - loads everything
        """
        if not os.path.exists(path):
            print(f'Checkpoint not found: {path}')
            return

        print(f'Loading weights from {path}')

        # Check if this is a distilled checkpoint
        if path.endswith('.pt'):
            ckpt = torch.load(path, map_location='cpu')
            if (
                'encoder_conv_in_state_dict' in ckpt
                or 'decoder_conv_out_state_dict' in ckpt
            ):
                self._load_distilled_checkpoint(ckpt)
                return

        # Load state dict (safetensors or regular checkpoint)
        if path.endswith('.safetensors'):
            sd = {}
            with safe_open(path, framework='pt', device='cpu') as f:
                for k in f.keys():
                    sd[k] = f.get_tensor(k)
        else:
            sd = torch.load(path, map_location='cpu')
            sd = sd.get('state_dict', sd)

        # Filter keys for dynamic layers
        keys_to_remove = []
        for k in sd.keys():
            # Skip static conv_in/conv_out if using dynamic ops
            if self.encoder.use_dynamic_ops and 'encoder.conv_in' in k:
                if 'weight_generator' not in k and 'fclayer' not in k:
                    keys_to_remove.append(k)
                    continue

            if self.decoder.use_dynamic_ops and 'decoder.conv_out' in k:
                if 'weight_generator' not in k and 'fclayer' not in k:
                    keys_to_remove.append(k)
                    continue

            # User-specified ignore keys
            for ik in ignore_keys:
                if k.startswith(ik):
                    keys_to_remove.append(k)
                    break

        for k in keys_to_remove:
            del sd[k]

        # Load
        missing, unexpected = self.load_state_dict(sd, strict=False)

        # Verify critical weights loaded
        self._verify_loading(missing, unexpected, ignore_keys)

    def _load_distilled_checkpoint(self, ckpt: dict) -> None:
        """Load weights from a distillation checkpoint.

        Args:
            ckpt: Checkpoint dict with encoder/decoder state dicts
        """
        print('Detected distillation checkpoint format')

        # Load encoder dynamic layer
        if self.encoder.use_dynamic_ops and ckpt.get('encoder_conv_in_state_dict'):
            self.encoder.conv_in.load_state_dict(ckpt['encoder_conv_in_state_dict'])
            print('  Loaded encoder.conv_in from distilled checkpoint')

        # Load decoder dynamic layer
        if self.decoder.use_dynamic_ops and ckpt.get('decoder_conv_out_state_dict'):
            self.decoder.conv_out.load_state_dict(ckpt['decoder_conv_out_state_dict'])
            print('  Loaded decoder.conv_out from distilled checkpoint')

        # Log distillation info if available
        if 'distill_config' in ckpt:
            print(
                f'  Distillation loss was: {ckpt["distill_config"].get("final_loss", "N/A")}'
            )

    def _verify_loading(
        self,
        missing_keys: list[str],
        unexpected_keys: list[str],
        ignore_keys: list[str],
    ) -> None:
        """Verify that critical weights were loaded correctly."""
        allowed_missing = []

        if self.encoder.use_dynamic_ops:
            allowed_missing.append('encoder.conv_in')
        if self.decoder.use_dynamic_ops:
            allowed_missing.append('decoder.conv_out')
        if getattr(self.encoder, 'use_adain', False):
            allowed_missing.append('encoder.conditioner')
        if getattr(self.decoder, 'use_adain', False):
            allowed_missing.append('decoder.conditioner')
        allowed_missing.extend(ignore_keys)

        # emb_proj lives inside ResnetBlocks; allow it missing when adain is enabled
        # (new params not present in Flux/distilled checkpoints)
        adain_active = getattr(self.encoder, 'use_adain', False) or getattr(
            self.decoder, 'use_adain', False
        )

        critical_missing = []
        for k in missing_keys:
            if any(k.startswith(p) for p in allowed_missing):
                continue
            if adain_active and 'emb_proj' in k:
                continue
            critical_missing.append(k)

        if critical_missing:
            raise RuntimeError(
                f'Critical weights missing from checkpoint:\n'
                f'{critical_missing[:20]}...\n'
                f'Total: {len(critical_missing)} missing keys'
            )

        print(
            f'Checkpoint loaded: {len(missing_keys)} missing (expected), '
            f'{len(unexpected_keys)} unexpected (ignored)'
        )

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    def encode(self, x: Tensor, wvs: Tensor) -> DiagonalGaussianDistribution:
        """Encode image to latent distribution."""
        moments = self.encoder(x, wvs)
        return DiagonalGaussianDistribution(moments)

    def decode(self, z: Tensor, wvs: Tensor) -> Tensor:
        """Decode latent to image."""
        z = self._inv_normalize_latent(z)
        z = rearrange(
            z, '... (c pi pj) i j -> ... c (i pi) (j pj)', pi=self.ps[0], pj=self.ps[1]
        )
        return self.decoder(z, wvs)

    def decode_raw(self, z: Tensor, wvs: Tensor) -> Tensor:
        """Decode raw latent (unshuffled, no BN) to image.

        Use this when decoding latents that come directly from the encoder output
        (e.g., during super-res training/inference where we skip the VAE's internal
        regularization layers).
        """
        return self.decoder(z, wvs)

    def noising(self, x: torch.Tensor) -> torch.Tensor:
        noise_sigma = self.noise_tau * torch.rand(
            (x.size(0),) + (1,) * (len(x.shape) - 1), device=x.device
        )
        noise = noise_sigma * torch.randn_like(x)
        return x + noise

    def forward(
        self,
        x: Tensor,
        wvs: Tensor,
        sample_posterior: bool = True,
        scale: float | tuple[float, float] | None = None,
        angle: int | None = None,
        return_z: bool = False,
    ) -> (
        tuple[Tensor, DiagonalGaussianDistribution]
        | tuple[Tensor, DiagonalGaussianDistribution, Tensor]
    ):
        """Full forward pass: encode -> transform -> decode."""
        posterior = self.encode(x, wvs)
        z = posterior.sample() if sample_posterior else posterior.mode()
        z_raw = z  # save before EQ-VAE transforms

        # EQ-VAE transformations
        if scale is not None:
            z = self._apply_scale(z, scale)
        if angle is not None:
            z = torch.rot90(z, k=angle, dims=[-1, -2])

        # Shuffle and normalize
        z_shuffled = rearrange(
            z, '... c (i pi) (j pj) -> ... (c pi pj) i j', pi=self.ps[0], pj=self.ps[1]
        )
        z_normalized = self._normalize_latent(z_shuffled)

        # optionally add small amount of noise during training for robustness
        if self.training:
            # if random > latent_noise_p, add noise
            if random.random() < self.latent_noise_p:
                z_normalized = self.noising(z_normalized)

        reconstruction = self.decode(z_normalized, wvs)
        if return_z:
            return reconstruction, posterior, z_raw
        return reconstruction, posterior

    @torch.no_grad()
    def encode_spatial_normalized(self, x: Tensor, wvs: Tensor) -> Tensor:
        """Encode to spatially-structured normalized latent.

        Process:
        1. Encode -> z
        2. Shuffle -> z_shuffled
        3. BN using VAE stats -> z_norm
        4. Unshuffle -> z_spatial

        Returns: [B, C, H, W] where C=32, preserving spatial layout but with VAE normalization applied.
        """
        # Get normalized packed latent [B, 128, H/16, W/16]
        z_norm = self.encode_to_latent(x, wvs)

        # Unshuffle back to spatial [B, 32, H/8, W/8]
        z_spatial = rearrange(
            z_norm,
            '... (c pi pj) i j -> ... c (i pi) (j pj)',
            pi=self.ps[0],
            pj=self.ps[1],
        )
        return z_spatial

    @torch.no_grad()
    def decode_spatial_normalized(self, z: Tensor, wvs: Tensor) -> Tensor:
        """Decode from spatially-structured normalized latent.

        Process:
        1. Shuffle -> z_packed
        2. Inverse BN (handled by decode)
        3. Unshuffle & Decode (handled by decode)
        """
        # Shuffle to packed format [B, 128, H/16, W/16]
        z_packed = rearrange(
            z, '... c (i pi) (j pj) -> ... (c pi pj) i j', pi=self.ps[0], pj=self.ps[1]
        )
        # Decode expects packed normalized latent
        return self.decode(z_packed, wvs)

    def _apply_scale(self, z: Tensor, scale: float | tuple[float, float]) -> Tensor:
        """Apply scale transformation to latent."""
        h, w = z.shape[-2:]
        if isinstance(scale, (tuple, list)):
            new_h = round(h * scale[0] / self.ps[0]) * self.ps[0]
            new_w = round(w * scale[1] / self.ps[1]) * self.ps[1]
        else:
            new_h = round(h * scale / self.ps[0]) * self.ps[0]
            new_w = round(w * scale / self.ps[1]) * self.ps[1]
        return F.interpolate(
            z, size=(new_h, new_w), mode='bilinear', align_corners=False
        )

    def _normalize_latent(self, z: Tensor) -> Tensor:
        """Apply batch normalization to latent."""
        self.bn.train() if self.training else self.bn.eval()
        return self.bn(z)

    def _inv_normalize_latent(self, z: Tensor) -> Tensor:
        """Inverse batch normalization."""
        self.bn.eval()
        s = torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + self.bn_eps)
        m = self.bn.running_mean.view(1, -1, 1, 1)
        return z * s + m

    # =========================================================================
    # TRAINING
    # =========================================================================

    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        # Collect trainable parameters
        ae_params = [p for p in self.encoder.parameters() if p.requires_grad] + [
            p for p in self.decoder.parameters() if p.requires_grad
        ]

        opt_ae = torch.optim.Adam(ae_params, lr=self.base_lr)
        optimizers = [opt_ae]
        schedulers = []

        # Discriminator optimizer (if present)
        if hasattr(self.loss_fn, 'discriminator'):
            opt_disc = torch.optim.Adam(
                self.loss_fn.discriminator.parameters(), lr=self.base_lr
            )
            optimizers.append(opt_disc)

        # Learning rate schedulers
        if all([self.final_lr, self.warmup_epochs, self.decay_end_epoch]):
            # estimated_stepping_batches reflects the actual optimizer step count
            # (DistributedSampler already divides data by num_devices, and global_step
            # increments once per optimizer step regardless of num_devices).
            steps_per_epoch = (
                self.trainer.estimated_stepping_batches // self.trainer.max_epochs
            )
            num_warmup = self.warmup_epochs * steps_per_epoch
            num_total = self.decay_end_epoch * steps_per_epoch

            for opt in optimizers:
                sch = get_cosine_schedule_with_warmup(
                    opt,
                    num_warmup_steps=num_warmup,
                    num_training_steps=num_total,
                    base_lr=self.base_lr,
                    final_lr=self.final_lr,
                )
                schedulers.append({'scheduler': sch, 'interval': 'step'})

        if schedulers:
            return optimizers, schedulers
        return optimizers

    def training_step(self, batch, batch_idx):
        """Training step with optional EQ-VAE regularization."""
        opts = self.optimizers()
        opt_gen = opts[0] if isinstance(opts, list) else opts
        opt_disc = opts[1] if isinstance(opts, list) and len(opts) > 1 else None

        schs = self.lr_schedulers()
        sch_gen = schs[0] if isinstance(schs, list) and schs else schs
        sch_disc = schs[1] if isinstance(schs, list) and len(schs) > 1 else None

        images = batch[self.image_key]
        wvs = batch['wvs']

        # === EQ-VAE Mode Selection ===
        scale_bins = [0.375, 0.5, 0.75]
        scale, angle = None, None
        target_images = images

        if random.random() < self.p_prior:
            # Latent equivariance mode
            angle = random.choice([1, 2, 3])
            scale = (
                (random.choice(scale_bins), random.choice(scale_bins))
                if self.anisotropic
                else random.choice(scale_bins)
            )
            recon, posterior, z = self.forward(
                images, wvs, scale=scale, angle=angle, return_z=True
            )
            with torch.no_grad():
                target_images = F.interpolate(
                    images, size=recon.shape[-2:], mode='area'
                )
                target_images = torch.rot90(target_images, k=angle, dims=[-1, -2])

        elif random.random() < self.p_prior_s:
            # Prior preservation mode
            scale = random.choice(scale_bins)
            recon, posterior, z = self.forward(images, wvs, scale=scale, return_z=True)
            with torch.no_grad():
                target_images = F.interpolate(
                    images, size=recon.shape[-2:], mode='area'
                )

        else:
            # Standard reconstruction
            recon, posterior, z = self.forward(images, wvs, return_z=True)

        # === Generator Training ===
        opt_gen.zero_grad()
        if opt_disc is not None and hasattr(self.loss_fn, 'discriminator'):
            self.loss_fn.discriminator.eval()

        gen_loss, log_dict_gen = self.loss_fn(
            inputs=target_images,
            wvs=wvs,
            reconstructions=recon,
            optimizer_idx=0,
            global_step=self.global_step,
            last_layer=self.get_last_layer(),
            split='train',
            posterior=posterior,
            z=z,
        )

        self.manual_backward(gen_loss)
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(
                opt_gen.param_groups[0]['params'], self.clip_grad
            )
        opt_gen.step()
        if sch_gen:
            sch_gen.step()

        # === Discriminator Training ===
        train_disc = (
            opt_disc is not None
            and self.global_step >= self.loss_fn.disc_start
            and self.loss_fn.disc_weight > 0.0
        )

        if train_disc:
            if hasattr(self.loss_fn, 'discriminator'):
                self.loss_fn.discriminator.train()

            opt_disc.zero_grad()
            disc_loss, log_dict_disc = self.loss_fn(
                inputs=target_images,
                wvs=wvs,
                reconstructions=recon.detach(),
                optimizer_idx=1,
                global_step=self.global_step,
                last_layer=None,
                split='train',
            )
            self.manual_backward(disc_loss)
            opt_disc.step()
            if sch_disc:
                sch_disc.step()
            log_dict_gen.update(log_dict_disc)

        # Logging
        log_dict_gen['train/lr'] = opt_gen.param_groups[0]['lr']
        self.log_dict(
            log_dict_gen, prog_bar=True, logger=True, on_step=True, on_epoch=False
        )

        return gen_loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        images = batch[self.image_key]
        wvs = batch['wvs']

        recon, posterior, z = self.forward(images, wvs, return_z=True)

        val_loss, log_dict = self.loss_fn(
            inputs=images,
            wvs=wvs,
            reconstructions=recon,
            optimizer_idx=0,
            global_step=self.global_step,
            last_layer=None,
            split='val',
            posterior=posterior,
            z=z,
        )

        self.log_dict(
            log_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True
        )
        return val_loss

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def get_last_layer(self) -> Tensor:
        """Get last layer weight for adaptive loss weighting."""
        if hasattr(self.decoder, 'output_conv_weight'):
            return self.decoder.output_conv_weight
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def reconstruct(self, x: Tensor, wvs: Tensor) -> Tensor:
        """Reconstruct image (inference mode)."""
        recon, _ = self.forward(x, wvs, sample_posterior=False)
        return recon

    @torch.no_grad()
    def encode_to_latent(self, x: Tensor, wvs: Tensor) -> Tensor:
        """Encode to normalized latent space."""
        posterior = self.encode(x, wvs)
        z = posterior.mode()
        z_shuffled = rearrange(
            z, '... c (i pi) (j pj) -> ... (c pi pj) i j', pi=self.ps[0], pj=self.ps[1]
        )
        return self._normalize_latent(z_shuffled)
