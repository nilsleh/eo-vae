
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .consistency_loss import EOConsistencyLoss
from .loss_utils import hinge_d_loss, vanilla_d_loss, vanilla_g_loss


class EOVAVAELoss(nn.Module):
    """VA-VAE-style loss: EOConsistencyLoss + KL regularization + VF loss.

    Drop-in replacement for EOConsistencyLoss. Swap _target_ in config to enable.

    VF loss aligns latent z with DINOv2 features via:
      - vf_loss_1: distance-matrix alignment (cosine sim matrices of z vs aux)
      - vf_loss_2: direct cosine alignment between z and aux_feature

    Feature extraction (dino_net) and projection (linear_proj) live here.
    EOFluxVAE inspects this loss via hasattr to decide whether to extract features.

    Optional PatchGAN discriminator loss is applied only to disc_modalities batches.
    """

    def __init__(
        self,
        # Passed through to EOConsistencyLoss
        pixel_weight: float = 1.0,
        rec_loss_type: str = 'l1',
        spectral_weight: float = 0.0,
        spatial_weight: float = 0.0,
        freq_weight: float = 0.0,
        feature_weight: float = 0.0,
        msssim_weight: float = 0.0,
        spectral_start_step: int = 0,
        spatial_start_step: int = 0,
        freq_start_step: int = 0,
        feature_start_step: int = 0,
        msssim_start_step: int = 0,
        patch_factor: int = 2,
        ffl_alpha: float = 1.0,
        # KL
        kl_weight: float = 1e-6,
        # VF distance-matrix + cosine alignment loss
        distmat_weight: float = 0.0,
        distmat_margin: float = 0.0,
        cos_weight: float = 0.0,
        cos_margin: float = 0.0,
        vf_start_step: int = 0,
        # VF foundation model
        dino_net: nn.Module | None = None,
        vf_feature_dim: int = 1024,
        vf_embed_dim: int = 32,
        vf_spatial_size: int = 16,
        vf_modalities: list[str] | None = None,
        # PatchGAN discriminator
        discriminator: nn.Module | None = None,
        disc_weight: float = 0.0,
        disc_start: int = 0,
        disc_update_start_step: int = 0,
        max_d_weight: float = 1e4,
        disc_loss_type: str = 'hinge',
        normalize_disc_input: bool = True,
        disc_modalities: list[str] | None = None,
    ):
        super().__init__()
        self.base_loss = EOConsistencyLoss(
            pixel_weight=pixel_weight,
            rec_loss_type=rec_loss_type,
            spectral_weight=spectral_weight,
            spatial_weight=spatial_weight,
            freq_weight=freq_weight,
            feature_weight=feature_weight,
            msssim_weight=msssim_weight,
            spectral_start_step=spectral_start_step,
            spatial_start_step=spatial_start_step,
            freq_start_step=freq_start_step,
            feature_start_step=feature_start_step,
            msssim_start_step=msssim_start_step,
            patch_factor=patch_factor,
            ffl_alpha=ffl_alpha,
            dofa_net=None,
        )

        self.kl_weight = kl_weight
        self.distmat_weight = distmat_weight
        self.distmat_margin = distmat_margin
        self.cos_weight = cos_weight
        self.cos_margin = cos_margin
        self.vf_start_step = vf_start_step

        self.dino_net = dino_net
        self.vf_spatial_size = vf_spatial_size
        self.vf_modalities = set(vf_modalities) if vf_modalities else {'S2RGB'}
        self.linear_proj = nn.Conv2d(vf_feature_dim, vf_embed_dim, 1) if dino_net is not None else None

        # PatchGAN discriminator
        self.discriminator = discriminator
        self.disc_weight = disc_weight
        self.disc_start = disc_start
        self.disc_update_start_step = disc_update_start_step
        self.max_d_weight = max_d_weight
        self.normalize_disc_input = normalize_disc_input
        self.disc_modalities = set(disc_modalities) if disc_modalities else {'S2RGB'}
        self.disc_loss_fn = hinge_d_loss if disc_loss_type == 'hinge' else vanilla_d_loss

    def robust_normalize(self, x: torch.Tensor, clip_val: float = 3.0) -> torch.Tensor:
        return torch.clamp(x, -clip_val, clip_val) / clip_val

    def calculate_adaptive_weight(
        self, recon_loss: torch.Tensor, gan_loss: torch.Tensor, last_layer: torch.Tensor
    ) -> torch.Tensor:
        recon_grads = torch.autograd.grad(recon_loss, last_layer, retain_graph=True)[0]
        gan_grads = torch.autograd.grad(gan_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(recon_grads) / (torch.norm(gan_grads) + 1e-6)
        return torch.clamp(d_weight, 0.0, self.max_d_weight).detach()

    def forward(
        self,
        inputs: torch.Tensor,
        wvs: torch.Tensor,
        reconstructions: torch.Tensor,
        global_step: int = 0,
        split: str = 'train',
        posterior=None,
        z: torch.Tensor = None,
        aux_feature: torch.Tensor = None,
        optimizer_idx: int = 0,
        last_layer: torch.Tensor = None,
        modality: str | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict]:
        """Compute total loss.

        Args:
            inputs: Original images [B, C, H, W].
            wvs: Wavelengths tensor.
            reconstructions: Reconstructed images [B, C, H, W].
            global_step: Current training step.
            split: 'train' or 'val'.
            posterior: DiagonalGaussianDistribution from encoder (for KL).
            z: Raw encoder latent [B, z_ch, h, w] (before patch-shuffle/BN).
            aux_feature: DINOv2 features already projected to z_ch channels and
                resized to match z spatial dims [B, z_ch, h_vf, w_vf].
                When None, VF loss is skipped.
            optimizer_idx: 0 for generator, 1 for discriminator.
            last_layer: Last decoder layer weights for adaptive loss weighting.
            modality: Current batch modality string (used for disc/VF filtering).
        """
        # === Discriminator update (optimizer_idx == 1) ===
        if optimizer_idx == 1:
            train_disc = (
                self.discriminator is not None
                and self.disc_weight > 0.0
                and global_step >= self.disc_update_start_step
                and modality in self.disc_modalities
            )
            if not train_disc:
                return torch.tensor(0.0, requires_grad=True), {}

            self.discriminator.train()
            recon_norm = self.robust_normalize(reconstructions) if self.normalize_disc_input else reconstructions
            inputs_norm = self.robust_normalize(inputs) if self.normalize_disc_input else inputs

            logits_fake, logits_real = self.discriminator(recon_norm.detach(), inputs_norm, wvs)
            d_loss = self.disc_loss_fn(logits_real, logits_fake)

            logs = {
                f'{split}/disc/d_loss': d_loss.detach(),
                f'{split}/disc/logits_real': logits_real.detach().mean(),
                f'{split}/disc/logits_fake': logits_fake.detach().mean(),
            }
            return d_loss, logs

        # === Generator update (optimizer_idx == 0) ===
        total_loss, logs = self.base_loss(
            inputs, wvs, reconstructions, global_step, split, **kwargs
        )

        # KL Loss
        if self.kl_weight > 0 and posterior is not None:
            kl = posterior.kl()  # [B]
            kl_loss = torch.sum(kl) / kl.shape[0]
            total_loss = total_loss + self.kl_weight * kl_loss
            logs[f'{split}/loss_kl'] = kl_loss.detach()

        # VF Loss
        vf_active = self.distmat_weight > 0 or self.cos_weight > 0
        if vf_active and aux_feature is not None and z is not None and global_step >= self.vf_start_step:
            # z and aux_feature must have matching spatial dims (ensured by caller)
            z_flat = rearrange(z, 'b c h w -> b c (h w)')
            aux_flat = rearrange(aux_feature, 'b c h w -> b c (h w)')
            z_norm = F.normalize(z_flat, dim=1)
            aux_norm = F.normalize(aux_flat, dim=1)

            if self.distmat_weight > 0:
                sim_z = torch.einsum('bci,bcj->bij', z_norm, z_norm)
                sim_aux = torch.einsum('bci,bcj->bij', aux_norm, aux_norm)
                vf_loss_1 = F.relu(
                    torch.abs(sim_z - sim_aux) - self.distmat_margin
                ).mean()
                total_loss = total_loss + self.distmat_weight * vf_loss_1
                logs[f'{split}/loss_vf_distmat'] = vf_loss_1.detach()

            if self.cos_weight > 0:
                vf_loss_2 = F.relu(
                    1 - self.cos_margin - F.cosine_similarity(aux_feature, z)
                ).mean()
                total_loss = total_loss + self.cos_weight * vf_loss_2
                logs[f'{split}/loss_vf_cos'] = vf_loss_2.detach()

        # GAN Generator Loss
        use_gan = (
            self.discriminator is not None
            and self.disc_weight > 0.0
            and global_step >= self.disc_start
            and modality in self.disc_modalities
        )
        if use_gan:
            self.discriminator.eval()
            recon_norm = self.robust_normalize(reconstructions) if self.normalize_disc_input else reconstructions
            logits_fake, _ = self.discriminator(recon_norm, None, wvs)
            g_loss = vanilla_g_loss(logits_fake)

            if last_layer is not None:
                try:
                    d_weight = self.calculate_adaptive_weight(total_loss, g_loss, last_layer)
                except RuntimeError:
                    d_weight = torch.tensor(1.0, device=total_loss.device)
            else:
                d_weight = torch.tensor(1.0, device=total_loss.device)

            total_loss = total_loss + d_weight * self.disc_weight * g_loss
            logs[f'{split}/disc/g_loss'] = g_loss.detach()
            logs[f'{split}/disc/d_weight'] = d_weight.detach() if isinstance(d_weight, torch.Tensor) else d_weight

        logs[f'{split}/loss_total'] = total_loss.detach()
        return total_loss, logs
