import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .consistency_loss import EOConsistencyLoss


class EOVAVAELoss(nn.Module):
    """VA-VAE-style loss: EOConsistencyLoss + KL regularization + VF loss.

    Drop-in replacement for EOConsistencyLoss. Swap _target_ in config to enable.
    The existing EOConsistencyLoss is untouched.

    VF loss aligns latent z with aux_feature (a spatial feature map) via:
      - vf_loss_1: distance-matrix alignment (cosine sim matrices of z vs aux)
      - vf_loss_2: direct cosine alignment between z and aux_feature

    aux_feature can be:
      - passed directly to forward() (pre-extracted features at matching resolution), or
      - computed on the fly from a frozen dofa_net; z is resized to vf_spatial_size
        to match the extracted features.
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
        dofa_net: nn.Module = None,
        # KL
        kl_weight: float = 1e-6,
        # VF distance-matrix + cosine alignment loss
        distmat_weight: float = 0.0,
        distmat_margin: float = 0.0,
        cos_weight: float = 0.0,
        cos_margin: float = 0.0,
        vf_start_step: int = 0,
        vf_spatial_size: int = 16,
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
            dofa_net=dofa_net,
        )

        self.kl_weight = kl_weight
        self.distmat_weight = distmat_weight
        self.distmat_margin = distmat_margin
        self.cos_weight = cos_weight
        self.cos_margin = cos_margin
        self.vf_start_step = vf_start_step
        self.vf_spatial_size = vf_spatial_size

        # Frozen DOFA reference for VF feature extraction
        self.dofa_net = dofa_net
        if dofa_net is not None:
            for p in dofa_net.parameters():
                p.requires_grad = False

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
        **kwargs,
    ) -> tuple[torch.Tensor, dict]:
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
        if vf_active and z is not None and global_step >= self.vf_start_step:
            # If aux_feature not provided, extract from DOFA and resize z to match
            if aux_feature is None and self.dofa_net is not None:
                aux_feature = self._extract_dofa_features(inputs, wvs)
                # Resize z to (S, S) to match DOFA feature resolution
                z_vf = F.interpolate(
                    z.float(), self.vf_spatial_size, mode='bilinear', align_corners=False
                )
            else:
                z_vf = z  # caller is responsible for compatible spatial dims

            if aux_feature is not None:
                z_flat = rearrange(z_vf, 'b c h w -> b c (h w)')
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
                        1 - self.cos_margin - F.cosine_similarity(aux_feature, z_vf)
                    ).mean()
                    total_loss = total_loss + self.cos_weight * vf_loss_2
                    logs[f'{split}/loss_vf_cos'] = vf_loss_2.detach()

        logs[f'{split}/loss_total'] = total_loss.detach()
        return total_loss, logs

    def _extract_dofa_features(
        self, inputs: torch.Tensor, wvs: torch.Tensor
    ) -> torch.Tensor:
        """Extract spatial feature map from frozen DOFA, resized to vf_spatial_size."""
        B, S = inputs.shape[0], self.vf_spatial_size
        with torch.no_grad():
            inp_224 = F.interpolate(inputs, 224, mode='bilinear', align_corners=False)
            feats = self.dofa_net.forward_features(inp_224, wvs)  # list[B, N+1, D]
            feat = feats[-1][:, 1:, :]  # remove CLS token → [B, N, D]
            N, D = feat.shape[1], feat.shape[2]
            hw = int(math.sqrt(N))
            feat_spatial = feat.permute(0, 2, 1).view(B, D, hw, hw)
            return F.interpolate(
                feat_spatial.float(), (S, S), mode='bilinear', align_corners=False
            )  # [B, D, S, S]
