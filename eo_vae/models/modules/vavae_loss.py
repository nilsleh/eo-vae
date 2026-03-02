import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .consistency_loss import EOConsistencyLoss


class EOVAVAELoss(nn.Module):
    """VA-VAE-style loss: EOConsistencyLoss + KL regularization + VF distance-matrix loss.

    Drop-in replacement for EOConsistencyLoss. Swap _target_ in config to enable.
    The existing EOConsistencyLoss is untouched.
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
        # VF distance-matrix loss
        vf_weight: float = 0.0,
        vf_distmat_margin: float = 0.0,
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
        self.vf_weight = vf_weight
        self.vf_distmat_margin = vf_distmat_margin
        self.vf_start_step = vf_start_step
        self.vf_spatial_size = vf_spatial_size

        # Frozen DOFA reference for VF loss (reuse the same network)
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
        z=None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict]:
        total_loss, logs = self.base_loss(
            inputs, wvs, reconstructions, global_step, split, **kwargs
        )

        # KL Loss
        if self.kl_weight > 0 and posterior is not None:
            kl = posterior.kl()  # [B]
            kl_loss = torch.sum(kl) / kl.shape[0]  # scalar
            total_loss = total_loss + self.kl_weight * kl_loss
            logs[f'{split}/loss_kl'] = kl_loss.detach()

        # VF Distance-Matrix Loss
        if self.vf_weight > 0 and z is not None and self.dofa_net is not None:
            if global_step >= self.vf_start_step:
                vf_loss = self._compute_vf_loss(inputs, wvs, z)
                total_loss = total_loss + self.vf_weight * vf_loss
                logs[f'{split}/loss_vf'] = vf_loss.detach()

        logs[f'{split}/loss_total'] = total_loss.detach()
        return total_loss, logs

    def _compute_vf_loss(
        self, inputs: torch.Tensor, wvs: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        B, S = inputs.shape[0], self.vf_spatial_size

        # DOFA side — no gradients
        with torch.no_grad():
            inp_224 = F.interpolate(inputs, 224, mode='bilinear', align_corners=False)
            feats = self.dofa_net.forward_features(inp_224, wvs)  # list[B, N+1, D]
            feat = feats[-1][:, 1:, :]  # remove CLS → [B, N, D]
            N, D = feat.shape[1], feat.shape[2]
            hw = int(math.sqrt(N))
            feat_spatial = feat.permute(0, 2, 1).view(B, D, hw, hw)
            feat_r = F.interpolate(
                feat_spatial.float(), (S, S), mode='bilinear', align_corners=False
            )
            feat_flat = feat_r.view(B, D, S * S)
            feat_norm = F.normalize(feat_flat, dim=1)
            sim_feat = torch.einsum('bci,bcj->bij', feat_norm, feat_norm)  # [B, S², S²]

        # z side — gradients flow to encoder
        z_r = F.interpolate(z.float(), (S, S), mode='bilinear', align_corners=False)
        z_flat = z_r.view(B, -1, S * S)
        z_norm = F.normalize(z_flat, dim=1)
        sim_z = torch.einsum('bci,bcj->bij', z_norm, z_norm)  # [B, S², S²]

        return F.relu(torch.abs(sim_z - sim_feat) - self.vf_distmat_margin).mean()
