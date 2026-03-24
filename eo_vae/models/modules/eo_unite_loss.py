"""EO-UNITE loss module.

Combines three objectives following UNITE:
1. rec_loss: L1 pixel reconstruction — all modalities.
2. lpips_loss: Perceptual loss via DOFA-LPIPS — configurable modalities.
   DOFALPIPS uses a frozen DOFA backbone with learnable linear projection heads.
   Wavelength-aware: works for any EO band combination including multi-spectral
   and SAR. Use lpips_modalities to control which modalities receive this loss.
3. flow_loss: Velocity matching loss for flow-matching denoiser.

Loss signature mirrors EOConsistencyLoss for compatibility with existing logging.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class EOUniteLoss(nn.Module):
    """Combined reconstruction + perceptual + flow-matching loss for EO-UNITE.

    Args:
        pixel_weight: Weight for L1 reconstruction loss.
        perceptual_weight: Weight for DOFA-LPIPS perceptual loss.
        gen_loss_weight: Weight for flow matching (denoising) loss.
        flow_steps_per_recon: Number of flow denoising steps per forward pass.
        flow_mini_batch: Chunk size for flow steps (memory management).
        lpips_modalities: Modalities for which DOFA-LPIPS is computed. Defaults
            to ['S2RGB']. Set to a broader list (e.g. ['S2RGB', 'S2L2A',
            'S1RTC']) to apply perceptual loss across all modalities.
        dofa_net: Pretrained DOFA model instance used as the perceptual backbone.
            Must expose forward_features(x, wvs) returning a list of token
            feature tensors [B, N, D] (DOFAViT v1/v2, not v3).
        gradient_checkpointing_denoiser: Enable gradient checkpointing in the
            denoiser forward pass (saves memory at cost of recomputation).
    """

    def __init__(
        self,
        pixel_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        gen_loss_weight: float = 1.0,
        flow_steps_per_recon: int = 1,
        flow_mini_batch: int = 4,
        lpips_modalities: list[str] | None = None,
        dofa_net: nn.Module | None = None,
        gradient_checkpointing_denoiser: bool = False,
    ):
        super().__init__()
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.gen_loss_weight = gen_loss_weight
        self.flow_steps_per_recon = flow_steps_per_recon
        self.flow_mini_batch = max(1, flow_mini_batch)
        self.lpips_modalities = lpips_modalities if lpips_modalities is not None else ['S2RGB']
        self.gradient_checkpointing_denoiser = gradient_checkpointing_denoiser
        self._dofa_net = dofa_net
        # DOFA-LPIPS backbone in this setup is trained with 224x224 positional embeddings.
        self.lpips_input_size = 224

        # DOFA-LPIPS is lazy-initialized on first use
        self._dofa_lpips = None

    @property
    def dofa_lpips(self):
        if self._dofa_lpips is None:
            from eo_vae.models.modules.loss_utils import DOFALPIPS
            if self._dofa_net is None:
                raise RuntimeError(
                    'EOUniteLoss: dofa_net must be provided to use DOFA-LPIPS. '
                    'Set perceptual_weight=0 or pass a dofa_net instance.'
                )
            self._dofa_lpips = DOFALPIPS(self._dofa_net)
        return self._dofa_lpips

    def forward(
        self,
        recon: Tensor,
        target: Tensor,
        wvs: Tensor,
        modality: str | None,
        z: Tensor,
        transport,
        encoder: nn.Module,
        encoder_ln: nn.Module,
        latent_tokens: Tensor,
        cond: Tensor,
        global_step: int,
        split: str = 'train',
        alibi_bias: Tensor | None = None,
    ) -> tuple[Tensor, dict]:
        """Compute combined EO-UNITE loss.

        Args:
            recon: Reconstructed images [B, C, H, W].
            target: Target images [B, C, H, W].
            wvs: Band wavelengths [C].
            modality: Batch modality name (e.g. 'S2RGB', 'S2L2A', 'S1RTC').
            z: Latent tokens after LayerNorm [B, L, D]. Detached for flow loss.
            transport: Transport object (flow matching).
            encoder: Encoder module (used as denoiser).
            encoder_ln: LayerNorm applied to encoder output.
            latent_tokens: Learnable latent token positional embeddings.
            cond: Modality conditioning vector [B, cond_dim].
            global_step: Current training step.
            split: 'train' or 'val'.
            alibi_bias: 2D ALiBi bias for attention [1, H, N, N] or None.

        Returns:
            (total_loss, log_dict)
        """
        logs = {}

        # --- Reconstruction loss (L1) ---
        rec_loss = F.l1_loss(recon, target)
        logs[f'{split}/rec_loss'] = rec_loss.detach()

        # --- Perceptual loss (DOFA-LPIPS) ---
        lpips_loss = torch.tensor(0.0, device=recon.device)
        if self.perceptual_weight > 0 and modality in self.lpips_modalities:
            lpips_module = self.dofa_lpips.to(recon.device)
            # Keep EOUnite reconstruction/pixel losses at native resolution while
            # evaluating perceptual distance on DOFA's expected 224x224 grid.
            recon_lpips = F.interpolate(
                recon,
                size=(self.lpips_input_size, self.lpips_input_size),
                mode='bilinear',
                align_corners=False,
            )
            target_lpips = F.interpolate(
                target,
                size=(self.lpips_input_size, self.lpips_input_size),
                mode='bilinear',
                align_corners=False,
            )
            lpips_loss = lpips_module(recon_lpips, target_lpips, wvs)
        logs[f'{split}/lpips_loss'] = lpips_loss.detach()

        recon_loss = self.pixel_weight * rec_loss + self.perceptual_weight * lpips_loss
        logs[f'{split}/recon_loss'] = recon_loss.detach()

        # --- Flow matching loss ---
        flow_loss = self._compute_flow_loss(
            z=z.detach(),
            transport=transport,
            encoder=encoder,
            encoder_ln=encoder_ln,
            latent_tokens=latent_tokens,
            cond=cond,
            alibi_bias=alibi_bias,
        )
        logs[f'{split}/flow_loss'] = flow_loss.detach()

        total_loss = recon_loss + self.gen_loss_weight * flow_loss
        logs[f'{split}/total_loss'] = total_loss.detach()

        return total_loss, logs

    def _compute_flow_loss(
        self,
        z: Tensor,
        transport,
        encoder: nn.Module,
        encoder_ln: nn.Module,
        latent_tokens: Tensor,
        cond: Tensor,
        alibi_bias: Tensor | None,
    ) -> Tensor:
        """Velocity matching loss across flow_steps_per_recon steps.

        Chunks steps into mini-batches for memory efficiency, following UNITE.
        """
        t_list = [
            transport.sample(z, timestep_shift=getattr(transport, 'timestep_shift_alpha', 0.0))[0]
            for _ in range(self.flow_steps_per_recon)
        ]

        flow_loss = None
        offset = 0
        for chunk_size in _iter_chunks(self.flow_steps_per_recon, self.flow_mini_batch):
            t_chunk = torch.cat(t_list[offset: offset + chunk_size], dim=0)
            offset += chunk_size

            z_chunk = z.repeat(chunk_size, 1, 1)
            cond_chunk = cond.repeat(chunk_size, 1)
            pos_embed_chunk = latent_tokens[:, :z.shape[1]].expand(z_chunk.shape[0], -1, -1)

            model_kwargs = dict(
                pos_embed=pos_embed_chunk,
                precomputed_cond=cond_chunk,
                checkpoint_blocks=self.gradient_checkpointing_denoiser,
                alibi_bias=alibi_bias,
            )

            flow_dict = transport.training_losses(encoder, z_chunk, t_chunk, model_kwargs=model_kwargs)
            chunk_loss = _velocity_loss(flow_dict, encoder_ln, transport.train_eps)
            weighted = chunk_loss * chunk_size
            flow_loss = weighted if flow_loss is None else (flow_loss + weighted)

        return flow_loss / self.flow_steps_per_recon


def _velocity_loss(flow_dict: dict, encoder_ln: nn.Module, train_eps: float) -> Tensor:
    """Compute velocity prediction MSE loss from transport output dict."""
    x1 = flow_dict['x1']
    xt = flow_dict['xt']
    t = flow_dict['sampled_t'][:, None, None]
    model_out = encoder_ln(flow_dict['model_output'])
    v_gt = (x1 - xt) / (1 - t).clamp_min(train_eps)
    v_pred = (model_out - xt) / (1 - t).clamp_min(train_eps)
    return ((v_gt - v_pred) ** 2).mean(dim=(1, 2)).mean()


def _iter_chunks(total: int, chunk_size: int) -> list[int]:
    """Split total into chunk_size chunks (last chunk may be smaller)."""
    if total <= 0:
        return []
    full, remainder = divmod(total, chunk_size)
    chunks = [chunk_size] * full
    if remainder:
        chunks.append(remainder)
    return chunks
