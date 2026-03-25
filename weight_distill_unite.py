#!/usr/bin/env python
"""EO-Unite IO Layer Weight Distillation.

Warm-starts the patch_embed, unpatchify, modality_conditioner and cond_proj
of EO-Unite by training them on RGB images with the UNITE body frozen.

Teacher signal: RGB patch tokens from the pretrained UNITE-B patch embedding
(Conv2d(3, 768, 16, stride=16) stored in UNITE-B.pt).

Loss: L1 reconstruction + token MSE between student and teacher patch tokens.

Usage:
    python weight_distill_unite.py \\
        --config configs/eo-unite-panopticon.yaml \\
        --unite-ckpt /path/to/UNITE-B.pt \\
        --max-epochs 10

    python weight_distill_unite.py \\
        --config configs/eo-unite-hypernetwork.yaml \\
        --unite-ckpt /path/to/UNITE-B.pt \\
        --debug

The script overrides the datamodule to train on S2RGB only, so you can pass
any existing eo-unite config without modification.

Output: distilled_io.pt containing state dicts for the four IO modules.
"""

import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import OmegaConf
from torch import Tensor

from eo_vae.utils.image_logger import ImageLogger

OmegaConf.register_new_resolver('eval', eval)

# RGB wavelengths in µm (R, G, B)
RGB_WAVELENGTHS = [0.665, 0.560, 0.490]


# =============================================================================
# DISTILLATION LIGHTNING MODULE
# =============================================================================


class UniteDistillModule(L.LightningModule):
    """Trains EO-Unite IO layers on RGB images with a frozen UNITE body.

    Args:
        model: EOUnite instance with body already frozen.
        teacher_conv: Frozen Conv2d(3, D, P, stride=P) from UNITE-B.pt.
        token_mse_weight: Weight for the token MSE loss term.
        base_lr: Peak learning rate.
        final_lr: Minimum LR after cosine decay.
        warmup_epochs: Linear warmup epochs.
        decay_end_epoch: Epoch at which cosine decay reaches final_lr.
    """

    def __init__(
        self,
        model: nn.Module,
        teacher_conv: nn.Module,
        token_mse_weight: float = 1.0,
        base_lr: float = 1e-4,
        final_lr: float = 1e-5,
        warmup_epochs: int = 1,
        decay_end_epoch: int = 10,
    ):
        super().__init__()
        self.model = model
        self.teacher_conv = teacher_conv
        self.token_mse_weight = token_mse_weight
        self.base_lr = base_lr
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.decay_end_epoch = decay_end_epoch

        # Freeze teacher
        for p in self.teacher_conv.parameters():
            p.requires_grad = False

        self.register_buffer(
            'rgb_wvs', torch.tensor(RGB_WAVELENGTHS, dtype=torch.float32)
        )

    def forward(self, images: Tensor, wvs: Tensor | None = None) -> Tensor:
        """Reconstruction forward used by callbacks such as ImageLogger.

        During distillation training we use ``_forward(batch)`` to compute both
        student/teacher tokens and reconstruction. This method exists so generic
        callback code can call ``pl_module(images, wvs)`` for visualization.
        """
        if wvs is None:
            wvs = self.rgb_wvs
        if not isinstance(wvs, torch.Tensor):
            wvs = torch.as_tensor(wvs, dtype=torch.float32)
        wvs = wvs.to(images.device)
        return self._reconstruct(images, wvs, batch=None)

    def _compute_cond(self, images: Tensor, wvs: Tensor, batch: dict | None) -> Tensor:
        """Build modality conditioning; gracefully fall back when metadata is absent."""
        geo = None
        time = None
        if batch is not None:
            geo = self.model._build_geo(batch, images.device, images.dtype)
            time = batch.get('time')
            if time is not None:
                time = time.float().to(images.device)

        cond_raw = self.model.modality_conditioner(wvs, geo=geo, time=time)
        cond = self.model.cond_proj(cond_raw)

        # Some conditioner implementations return a single conditioning vector
        # when metadata is missing; expand to batch size for encoder use.
        if cond.ndim == 2 and cond.shape[0] == 1 and images.shape[0] > 1:
            cond = cond.expand(images.shape[0], -1)
        return cond

    def _get_teacher_patch_hidden(
        self, images: Tensor, teacher_tokens: Tensor, batch: dict | None
    ) -> Tensor:
        """Run the frozen encoder/decoder with teacher tokens; returns patch_hidden.

        Using teacher tokens (not student) ensures the frozen body always sees the
        correct token distribution, giving unpatchify a stable training signal from
        the very first epoch.
        """
        wvs = self.rgb_wvs.to(images.device)
        cond = self._compute_cond(images, wvs, batch=batch)

        B = images.shape[0]
        z_noise = torch.randn(
            B, self.model.num_latent_tokens, self.model.diffusion_input_dim,
            device=images.device, dtype=images.dtype,
        )
        pos_embed = self.model.latent_tokens[:, :self.model.num_latent_tokens].expand(B, -1, -1)
        flow_t = torch.rand(B, device=images.device) * self.model.modulation_recon_timestep_max

        z = self.model.encoder(
            z_noise, t=flow_t, pos_embed=pos_embed,
            img_patch_embed=teacher_tokens, precomputed_cond=cond,
            alibi_bias=self.model.alibi,
        )
        z = z[:, :self.model.num_latent_tokens]
        z_normed = self.model.encoder_ln(z)
        z_up = self.model.up_sample_decoder(z_normed)
        return self.model.decoder(z_up, drop_cls_token=False, return_hidden=True)

    def _reconstruct(self, images: Tensor, wvs: Tensor, batch: dict | None) -> Tensor:
        """Reconstruction for callbacks (e.g. ImageLogger): uses teacher tokens."""
        with torch.no_grad():
            t_feat = self.teacher_conv(images)
            teacher_tokens = t_feat.flatten(2).transpose(1, 2)
            patch_hidden = self._get_teacher_patch_hidden(images, teacher_tokens, batch)
        return self.model.unpatchify(patch_hidden, wvs)

    def _forward(self, batch: dict) -> tuple[Tensor, Tensor, Tensor]:
        """Run one forward pass. Returns (recon, student_tokens, teacher_tokens)."""
        images = batch['image']   # [B, 3, H, W]
        wvs = self.rgb_wvs.to(images.device)

        # --- Teacher tokens (no grad on teacher conv) ---
        with torch.no_grad():
            t_feat = self.teacher_conv(images)
            teacher_tokens = t_feat.flatten(2).transpose(1, 2)  # [B, P², D]

        # --- Student token path (trains patch_embed via token_mse) ---
        student_tokens = self.model.patch_embed(images, wvs)  # [B, P², D]

        # --- Teacher reconstruction path (trains unpatchify via recon_loss) ---
        # Use teacher tokens so the frozen body sees its native token distribution.
        # This decouples unpatchify training from the early-stage noisy student tokens.
        with torch.no_grad():
            patch_hidden = self._get_teacher_patch_hidden(images, teacher_tokens, batch)
        recon = self.model.unpatchify(patch_hidden, wvs)

        return recon, student_tokens, teacher_tokens

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        recon, student_tokens, teacher_tokens = self._forward(batch)
        images = batch['image']

        recon_loss = F.l1_loss(recon, images)
        token_mse = F.mse_loss(student_tokens, teacher_tokens)
        total_loss = recon_loss + self.token_mse_weight * token_mse

        self.log_dict({
            'train/recon_loss': recon_loss,
            'train/token_mse': token_mse,
            'train/total_loss': total_loss,
        }, prog_bar=True, on_step=True, on_epoch=False)
        return total_loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        recon, student_tokens, teacher_tokens = self._forward(batch)
        images = batch['image']

        recon_loss = F.l1_loss(recon, images)
        token_mse = F.mse_loss(student_tokens, teacher_tokens)
        total_loss = recon_loss + self.token_mse_weight * token_mse

        self.log_dict({
            'val/recon_loss': recon_loss,
            'val/token_mse': token_mse,
            'val/total_loss': total_loss,
        }, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        trainable = (
            list(self.model.patch_embed.parameters())
            + list(self.model.unpatchify.parameters())
            + list(self.model.modality_conditioner.parameters())
            + list(self.model.cond_proj.parameters())
        )
        opt = torch.optim.AdamW(trainable, lr=self.base_lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, self._lr_lambda)
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}

    def _lr_lambda(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return (epoch + 1) / max(1, self.warmup_epochs)
        progress = (epoch - self.warmup_epochs) / max(1, self.decay_end_epoch - self.warmup_epochs)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress)).item())
        return self.final_lr / self.base_lr + (1 - self.final_lr / self.base_lr) * cosine


# =============================================================================
# TEACHER EXTRACTION
# =============================================================================


def load_teacher_patch_proj(unite_ckpt_path: str) -> nn.Conv2d:
    """Extract the RGB patch projection Conv2d from UNITE-B.pt."""
    state = torch.load(unite_ckpt_path, map_location='cpu', weights_only=False)
    sd = state.get('model', state)

    w_key = 'patch_embed.patch_embeddings.projection.weight'
    b_key = 'patch_embed.patch_embeddings.projection.bias'

    if w_key not in sd:
        raise KeyError(
            f'Could not find patch embedding weights in UNITE checkpoint.\n'
            f"Expected key: '{w_key}'\n"
            f"Available keys with 'patch': {[k for k in sd if 'patch' in k]}"
        )

    w = sd[w_key]   # [out_ch, in_ch, kH, kW]
    b = sd[b_key]
    out_ch, in_ch, kH, kW = w.shape
    print(f'[Distill] Teacher patch proj: Conv2d({in_ch}, {out_ch}, {kH}, stride={kH})')

    conv = nn.Conv2d(in_ch, out_ch, kernel_size=kH, stride=kH, bias=True)
    conv.weight = nn.Parameter(w, requires_grad=False)
    conv.bias = nn.Parameter(b, requires_grad=False)
    return conv


# =============================================================================
# DATAMODULE OVERRIDE (S2RGB only)
# =============================================================================


def make_rgb_datamodule(config, batch_size: int | None = None):
    """Instantiate TerraMeshDataModule forced to S2RGB for both train and val."""
    dm_cfg = OmegaConf.to_container(config.datamodule, resolve=True)
    dm_cfg['modalities'] = ['S2RGB']
    dm_cfg['train_collate_mode'] = 'S2RGB'
    dm_cfg['val_collate_mode'] = 'S2RGB'
    dm_cfg['return_metadata'] = True  # for lat/lon conditioning
    if batch_size is not None:
        dm_cfg['batch_size'] = batch_size
    from omegaconf import DictConfig
    return instantiate(DictConfig(dm_cfg))


# =============================================================================
# CHECKPOINT SAVE / LOAD
# =============================================================================


def save_distilled_io_checkpoint(module: UniteDistillModule, save_path: str) -> None:
    """Save IO layer state dicts to a .pt file."""
    model = module.model
    checkpoint = {
        'patch_embed_state_dict': model.patch_embed.state_dict(),
        'unpatchify_state_dict': model.unpatchify.state_dict(),
        'modality_conditioner_state_dict': model.modality_conditioner.state_dict(),
        'cond_proj_state_dict': model.cond_proj.state_dict(),
        'patch_embed_type': model.patch_embed_type,
        'diffusion_input_dim': model.diffusion_input_dim,
        'rgb_wavelengths': RGB_WAVELENGTHS,
    }
    torch.save(checkpoint, save_path)
    print(f'[Distill] IO checkpoint saved → {save_path}')


# =============================================================================
# MAIN
# =============================================================================


def run_distillation(
    config,
    unite_ckpt: str,
    max_epochs: int = 10,
    token_mse_weight: float = 1.0,
    base_lr: float = 1e-4,
    final_lr: float = 1e-5,
    save_dir: str | None = None,
    debug: bool = False,
) -> str:
    torch.set_float32_matmul_precision('medium')

    # --- Build EOUnite model ---
    print('Instantiating EOUnite...')
    loss_fn = instantiate(config.model.loss_fn)
    model = instantiate(config.model, loss_fn=loss_fn)

    # Load UNITE body
    print(f'Loading UNITE body from {unite_ckpt}')
    model._load_checkpoint(unite_ckpt)

    # Freeze body (encoder, decoder, latent_tokens, encoder_ln, up_sample_decoder)
    model._freeze_body()

    # --- Build teacher ---
    teacher_conv = load_teacher_patch_proj(unite_ckpt)

    # --- Distillation module ---
    distill_module = UniteDistillModule(
        model=model,
        teacher_conv=teacher_conv,
        token_mse_weight=token_mse_weight,
        base_lr=base_lr,
        final_lr=final_lr,
        warmup_epochs=1,
        decay_end_epoch=max_epochs,
    )

    # --- Datamodule (S2RGB override) ---
    print('Building S2RGB datamodule...')
    datamodule = make_rgb_datamodule(config)

    # --- Setup ---
    if debug:
        loggers = []
        callbacks = []
        trainer_kwargs = dict(
            accelerator='cpu', devices=1,
            limit_train_batches=2, limit_val_batches=2, max_epochs=1,
        )
    else:
        assert save_dir is not None
        loggers = [
            CSVLogger(save_dir=save_dir),
            WandbLogger(
                name=config['experiment']['experiment_name'],
                save_dir=save_dir,
                project=config['wandb']['project'],
                entity=config['wandb']['entity'],
                mode=config.get('wandb', {}).get('mode', 'online'),
            ),
        ]
        img_logger = ImageLogger(max_images=4, save_dir=save_dir)
        ckpt_cb = ModelCheckpoint(
            dirpath=save_dir, save_top_k=1,
            monitor='val/total_loss', mode='min', save_last=True,
        )
        callbacks = [ckpt_cb, img_logger]
        trainer_cfg = config.get('trainer', {})
        trainer_kwargs = dict(
            accelerator='gpu',
            precision=trainer_cfg.get('precision', 'bf16-mixed'),
            devices=trainer_cfg.get('devices', [0]),
            max_epochs=max_epochs,
            limit_train_batches=trainer_cfg.get('limit_train_batches', 1.0),
            limit_val_batches=trainer_cfg.get('limit_val_batches', 1.0),
            log_every_n_steps=trainer_cfg.get('log_every_n_steps', 50),
        )

    trainer = L.Trainer(
        callbacks=callbacks,
        logger=loggers,
        **trainer_kwargs,
    )

    if not debug:
        with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
            OmegaConf.save(config=config, f=f)

    trainer.fit(distill_module, datamodule=datamodule)

    # --- Save IO checkpoint ---
    out_path = os.path.join(save_dir if save_dir else '.', 'distilled_io.pt')
    save_distilled_io_checkpoint(distill_module, out_path)

    return out_path


def main():
    parser = argparse.ArgumentParser(description='EO-Unite IO Layer Weight Distillation')
    parser.add_argument('--config', type=str, required=True,
                        help='Any eo-unite config (e.g. configs/eo-unite-panopticon.yaml)')
    parser.add_argument('--unite-ckpt', type=str, required=True,
                        help='Pretrained UNITE .pt checkpoint (e.g. UNITE-B.pt)')
    parser.add_argument('--max-epochs', type=int, default=10)
    parser.add_argument('--token-mse-weight', type=float, default=1.0)
    parser.add_argument('--base-lr', type=float, default=1e-4)
    parser.add_argument('--final-lr', type=float, default=1e-5)
    parser.add_argument('--debug', action='store_true',
                        help='CPU mode, no logging, 2 steps')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    if not args.debug:
        exp_name = (
            f'{config["experiment"]["experiment_name"]}_distill'
            f'_{datetime.now().strftime("%m-%d-%Y_%H-%M-%S")}'
        )
        save_dir = os.path.join(config['experiment']['exp_dir'], exp_name)
        os.makedirs(save_dir, exist_ok=True)
        config['experiment']['experiment_name'] = exp_name
        config['experiment']['save_dir'] = save_dir
        config['trainer']['default_root_dir'] = save_dir
    else:
        save_dir = None

    out_path = run_distillation(
        config=config,
        unite_ckpt=args.unite_ckpt,
        max_epochs=args.max_epochs,
        token_mse_weight=args.token_mse_weight,
        base_lr=args.base_lr,
        final_lr=args.final_lr,
        save_dir=save_dir,
        debug=args.debug,
    )

    if not args.debug:
        print(f'\nDistilled IO checkpoint: {out_path}')
        print('\nNext steps:')
        print(f'  python train_unite.py --config {args.config} \\')
        print(f'    --ckpt {args.unite_ckpt} \\')
        print(f'    --distilled-io-ckpt {out_path}')


if __name__ == '__main__':
    main()
