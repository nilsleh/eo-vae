#!/usr/bin/env python
"""EO-VAE Weight Distillation Script

This script distills RGB weights from a pretrained Flux VAE checkpoint into
the dynamic convolution layers of the EO-VAE. This is the first step before
multi-modality finetuning.

Usage:
    python distill_train.py --config configs/finetune.yaml --teacher-ckpt path/to/ae.safetensors
    
    # With custom distillation settings
    python distill_train.py --config configs/finetune.yaml --teacher-ckpt path/to/ae.safetensors \
        --max-steps 5000 --lr 1e-4 --val-every 500

The script:
1. Instantiates encoder/decoder from your config (with dynamic layers)
2. Extracts conv_in/conv_out weights from the teacher checkpoint
3. Trains dynamic layers to reproduce those weights for RGB wavelengths
4. Saves a checkpoint that can be loaded for finetuning
"""

import argparse
import os
from dataclasses import dataclass
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from lightning import LightningModule, Trainer
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf
from safetensors import safe_open
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class DistillConfig:
    """Configuration for distillation training."""

    # Training
    max_steps: int = 5000
    lr: float = 1e-4
    val_every_n_steps: int = 500
    log_every_n_steps: int = 50

    # Early stopping
    patience: int = 10
    min_delta: float = 1e-7

    # RGB wavelengths (microns) - R, G, B ordering to match Flux
    rgb_wavelengths: tuple[float, ...] = (0.665, 0.560, 0.490)

    # Loss weights
    weight_loss_scale: float = 1.0
    bias_loss_scale: float = 1.0


# =============================================================================
# TEACHER WEIGHT EXTRACTION
# =============================================================================


def load_teacher_weights(ckpt_path: str) -> dict[str, Tensor]:
    """Extract conv_in and conv_out weights from a pretrained Flux checkpoint.

    Args:
        ckpt_path: Path to .safetensors or .ckpt/.pt file

    Returns:
        Dictionary with encoder/decoder weights and biases
    """
    print(f'Loading teacher weights from: {ckpt_path}')

    if ckpt_path.endswith('.safetensors'):
        sd = {}
        with safe_open(ckpt_path, framework='pt', device='cpu') as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
    else:
        sd = torch.load(ckpt_path, map_location='cpu')
        sd = sd.get('state_dict', sd)

    # Find encoder conv_in weights
    enc_weight, enc_bias = None, None
    for key_pattern in ['encoder.conv_in.weight', 'conv_in.weight']:
        if key_pattern in sd:
            enc_weight = sd[key_pattern]
            break
    for key_pattern in ['encoder.conv_in.bias', 'conv_in.bias']:
        if key_pattern in sd:
            enc_bias = sd[key_pattern]
            break

    # Find decoder conv_out weights
    dec_weight, dec_bias = None, None
    for key_pattern in ['decoder.conv_out.weight', 'conv_out.weight']:
        if key_pattern in sd:
            dec_weight = sd[key_pattern]
            break
    for key_pattern in ['decoder.conv_out.bias', 'conv_out.bias']:
        if key_pattern in sd:
            dec_bias = sd[key_pattern]
            break

    if enc_weight is None:
        raise ValueError(
            f'Could not find encoder conv_in weights in checkpoint.\n'
            f"Available keys containing 'conv': {[k for k in sd.keys() if 'conv' in k.lower()]}"
        )
    if dec_weight is None:
        raise ValueError(
            f'Could not find decoder conv_out weights in checkpoint.\n'
            f"Available keys containing 'conv': {[k for k in sd.keys() if 'conv' in k.lower()]}"
        )

    print(f'  Encoder conv_in weight shape: {enc_weight.shape}')
    print(
        f'  Encoder conv_in bias shape: {enc_bias.shape if enc_bias is not None else "None"}'
    )
    print(f'  Decoder conv_out weight shape: {dec_weight.shape}')
    print(
        f'  Decoder conv_out bias shape: {dec_bias.shape if dec_bias is not None else "None"}'
    )

    return {
        'encoder_weight': enc_weight,
        'encoder_bias': enc_bias,
        'decoder_weight': dec_weight,
        'decoder_bias': dec_bias,
    }


# =============================================================================
# DISTILLATION LIGHTNING MODULE
# =============================================================================


class DistillationModule(LightningModule):
    """Lightning module for distilling RGB weights into dynamic layers.

    This module:
    1. Takes dynamic encoder/decoder conv layers
    2. Compares their generated weights (for RGB wavelengths) to teacher weights
    3. Optimizes until the dynamic layers reproduce the teacher weights
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        teacher_weights: dict[str, Tensor],
        config: DistillConfig,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config

        # Register teacher weights as buffers (not parameters)
        self.register_buffer('teacher_enc_w', teacher_weights['encoder_weight'])
        self.register_buffer('teacher_dec_w', teacher_weights['decoder_weight'])

        if teacher_weights['encoder_bias'] is not None:
            self.register_buffer('teacher_enc_b', teacher_weights['encoder_bias'])
        else:
            self.teacher_enc_b = None

        if teacher_weights['decoder_bias'] is not None:
            self.register_buffer('teacher_dec_b', teacher_weights['decoder_bias'])
        else:
            self.teacher_dec_b = None

        # RGB wavelengths as buffer
        self.register_buffer(
            'rgb_wvs', torch.tensor(config.rgb_wavelengths, dtype=torch.float32)
        )

        # Track best loss for logging
        self.best_loss = float('inf')

        self.save_hyperparameters(ignore=['encoder', 'decoder', 'teacher_weights'])

    def compute_distillation_loss(self) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute loss between generated weights and teacher weights.

        Returns:
            Tuple of (total_loss, log_dict)
        """
        logs = {}
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # --- Encoder Distillation ---
        if self.encoder.use_dynamic_ops:
            conv_in = self.encoder.conv_in
            if not hasattr(conv_in, 'get_distillation_weight'):
                raise AttributeError(
                    f'Encoder conv_in ({type(conv_in).__name__}) does not have '
                    f"'get_distillation_weight' method. This method is required for distillation."
                )

            # Get dynamic weights for RGB wavelengths
            student_enc_w, student_enc_b = conv_in.get_distillation_weight(self.rgb_wvs)

            # Weight loss
            loss_enc_w = F.mse_loss(student_enc_w, self.teacher_enc_w)
            total_loss = total_loss + loss_enc_w * self.config.weight_loss_scale
            logs['enc_weight_loss'] = loss_enc_w

            # Bias loss (if available)
            if student_enc_b is not None and self.teacher_enc_b is not None:
                loss_enc_b = F.mse_loss(student_enc_b, self.teacher_enc_b)
                total_loss = total_loss + loss_enc_b * self.config.bias_loss_scale
                logs['enc_bias_loss'] = loss_enc_b

            # Log weight statistics for debugging
            with torch.no_grad():
                logs['enc_weight_mae'] = F.l1_loss(student_enc_w, self.teacher_enc_w)
                logs['enc_weight_max_err'] = (
                    (student_enc_w - self.teacher_enc_w).abs().max()
                )

        # --- Decoder Distillation ---
        if self.decoder.use_dynamic_ops:
            conv_out = self.decoder.conv_out
            if not hasattr(conv_out, 'get_distillation_weight'):
                raise AttributeError(
                    f'Decoder conv_out ({type(conv_out).__name__}) does not have '
                    f"'get_distillation_weight' method. This method is required for distillation. "
                    f"Make sure you're using an updated version of the multi-stage decoder modules."
                )

            # Get dynamic weights for RGB wavelengths
            student_dec_w, student_dec_b = conv_out.get_distillation_weight(
                self.rgb_wvs
            )

            # Weight loss
            loss_dec_w = F.mse_loss(student_dec_w, self.teacher_dec_w)
            total_loss = total_loss + loss_dec_w * self.config.weight_loss_scale
            logs['dec_weight_loss'] = loss_dec_w

            # Bias loss (if available)
            if student_dec_b is not None and self.teacher_dec_b is not None:
                loss_dec_b = F.mse_loss(student_dec_b, self.teacher_dec_b)
                total_loss = total_loss + loss_dec_b * self.config.bias_loss_scale
                logs['dec_bias_loss'] = loss_dec_b

            # Log weight statistics for debugging
            with torch.no_grad():
                logs['dec_weight_mae'] = F.l1_loss(student_dec_w, self.teacher_dec_w)
                logs['dec_weight_max_err'] = (
                    (student_dec_w - self.teacher_dec_w).abs().max()
                )

        logs['total_loss'] = total_loss

        return total_loss, logs

    def training_step(self, batch, batch_idx):
        """Training step - batch is dummy, not used."""
        loss, logs = self.compute_distillation_loss()

        # Log with train/ prefix
        self.log_dict(
            {f'train/{k}': v for k, v in logs.items()},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        loss, logs = self.compute_distillation_loss()

        # Track best loss
        if loss < self.best_loss:
            self.best_loss = loss.item()

        # Log with val/ prefix
        self.log_dict(
            {f'val/{k}': v for k, v in logs.items()},
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def configure_optimizers(self):
        """Configure optimizer for dynamic layer parameters only."""
        params = []

        if self.encoder.use_dynamic_ops:
            params.extend(self.encoder.conv_in.parameters())

        if self.decoder.use_dynamic_ops:
            params.extend(self.decoder.conv_out.parameters())

        if not params:
            raise ValueError(
                'No dynamic layers found! Make sure use_dynamic_ops=True '
                'in encoder and/or decoder config.'
            )

        optimizer = torch.optim.AdamW(params, lr=self.config.lr, weight_decay=1e-5)

        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.max_steps, eta_min=self.config.lr * 0.01
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'},
        }


# =============================================================================
# DUMMY DATASET (Distillation doesn't need real data)
# =============================================================================


class DummyDataset(Dataset):
    """Dummy dataset - distillation doesn't use actual data."""

    def __init__(self, length: int = 10000):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {'dummy': torch.tensor(0)}


class DummyDataModule:
    """Simple data module for distillation."""

    def __init__(self, train_size: int = 10000, val_size: int = 100):
        self.train_size = train_size
        self.val_size = val_size

    def train_dataloader(self):
        return DataLoader(DummyDataset(self.train_size), batch_size=1, num_workers=0)

    def val_dataloader(self):
        return DataLoader(DummyDataset(self.val_size), batch_size=1, num_workers=0)


# =============================================================================
# EXPERIMENT SETUP
# =============================================================================


def create_experiment_dir(config: DictConfig, suffix: str = 'distill') -> DictConfig:
    """Create experiment directory and update config paths."""
    os.makedirs(config['experiment']['exp_dir'], exist_ok=True)

    exp_name = (
        f'{config["experiment"]["experiment_name"]}'
        f'_{suffix}'
        f'_{datetime.now().strftime("%m-%d-%Y_%H-%M-%S")}'
    )

    exp_dir = os.path.join(config['experiment']['exp_dir'], exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Update config
    config['experiment']['experiment_name'] = exp_name
    config['experiment']['save_dir'] = exp_dir

    print(f'Experiment directory: {exp_dir}')

    return config


def save_distillation_checkpoint(
    module: DistillationModule, save_path: str, config: DictConfig
) -> None:
    """Save distillation results in a format that can be loaded for finetuning.

    The checkpoint includes:
    - Dynamic layer state dicts
    - Config used for distillation
    - Teacher weight shapes (for verification)
    """
    checkpoint = {
        # Dynamic layer weights
        'encoder_conv_in_state_dict': (
            module.encoder.conv_in.state_dict()
            if module.encoder.use_dynamic_ops
            else None
        ),
        'decoder_conv_out_state_dict': (
            module.decoder.conv_out.state_dict()
            if module.decoder.use_dynamic_ops
            else None
        ),
        # Full encoder/decoder state dicts (for convenience)
        'encoder_state_dict': module.encoder.state_dict(),
        'decoder_state_dict': module.decoder.state_dict(),
        # Metadata
        'distill_config': {
            'rgb_wavelengths': module.config.rgb_wavelengths,
            'final_loss': module.best_loss,
            'max_steps': module.config.max_steps,
            'lr': module.config.lr,
        },
        'teacher_shapes': {
            'encoder_weight': tuple(module.teacher_enc_w.shape),
            'decoder_weight': tuple(module.teacher_dec_w.shape),
        },
        # Model config for reconstruction
        'model_config': OmegaConf.to_container(config.model, resolve=True),
    }

    torch.save(checkpoint, save_path)
    print(f'Distillation checkpoint saved to: {save_path}')


def load_distilled_checkpoint(
    encoder: nn.Module, decoder: nn.Module, ckpt_path: str, strict: bool = True
) -> dict:
    """Load distilled weights into encoder/decoder.

    Args:
        encoder: Encoder module with dynamic conv_in
        decoder: Decoder module with dynamic conv_out
        ckpt_path: Path to distillation checkpoint
        strict: Whether to require all keys to match

    Returns:
        Checkpoint metadata dict
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')

    # Load encoder dynamic layer
    if encoder.use_dynamic_ops and ckpt.get('encoder_conv_in_state_dict'):
        missing, unexpected = encoder.conv_in.load_state_dict(
            ckpt['encoder_conv_in_state_dict'], strict=strict
        )
        print(
            f'Loaded encoder.conv_in: {len(missing)} missing, {len(unexpected)} unexpected'
        )

    # Load decoder dynamic layer
    if decoder.use_dynamic_ops and ckpt.get('decoder_conv_out_state_dict'):
        missing, unexpected = decoder.conv_out.load_state_dict(
            ckpt['decoder_conv_out_state_dict'], strict=strict
        )
        print(
            f'Loaded decoder.conv_out: {len(missing)} missing, {len(unexpected)} unexpected'
        )

    print(f'Distillation loss was: {ckpt["distill_config"]["final_loss"]:.6f}')

    return ckpt


# =============================================================================
# MAIN DISTILLATION FUNCTION
# =============================================================================


def run_distillation(
    config: DictConfig, teacher_ckpt_path: str, distill_config: DistillConfig
) -> str:
    """Run the distillation training process.

    Args:
        config: Hydra config with model definitions
        teacher_ckpt_path: Path to pretrained Flux checkpoint
        distill_config: Distillation-specific configuration

    Returns:
        Path to saved distillation checkpoint
    """
    print('=' * 70)
    print('EO-VAE Weight Distillation')
    print('=' * 70)

    # --- 1. Load Teacher Weights ---
    teacher_weights = load_teacher_weights(teacher_ckpt_path)

    # --- 2. Instantiate Encoder & Decoder ---
    print('\nInstantiating encoder and decoder from config...')
    encoder = instantiate(config.model.encoder)
    decoder = instantiate(config.model.decoder)

    # Verify dynamic ops are enabled
    if not encoder.use_dynamic_ops:
        print('WARNING: Encoder does not use dynamic ops, nothing to distill!')
    if not decoder.use_dynamic_ops:
        print('WARNING: Decoder does not use dynamic ops, nothing to distill!')

    if not encoder.use_dynamic_ops and not decoder.use_dynamic_ops:
        raise ValueError(
            'Neither encoder nor decoder uses dynamic ops. Nothing to distill!'
        )

    print(f'  Encoder dynamic ops: {encoder.use_dynamic_ops}')
    print(f'  Decoder dynamic ops: {decoder.use_dynamic_ops}')

    # Print decoder head type if using multi-stage
    if hasattr(decoder, 'decoder_head_type'):
        print(f'  Decoder head type: {decoder.decoder_head_type}')

    # --- 3. Create Distillation Module ---
    module = DistillationModule(
        encoder=encoder,
        decoder=decoder,
        teacher_weights=teacher_weights,
        config=distill_config,
    )

    # --- 4. Setup Data ---
    datamodule = DummyDataModule(train_size=distill_config.max_steps, val_size=100)

    # --- 6. Setup Loggers ---
    loggers = [
        CSVLogger(save_dir=config['experiment']['save_dir'], name='distill_logs')
    ]

    # --- 7. Setup Trainer ---
    trainer = Trainer(
        max_steps=distill_config.max_steps,
        accelerator='auto',
        devices=1,
        precision='32-true',  # Full precision for distillation accuracy
        default_root_dir=config['experiment']['save_dir'],
        logger=loggers,
        val_check_interval=distill_config.val_every_n_steps,
        log_every_n_steps=distill_config.log_every_n_steps,
        enable_progress_bar=True,
    )

    # --- 8. Save Config ---
    config_path = os.path.join(config['experiment']['save_dir'], 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)
    print(f'\nConfig saved to: {config_path}')

    # --- 9. Run Distillation ---
    print('\nStarting distillation training...')
    trainer.fit(
        module,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
    )

    # --- 10. Save Final Checkpoint ---
    final_path = os.path.join(config['experiment']['save_dir'], 'distilled_final.pt')
    save_distillation_checkpoint(module, final_path, config)

    # Also save a Lightning checkpoint for compatibility
    trainer.save_checkpoint(
        os.path.join(config['experiment']['save_dir'], 'distilled_lightning.ckpt')
    )

    print('\n' + '=' * 70)
    print('Distillation Complete!')
    print('=' * 70)
    print(f'Final loss: {module.best_loss:.6f}')
    print(f'Checkpoint saved to: {final_path}')
    print('\nTo use in finetuning, load with:')
    print(f"  load_distilled_checkpoint(encoder, decoder, '{final_path}')")

    return final_path


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description='EO-VAE Weight Distillation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to model config YAML (e.g., finetune_consistency.yaml)',
    )
    parser.add_argument(
        '--teacher-ckpt',
        type=str,
        required=True,
        help='Path to pretrained Flux VAE checkpoint (.safetensors or .ckpt)',
    )

    # Distillation settings
    parser.add_argument(
        '--max-steps', type=int, default=5000, help='Maximum training steps'
    )
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument(
        '--val-every', type=int, default=500, help='Validation frequency (steps)'
    )
    parser.add_argument(
        '--patience', type=int, default=10, help='Early stopping patience'
    )

    # Output settings
    parser.add_argument(
        '--exp-name', type=str, default=None, help='Override experiment name'
    )

    args = parser.parse_args()

    # Load config
    config = OmegaConf.load(args.config)

    # Override experiment name if provided
    if args.exp_name:
        config['experiment']['experiment_name'] = args.exp_name

    # Create experiment directory
    config = create_experiment_dir(config, suffix='distill')

    # Create distillation config
    distill_config = DistillConfig(
        max_steps=args.max_steps,
        lr=args.lr,
        val_every_n_steps=args.val_every,
        patience=args.patience,
    )

    # Run distillation
    run_distillation(config, args.teacher_ckpt, distill_config)


if __name__ == '__main__':
    main()
