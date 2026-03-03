#!/usr/bin/env python
"""Linear probe training on pre-encoded geobench latent features.

Usage:
    python train_linear_probe.py --config configs/linear_probe_ben.yaml
    python train_linear_probe.py --config configs/linear_probe_treesat.yaml
    python train_linear_probe.py --config configs/linear_probe_ben.yaml --debug
"""

import argparse
import math
import os
from datetime import datetime
from typing import Any

import torch
import torch.nn as nn
from hydra.utils import instantiate
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import OmegaConf
from torchmetrics import AveragePrecision

OmegaConf.register_new_resolver('eval', eval)


# =============================================================================
# Lightning Module
# =============================================================================


class LinearProbeModule(LightningModule):
    """Single linear layer trained on frozen VAE latent features.

    Args:
        feat_dim: Input feature dimensionality (32 for global-avg-pooled latents).
        num_classes: Number of output classes.
        base_lr: Peak learning rate.
        final_lr: Minimum LR at end of cosine schedule.
        warmup_epochs: Epochs for linear LR warmup.
        max_epochs: Total training epochs (for cosine schedule end).
        weight_decay: AdamW weight decay.
    """

    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        base_lr: float = 1e-3,
        final_lr: float = 1e-5,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.linear = nn.Linear(feat_dim, num_classes)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.val_map = AveragePrecision(task='multilabel', num_labels=num_classes, average='macro')
        self.test_map = AveragePrecision(task='multilabel', num_labels=num_classes, average='macro')

    def forward(self, feature):
        return self.linear(feature)

    def training_step(self, batch, batch_idx):
        logits = self(batch['feature'])
        loss = self.loss_fn(logits, batch['label'])
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch['feature'])
        loss = self.loss_fn(logits, batch['label'])
        preds = torch.sigmoid(logits)
        self.val_map.update(preds, batch['label'].int())
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log('val_mAP', self.val_map.compute(), prog_bar=True)
        self.val_map.reset()

    def test_step(self, batch, batch_idx):
        logits = self(batch['feature'])
        preds = torch.sigmoid(logits)
        self.test_map.update(preds, batch['label'].int())

    def on_test_epoch_end(self):
        self.log('test_mAP', self.test_map.compute())
        self.test_map.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.base_lr,
            weight_decay=self.hparams.weight_decay,
        )

        def lr_lambda(epoch):
            warmup = self.hparams.warmup_epochs
            total = self.hparams.max_epochs
            ratio = self.hparams.final_lr / self.hparams.base_lr

            if epoch < warmup:
                return (epoch + 1) / warmup
            # Cosine decay from base_lr to final_lr
            progress = (epoch - warmup) / max(total - warmup, 1)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return ratio + (1 - ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }


# =============================================================================
# Experiment helpers
# =============================================================================


def create_experiment_dir(config: dict[str, Any]) -> dict[str, Any]:
    os.makedirs(config['experiment']['exp_dir'], exist_ok=True)
    exp_name = (
        f'{config["experiment"]["experiment_name"]}'
        f'_{datetime.now().strftime("%m-%d-%Y_%H-%M-%S-%f")}'
    )
    config['experiment']['experiment_name'] = exp_name
    save_dir = os.path.join(config['experiment']['exp_dir'], exp_name)
    os.makedirs(save_dir)
    config['experiment']['save_dir'] = save_dir
    config['trainer']['default_root_dir'] = save_dir
    return config


def run_experiment(config, debug: bool = False) -> None:
    torch.set_float32_matmul_precision('medium')

    # Datamodule
    if debug:
        config['datamodule']['num_workers'] = 0
    datamodule = instantiate(config.datamodule)
    datamodule.setup()

    # Build model — pick up num_classes from datamodule if not set in config
    num_classes = config['lightning_module'].get('num_classes', None) or datamodule.num_classes
    config['lightning_module']['num_classes'] = num_classes
    config['lightning_module']['max_epochs'] = config['trainer']['max_epochs']
    model = instantiate(config.lightning_module)

    if debug:
        loggers = []
        callbacks = []
        config['trainer']['accelerator'] = 'cpu'
        config['trainer'].pop('devices', None)
        config['trainer']['max_epochs'] = 2
        config['trainer']['limit_train_batches'] = 2
        config['trainer']['limit_val_batches'] = 2
        print('Debug mode: CPU, 2 epochs, no logging.')
    else:
        loggers = [
            CSVLogger(save_dir=config['experiment']['save_dir']),
            WandbLogger(
                name=config['experiment']['experiment_name'],
                save_dir=config['experiment']['save_dir'],
                project=config['wandb']['project'],
                entity=config['wandb']['entity'],
                resume='allow',
                mode=config['wandb']['mode'],
            ),
        ]
        callbacks = [
            ModelCheckpoint(
                dirpath=config['experiment']['save_dir'],
                save_top_k=1,
                monitor='val_mAP',
                mode='max',
                save_last=True,
            ),
            LearningRateMonitor(logging_interval='epoch'),
        ]

        with open(os.path.join(config['experiment']['save_dir'], 'config.yaml'), 'w') as f:
            OmegaConf.save(config=config, f=f)

    trainer = instantiate(config.trainer, callbacks=callbacks, logger=loggers)
    trainer.fit(model, datamodule=datamodule)

    if not debug:
        trainer.test(model, datamodule=datamodule, ckpt_path='best')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    if not args.debug:
        config = create_experiment_dir(config)

    run_experiment(config, debug=args.debug)
