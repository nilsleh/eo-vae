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
from lightning import LightningModule
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import OmegaConf
from torchmetrics import AveragePrecision

OmegaConf.register_new_resolver('eval', eval)


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

    # Build model — pick up num_classes and feat_dim from datamodule
    num_classes = config['lightning_module'].get('num_classes', None) or datamodule.num_classes
    config['lightning_module']['num_classes'] = num_classes
    config['lightning_module']['feat_dim'] = datamodule.feat_dim
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
