#!/usr/bin/env python
"""Training script for EO-UNITE.

Usage:
    python train_unite.py --config configs/eo-unite.yaml
    python train_unite.py --config configs/eo-unite.yaml --ckpt path/to/unite.pt
    python train_unite.py --config configs/eo-unite.yaml --resume path/to/last.ckpt
    python train_unite.py --config configs/eo-unite.yaml --debug
"""

import argparse
import os
from datetime import datetime
from typing import Any

import torch
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import OmegaConf

from eo_vae.utils.image_logger import ImageLogger

OmegaConf.register_new_resolver('eval', eval)


def create_experiment_dir(config: dict[str, Any]) -> dict:
    os.makedirs(config['experiment']['exp_dir'], exist_ok=True)
    exp_dir_name = (
        f'{config["experiment"]["experiment_name"]}'
        f'_{datetime.now().strftime("%m-%d-%Y_%H-%M-%S-%f")}'
    )
    config['experiment']['experiment_name'] = exp_dir_name
    exp_dir_path = os.path.join(config['experiment']['exp_dir'], exp_dir_name)
    os.makedirs(exp_dir_path)
    config['experiment']['save_dir'] = exp_dir_path
    config['trainer']['default_root_dir'] = exp_dir_path
    return config


def run_experiment(config, unite_ckpt: str | None = None, resume_ckpt: str | None = None,
                   debug: bool = False) -> None:
    torch.set_float32_matmul_precision('medium')

    print('Instantiating loss function...')
    loss_fn = instantiate(config.model.loss_fn)

    print('Instantiating EOUnite model...')
    model = instantiate(config.model, loss_fn=loss_fn)

    # Load UNITE pretrained weights (partial load; skips new EO modules)
    if unite_ckpt is not None:
        print(f'Loading UNITE checkpoint: {unite_ckpt}')
        model._load_checkpoint(unite_ckpt)

    print('Instantiating datamodule...')
    datamodule = instantiate(config.datamodule)

    if debug:
        loggers = []
        callbacks = []
    else:
        save_dir = config['experiment']['save_dir']
        loggers = [
            CSVLogger(save_dir=save_dir),
            WandbLogger(
                name=config['experiment']['experiment_name'],
                save_dir=save_dir,
                project=config['wandb']['project'],
                entity=config['wandb']['entity'],
                resume='allow',
                mode=config['wandb']['mode'],
            ),
        ]
        img_logger = ImageLogger(max_images=8, save_dir=save_dir)
        checkpoint_callback = ModelCheckpoint(
            dirpath=save_dir,
            save_top_k=2,
            monitor='val/rec_loss',
            mode='min',
            save_last=True,
            every_n_epochs=1,
        )
        callbacks = [checkpoint_callback, img_logger]

        with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
            OmegaConf.save(config=config, f=f)

    trainer_kwargs = dict(callbacks=callbacks, logger=loggers)
    if debug:
        trainer_kwargs['accelerator'] = 'cpu'
        trainer_kwargs['devices'] = 1
        trainer_kwargs['limit_train_batches'] = 2
        trainer_kwargs['limit_val_batches'] = 2
        trainer_kwargs['max_epochs'] = 1
    trainer = instantiate(config.trainer, **trainer_kwargs)
    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train EO-UNITE')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default=None,
                        help='UNITE pretrained .pt checkpoint for partial weight loading')
    parser.add_argument('--resume', type=str, default=None,
                        help='Full .ckpt to resume training from')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode: no logging, run on CPU')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    if not args.debug:
        config = create_experiment_dir(config)

    run_experiment(config, unite_ckpt=args.ckpt, resume_ckpt=args.resume, debug=args.debug)
