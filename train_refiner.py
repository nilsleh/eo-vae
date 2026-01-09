#!/usr/bin/env python

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


def create_experiment_dir(config: dict[str, Any]) -> str:
    """Creates a unique experiment directory based on timestamp."""
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


def load_vae_checkpoint(vae_model, ckpt_path):
    """Loads the FULL state dict into the Base VAE."""
    print(f'--- Loading Base VAE from {ckpt_path} ---')

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # Clean up keys if strictly loading into the model (handling "model." prefix if exists)
    # If your VAE class structure matches the checkpoint structure exactly:
    keys = vae_model.load_state_dict(state_dict, strict=True)

    print(
        f'VAE Weights Loaded: Missing {len(keys.missing_keys)}, Unexpected {len(keys.unexpected_keys)}'
    )
    if len(keys.missing_keys) > 0:
        print(f'Warning: Missing keys: {keys.missing_keys}')


def run_experiment(config, vae_ckpt, debug: bool = False) -> None:
    torch.set_float32_matmul_precision('medium')

    # 1. Instantiate Encoder separately to load weights
    print('Instantiating Encoder...')
    vae = instantiate(config.model.base_vae)

    # 2. Load Pretrained Encoder Weights
    load_vae_checkpoint(vae, vae_ckpt)

    print('Instantiating Refiner...')
    model = instantiate(config.model, base_vae=vae)

    # 4. Instantiate Data
    print('Instantiating DataModule...')
    datamodule = instantiate(config.datamodule)

    # 5. Loggers & Callbacks
    if debug:
        loggers = []
        checkpoint_callback = None
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

        img_logger = ImageLogger(
            max_images=8, save_dir=config['experiment']['save_dir']
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=config['experiment']['save_dir'],
            save_top_k=1,
            monitor='val/mse_refined',
            mode='min',
            save_last=True,
            every_n_epochs=1,
        )

    callbacks = [checkpoint_callback, img_logger] if checkpoint_callback else []

    trainer = instantiate(config.trainer, callbacks=callbacks, logger=loggers)

    if not debug:
        with open(
            os.path.join(config['experiment']['save_dir'], 'config.yaml'), 'w'
        ) as f:
            OmegaConf.save(config=config, f=f)

    print('Starting Training...')
    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, required=True, help='Path to the config file'
    )
    parser.add_argument(
        '--vae-ckpt', type=str, required=True, help='Path to pretrained VAE checkpoint'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode: no logging, no experiment directory',
    )

    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    if not args.debug:
        config = create_experiment_dir(config)

    run_experiment(config, args.vae_ckpt, debug=args.debug)
