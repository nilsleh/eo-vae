#!/usr/bin/env python

import argparse
import os
from datetime import datetime
from typing import Any

import torch
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import OmegaConf

from eo_vae.utils.super_res_image_logger import SuperResImageLogger

OmegaConf.register_new_resolver('eval', eval)


def create_experiment_dir(config: dict[str, Any]) -> str:
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


def run_experiment(config, debug: bool = False) -> None:
    torch.set_float32_matmul_precision('medium')

    model = instantiate(config.lightning_module)
    # 3. Instantiate Data
    if debug:
        config['datamodule']['num_workers'] = 0
    datamodule = instantiate(config.datamodule)

    if debug:
        # in debugging no logging, checkpointing, or experiment dir
        loggers = []
        checkpoint_callback = None

        # Force CPU execution in debug mode
        print('Debug mode enabled: Switching trainer to CPU.')
        config['trainer']['accelerator'] = 'cpu'
        if 'devices' in config['trainer']:
            del config['trainer']['devices']
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

        img_logger = SuperResImageLogger(
            max_images=8, save_dir=config['experiment']['save_dir']
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=config['experiment']['save_dir'],
            save_top_k=1,
            monitor='val_mse',
            mode='min',
            save_last=True,
            every_n_epochs=1,
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')

    callbacks = (
        [checkpoint_callback, img_logger, lr_monitor] if checkpoint_callback else []
    )

    trainer = instantiate(config.trainer, callbacks=callbacks, logger=loggers)

    if not debug:
        # add vae_ckpt to config
        config['vae_ckpt'] = args.ckpt
        with open(
            os.path.join(config['experiment']['save_dir'], 'config.yaml'), 'w'
        ) as f:
            OmegaConf.save(config=config, f=f)

    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, required=True, help='Path to the config file'
    )

    # Allow overriding distilled path via CLI
    parser.add_argument(
        '--ckpt', type=str, default=None, help='Path to distilled checkpoint'
    )

    # Add debug flag
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode: no logging, no experiment directory',
    )

    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    if not args.debug:
        config = create_experiment_dir(config)

    run_experiment(config, debug=args.debug)
