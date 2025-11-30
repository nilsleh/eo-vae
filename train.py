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

# from eo_vae.utils.callbacks import ImageLogger # Uncomment if available

OmegaConf.register_new_resolver('eval', eval)


def load_distilled_weights(model, ckpt_path):
    """Loads weights from the distillation phase into the model.
    Strict=False is used because the distilled checkpoint might contain
    'teacher' buffers that are not needed/present in finetune mode.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Distilled checkpoint not found at {ckpt_path}')

    print(f'--- Loading Distilled Hypernetworks from {ckpt_path} ---')

    # Load PL Checkpoint
    checkpoint = torch.load(ckpt_path, map_location=model.device)

    # Extract State Dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Load
    keys = model.load_state_dict(state_dict, strict=False)
    print(
        f'Distilled weights loaded. Missing keys: {len(keys.missing_keys)}, Unexpected keys: {len(keys.unexpected_keys)}'
    )
    print('Ready for Finetuning.')


def create_experiment_dir(config: dict[str, Any]) -> str:
    # ... (Same as your original code) ...
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


def run_experiment(config) -> None:
    torch.set_float32_matmul_precision('medium')

    # 1. Instantiate Model (Random Init or Standard Init)
    print('Instantiating Model Architecture...')
    model = instantiate(config.model)

    # 2. [NEW] Load Distilled Checkpoint if provided
    # You can add a field 'distilled_ckpt' to your config file
    distilled_path = config.get('distilled_ckpt', None)

    if distilled_path:
        # Force mode to finetune just in case config had it set to distill
        model.training_mode = 'finetune'
        load_distilled_weights(model, distilled_path)
    else:
        print(
            'No distilled checkpoint provided. Starting from scratch/random initialization.'
        )

    # 3. Instantiate Data
    datamodule = instantiate(config.datamodule)

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

    checkpoint_callback = ModelCheckpoint(
        dirpath=config['experiment']['save_dir'],
        save_top_k=1,
        monitor='val/loss_total',
        mode='min',
        save_last=True,
        every_n_epochs=1,
    )

    # image_logger = ImageLogger(rgb_indices=[0, 1, 2], num_images=4)

    trainer = instantiate(
        config.trainer, callbacks=[checkpoint_callback], logger=loggers
    )

    with open(os.path.join(config['experiment']['save_dir'], 'config.yaml'), 'w') as f:
        OmegaConf.save(config=config, f=f)

    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, required=True, help='Path to the config file'
    )

    # Allow overriding distilled path via CLI
    parser.add_argument(
        '--distilled-ckpt', type=str, default=None, help='Path to distilled checkpoint'
    )

    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    config = create_experiment_dir(config)

    run_experiment(config)
