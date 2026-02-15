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

from eo_vae.utils.image_logger import ImageLogger  # Uncomment if available

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


def load_vae_weights_for_refinement(model, ckpt_path):
    """Loads VAE weights from a checkpoint into the model, ignoring the refiner.
    This is used for Phase 3 (Flow Refinement) where we start with a pre-trained VAE
    and train a fresh refiner on top.
    """
    print(f'--- Loading VAE Weights for Refinement from {ckpt_path} ---')
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # Filter out any keys that belong to the refiner (just in case the ckpt has them)
    # and ensure we only load matching keys for the VAE parts.
    model_state = model.state_dict()
    filtered_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith('refiner'):
            continue
        if k in model_state and model_state[k].shape == v.shape:
            filtered_state_dict[k] = v

    keys = model.load_state_dict(filtered_state_dict, strict=False)

    # Verify that critical VAE parts are loaded
    missing_vae_keys = [k for k in keys.missing_keys if not k.startswith('refiner')]
    if len(missing_vae_keys) > 0:
        print(
            f'Warning: {len(missing_vae_keys)} VAE keys missing (e.g. {missing_vae_keys[:5]}).'
        )
    else:
        print('Successfully loaded VAE backbone.')


def run_experiment(config, distilled_ckpt, vae_ckpt=None, debug: bool = False) -> None:
    torch.set_float32_matmul_precision('medium')

    # 1. Instantiate Components Individually
    print('Instantiating Encoder & Decoder...')
    encoder = instantiate(config.model.encoder)
    decoder = instantiate(config.model.decoder)

    # 2. Load Distilled Weights (Component-wise) - ONLY for Phase 2 (Finetuning)
    # If we are in Phase 3 (Flow Refine), we load the full VAE later.
    if distilled_ckpt is not None and vae_ckpt is None:
        print(f'--- Loading Distilled Weights from {distilled_ckpt} ---')
        checkpoint = torch.load(distilled_ckpt, map_location='cpu')
        state_dict = (
            checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        )

        def load_component(component, prefix, full_state_dict):
            # Filter keys for this component and strip the prefix (e.g. "encoder.")
            comp_dict = {
                k[len(prefix) :]: v
                for k, v in full_state_dict.items()
                if k.startswith(prefix)
            }
            if comp_dict:
                keys = component.load_state_dict(comp_dict, strict=False)
                print(
                    f'Loaded {prefix[:-1]}: Missing {len(keys.missing_keys)}, Unexpected {len(keys.unexpected_keys)}'
                )
            else:
                print(f'Warning: No weights found for {prefix[:-1]}')

        load_component(encoder, 'encoder.', state_dict)
        load_component(decoder, 'decoder.', state_dict)
    elif vae_ckpt is None:
        print(
            'No distilled checkpoint provided. Starting from scratch/random initialization (unless VAE ckpt provided later).'
        )

    # 3. Instantiate Loss & Discriminator (Transferring Weights)
    print('Instantiating Loss Function...')
    loss_cfg = config.model.loss_fn

    # Check if we need to inject the dynamic input layer into the discriminator
    if 'discriminator' in loss_cfg and hasattr(encoder, 'conv_in'):
        # We need to instantiate the discriminator with the correct input channels
        # This is a bit of a hack, but it works for now
        print('Injecting dynamic input layer into discriminator...')
        loss_fn = instantiate(
            loss_cfg, discriminator={'input_conv_generator': encoder.conv_in}
        )
    else:
        loss_fn = instantiate(loss_cfg)

    # 4. Instantiate Full Model
    # We pass the pre-instantiated objects as kwargs, which overrides the config definitions
    print('Instantiating FluxAutoencoderKL...')

    model = instantiate(config.model, encoder=encoder, decoder=decoder, loss_fn=loss_fn)

    if distilled_ckpt and not vae_ckpt:
        model.training_mode = 'finetune'

    # Load VAE weights manually if in flow-refine mode
    if vae_ckpt and config.model.training_mode == 'flow-refine':
        load_vae_weights_for_refinement(model, vae_ckpt)

    # 5. Instantiate Data
    datamodule = instantiate(config.datamodule)

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
            monitor='val/loss_rec',
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

    parser.add_argument(
        '--vae-ckpt', type=str, default=None, help='Path to full VAE checkpoint'
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

    run_experiment(
        config, args.distilled_ckpt, vae_ckpt=args.vae_ckpt, debug=args.debug
    )
