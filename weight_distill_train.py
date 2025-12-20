#!/usr/bin/env python

import argparse
import os
from datetime import datetime
from typing import Any

import torch
from hydra.utils import instantiate
from lightning import LightningDataModule, Trainer
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset


# --- 1. Dummy Data Implementation ---
class DummyDataset(Dataset):
    def __init__(self, size=10000):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.zeros(1)  # Return dummy value


class DummyDataModule(LightningDataModule):
    def __init__(self, batch_size=1):
        super().__init__()
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(DummyDataset(), batch_size=self.batch_size)

    def val_dataloader(self):
        # Small validation set to check convergence
        return DataLoader(DummyDataset(size=100), batch_size=self.batch_size)


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


# --- 2. Distillation Logic ---
def run_distillation(config, original_flux_ckpt_path):
    """Runs the anchor distillation to initialize Hypernetworks."""
    print(f'--- Starting Distillation from Teacher: {original_flux_ckpt_path} ---')

    # 1. Force Configuration for Distillation
    # We use the architecture defined in your yaml, but override modes
    config.model.training_mode = 'distill'
    config.model.ckpt_path = original_flux_ckpt_path  # Load Teacher weights

    # 2. Instantiate Model
    print('Instantiating Model...')
    model = instantiate(config.model)

    # 3. Setup Dummy Data
    # We don't need real images, just a loop to trigger training_step
    datamodule = DummyDataModule(batch_size=4)

    loggers = [
        CSVLogger(save_dir=config['experiment']['save_dir'])
        # WandbLogger(
        #     name=config['experiment']['experiment_name'],
        #     save_dir=config['experiment']['save_dir'],
        #     project=config['wandb']['project'],
        #     entity=config['wandb']['entity'],
        #     resume='allow',
        #     mode=config['wandb']['mode'],
        # ),
    ]

    # 5. Trainer for Short Run
    # Distillation is fast. 5-10 epochs of dummy data is usually enough to converge.
    trainer = Trainer(
        max_epochs=5,
        accelerator='gpu',
        devices=1,
        default_root_dir=config['experiment']['save_dir'],
        enable_checkpointing=True,
        logger=loggers,  # We usually don't need wandb for this quick step
        check_val_every_n_epoch=1,
    )

    # save config file to save_dir
    config_save_path = os.path.join(config['experiment']['save_dir'], 'config.yaml')
    with open(config_save_path, 'w') as f:
        OmegaConf.save(config, f)
    # 6. Run
    trainer.fit(model, datamodule=datamodule)

    # 7. Save the Final Model explicitly
    final_path = os.path.join(
        config['experiment']['save_dir'], 'distilled_anchors.ckpt'
    )
    trainer.save_checkpoint(final_path)
    print(f'SUCCESS: Distilled checkpoint saved to: {final_path}')
    print('You can now use this path in your main training config.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, required=True, help='Path to your standard model config'
    )
    parser.add_argument(
        '--teacher-ckpt',
        type=str,
        required=True,
        help='Path to original FLUX/SD safetensors',
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config = create_experiment_dir(config)

    run_distillation(config, args.teacher_ckpt)
