#!/usr/bin/env python
# coding: utf-8

import argparse
import os
from datetime import datetime
from typing import Any

import torch
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import OmegaConf


def create_experiment_dir(config: dict[str, Any]) -> str:
    """Create experiment directory.

    Args:
        config: config file

    Returns:
        config with updated save_dir
    """
    os.makedirs(config["experiment"]["exp_dir"], exist_ok=True)
    exp_dir_name = (
        f"{config['experiment']['experiment_name']}"
        f"_{config['seg_method']['_target_'].split('.')[-1]}"
        f"_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S-%f')}"
    )
    config["experiment"]["experiment_name"] = exp_dir_name
    exp_dir_path = os.path.join(config["experiment"]["exp_dir"], exp_dir_name)
    os.makedirs(exp_dir_path)
    config["experiment"]["save_dir"] = exp_dir_path
    config["trainer"]["default_root_dir"] = exp_dir_path
    return config


def run_experiment(config) -> None:
    """Run experiment.

    Args:
        config: config file
    """
    torch.set_float32_matmul_precision("medium")
    model = instantiate(config.seg_method)

    datamodule = instantiate(config.datamodule)

    loggers = [
        CSVLogger(config["experiment"]["save_dir"], name="csv_logs"),
        WandbLogger(
            name=config["experiment"]["experiment_name"],
            save_dir=config["experiment"]["save_dir"],
            project=config["wandb"]["project"],
            entity=config["wandb"]["entity"],
            resume="allow",
            mode=config["wandb"]["mode"],
        ),
    ]

    track_metric = "val_loss"
    mode = "min"

    checkpoint_callback = ModelCheckpoint(
        dirpath=config["experiment"]["save_dir"],
        save_top_k=1,
        monitor=track_metric,
        mode=mode,
        save_last=True,
        every_n_epochs=1,
    )

    trainer = instantiate(
        config.trainer, callbacks=[checkpoint_callback], logger=loggers
    )

    # save configuration file
    with open(os.path.join(config["experiment"]["save_dir"], "config.yaml"), "w") as f:
        OmegaConf.save(config=config, f=f)

    # fit model
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )

    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config = create_experiment_dir(config)

    run_experiment(config)
