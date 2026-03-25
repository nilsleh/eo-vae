#!/usr/bin/env python
"""Minimal overfit loop for debugging EO-Unite and distillation pipelines.

Bypasses Lightning entirely — no callbacks, no WandB, no LR scheduling.
Caches N batches from the streaming dataset and loops over them, printing
loss per epoch. Useful for verifying the training pipeline converges.

Usage:
    # EO-Unite joint training
    python overfit_debug.py --mode unite \\
        --config configs/eo-unite-panopticon.yaml \\
        --ckpt UNITE-B.pt --n-batches 4 --epochs 50

    # IO layer distillation
    python overfit_debug.py --mode distill \\
        --config configs/eo-unite-panopticon.yaml \\
        --unite-ckpt UNITE-B.pt --n-batches 4 --epochs 50
"""

import argparse
import sys

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

OmegaConf.register_new_resolver('eval', eval)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def run_loop(module, optimizer, loader, device, epochs):
    module.to(device).train()
    n = len(loader)
    for epoch in range(epochs):
        total = 0.0
        for i, batch in enumerate(loader):
            batch = move_batch(batch, device)
            loss = module.training_step(batch, i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f'Epoch {epoch + 1:3d}/{epochs} | loss {total / n:.4f}', flush=True)


def build_unite(config, ckpt, distilled_io_ckpt):
    loss_fn = instantiate(config.model.loss_fn)
    model = instantiate(config.model, loss_fn=loss_fn)
    if ckpt:
        print(f'Loading UNITE checkpoint: {ckpt}')
        model._load_checkpoint(ckpt)
    if distilled_io_ckpt:
        print(f'Loading distilled IO checkpoint: {distilled_io_ckpt}')
        model._load_distilled_io_ckpt(distilled_io_ckpt)
    return model


def build_distill(config, unite_ckpt):
    sys.path.insert(0, '.')
    from weight_distill_unite import UniteDistillModule, load_teacher_patch_proj, make_rgb_datamodule

    loss_fn = instantiate(config.model.loss_fn)
    model = instantiate(config.model, loss_fn=loss_fn)
    print(f'Loading UNITE body from {unite_ckpt}')
    model._load_checkpoint(unite_ckpt)
    model._freeze_body()

    teacher_conv = load_teacher_patch_proj(unite_ckpt)
    distill_module = UniteDistillModule(model=model, teacher_conv=teacher_conv)
    return distill_module


def main():
    parser = argparse.ArgumentParser(description='Overfit debug loop for EO-Unite')
    parser.add_argument('--mode', choices=['unite', 'distill'], required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt', default=None, help='UNITE .pt ckpt (unite mode)')
    parser.add_argument('--unite-ckpt', default=None, help='UNITE .pt ckpt (distill mode)')
    parser.add_argument('--distilled-io-ckpt', default=None)
    parser.add_argument('--n-batches', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    OmegaConf.update(config, 'datamodule.overfit_batches', args.n_batches)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    if args.mode == 'unite':
        module = build_unite(config, args.ckpt, args.distilled_io_ckpt)
        datamodule = instantiate(config.datamodule)
        optimizer = torch.optim.AdamW(
            [p for p in module.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4
        )

    else:  # distill
        if not args.unite_ckpt:
            parser.error('--unite-ckpt is required for distill mode')
        from weight_distill_unite import make_rgb_datamodule
        module = build_distill(config, args.unite_ckpt)
        datamodule = make_rgb_datamodule(config)
        OmegaConf.update(config, 'datamodule.overfit_batches', args.n_batches)
        datamodule.overfit_batches = args.n_batches
        optimizer = torch.optim.AdamW(
            list(module.model.patch_embed.parameters())
            + list(module.model.unpatchify.parameters())
            + list(module.model.modality_conditioner.parameters())
            + list(module.model.cond_proj.parameters()),
            lr=args.lr, weight_decay=1e-4,
        )

    datamodule.setup('fit')
    loader = datamodule.train_dataloader()
    print(f'Cached {len(loader)} batches. Starting {args.epochs} epochs...')

    run_loop(module, optimizer, loader, device, args.epochs)


if __name__ == '__main__':
    main()
