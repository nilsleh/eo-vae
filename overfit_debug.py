#!/usr/bin/env python
"""Minimal overfit loop for debugging EO-Unite and distillation pipelines.

Bypasses Lightning entirely — no callbacks, no WandB, no LR scheduling.
Caches N batches from the streaming dataset and loops over them.

Tracks per-epoch metrics to a CSV and saves input/recon/error image grids
every --vis-every epochs for visual inspection.

Usage:
    # EO-Unite joint training
    python overfit_debug.py --mode unite \\
        --config configs/eo-unite-panopticon.yaml \\
        --ckpt UNITE-B.pt --n-batches 4 --epochs 100

    # IO layer distillation
    python overfit_debug.py --mode distill \\
        --config configs/eo-unite-panopticon.yaml \\
        --unite-ckpt UNITE-B.pt --n-batches 4 --epochs 100
"""

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf

OmegaConf.register_new_resolver('eval', eval)

from eo_vae.utils.image_logger import NORM_STATS_CUSTOM, NORM_STATS_LEGACY, RGB_INDICES


# ---------------------------------------------------------------------------
# Visualization helpers (mirrors ImageLogger logic)
# ---------------------------------------------------------------------------

def _denormalize(x: torch.Tensor, modality: str, norm_scheme: str) -> torch.Tensor:
    if norm_scheme == 'custom' and modality in NORM_STATS_CUSTOM:
        stats = NORM_STATS_CUSTOM[modality]
    elif modality in NORM_STATS_LEGACY:
        stats = NORM_STATS_LEGACY[modality]
    else:
        return x
    mean = torch.tensor(stats['mean'], device=x.device).view(1, -1, 1, 1)
    std = torch.tensor(stats['std'], device=x.device).view(1, -1, 1, 1)
    x_phys = x * std + mean
    if norm_scheme == 'custom' and modality in ('S2L2A', 'S2L1C'):
        x_phys = x_phys.clamp(0.0, 10000.0)
    return x_phys


def _to_vis(x: torch.Tensor, rgb_indices: list) -> torch.Tensor:
    """Robust 2%–98% percentile scaling to [0, 1] for display."""
    x = x[:, rgb_indices]
    vis = []
    for i in range(x.shape[0]):
        img = x[i]
        low = torch.quantile(img.reshape(-1), 0.02)
        high = torch.quantile(img.reshape(-1), 0.98)
        vis.append((img - low).div_(high - low + 1e-5).clamp_(0, 1))
    return torch.stack(vis)


def save_grid(inputs: torch.Tensor, recons: torch.Tensor,
              modality: str, norm_scheme: str, epoch: int, out_dir: str, n: int = 4):
    """Save [input | reconstruction | error] grid as PNG."""
    n = min(n, inputs.shape[0])
    rgb = RGB_INDICES.get(modality, [0, 1, 2])

    inp_phys = _denormalize(inputs[:n].detach().float(), modality, norm_scheme)
    rec_phys = _denormalize(recons[:n].detach().float(), modality, norm_scheme)

    inp_vis = _to_vis(inp_phys, rgb)
    rec_vis = _to_vis(rec_phys, rgb)

    diff = (inp_phys[:, rgb] - rec_phys[:, rgb]).abs().mean(dim=1, keepdim=True)
    diff_vis = (diff - diff.min()) / (diff.max() - diff.min() + 1e-5)
    diff_vis = diff_vis.repeat(1, 3, 1, 1)

    rows = [torch.cat([inp_vis[i], rec_vis[i], diff_vis[i]], dim=2) for i in range(n)]
    grid = torch.cat(rows, dim=1).permute(1, 2, 0).cpu().numpy()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'epoch_{epoch:04d}_{modality}.png')
    plt.figure(figsize=(12, n * 3))
    plt.imshow(grid)
    plt.axis('off')
    plt.title(f'Epoch {epoch} | {modality} | Input  /  Reconstruction  /  |Error|')
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f'  [vis] {path}')


# ---------------------------------------------------------------------------
# Per-step forward functions (return loss + full logs dict)
# ---------------------------------------------------------------------------

def _unite_step(module, batch: dict, global_step: int):
    images = batch['image']
    wvs = module._prepare_wvs(batch['wvs'], device=images.device, dtype=images.dtype)
    geo = module._build_geo(batch, images.device, images.dtype)
    time = batch.get('time')
    if time is not None:
        time = time.float()
    cond = module._compute_cond(
        wvs=wvs, batch_size=images.shape[0],
        device=images.device, dtype=images.dtype,
        geo=geo, time=time,
    )
    z = module.encode(images, wvs, cond=cond)
    z_normed = module.encoder_ln(z)
    z_for_decode = module._noising(z_normed) if module.training else z_normed
    recon = module.decode(z_for_decode, wvs)
    loss, logs = module.loss_fn(
        recon=recon, target=images, wvs=wvs,
        modality=batch.get('modality'),
        z=z_normed, transport=module.transport,
        encoder=module.encoder, encoder_ln=module.encoder_ln,
        latent_tokens=module.latent_tokens, cond=cond,
        global_step=global_step, split='train', alibi_bias=module.alibi,
    )
    return loss, {k: v.item() for k, v in logs.items()}, recon.detach(), images


def _distill_step(module, batch: dict, global_step: int):
    recon, student_tokens, teacher_tokens = module._forward(batch)
    images = batch['image']
    recon_loss = F.l1_loss(recon, images)
    token_mse = F.mse_loss(student_tokens, teacher_tokens)
    total = recon_loss + module.token_mse_weight * token_mse
    logs = {
        'train/recon_loss': recon_loss.item(),
        'train/token_mse': token_mse.item(),
        'train/total_loss': total.item(),
    }
    return total, logs, recon.detach(), images


def _recon_nograd(module, batch: dict, mode: str) -> torch.Tensor:
    """Reconstruction without gradient for visualization."""
    with torch.no_grad():
        if mode == 'unite':
            return module(batch)        # EOUnite.forward accepts a batch dict
        else:
            images = batch['image']
            wvs = module.rgb_wvs.to(images.device)
            return module._reconstruct(images, wvs, batch)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_loop(module, optimizer, loader, device, epochs, mode,
             vis_every, out_dir, norm_scheme):
    module.to(device).train()
    batches = list(loader)   # already cached by CachedBatchDataset
    step_fn = _unite_step if mode == 'unite' else _distill_step

    csv_path = os.path.join(out_dir, 'metrics.csv')
    csv_file = open(csv_path, 'w', newline='')
    writer = None
    global_step = 0

    for epoch in range(1, epochs + 1):
        epoch_logs: dict[str, list] = {}

        for batch in batches:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            loss, logs, last_recon, last_images = step_fn(module, batch, global_step)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            for k, v in logs.items():
                epoch_logs.setdefault(k, []).append(v)

        means = {k: sum(v) / len(v) for k, v in epoch_logs.items()}

        # CSV
        if writer is None:
            writer = csv.DictWriter(csv_file, fieldnames=['epoch'] + sorted(means))
            writer.writeheader()
        writer.writerow({'epoch': epoch, **means})
        csv_file.flush()

        # Console
        parts = '  '.join(f'{k.split("/")[-1]}: {v:.4f}' for k, v in sorted(means.items()))
        print(f'Epoch {epoch:3d}/{epochs} | {parts}', flush=True)

        # Visualize
        if vis_every > 0 and epoch % vis_every == 0:
            module.eval()
            vis_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batches[0].items()}
            modality = vis_batch.get('modality', 'S2RGB')
            if isinstance(modality, (list, tuple)):
                modality = modality[0]
            recon = _recon_nograd(module, vis_batch, mode)
            save_grid(vis_batch['image'], recon, modality, norm_scheme, epoch, out_dir)
            module.train()

    csv_file.close()
    print(f'\nMetrics: {csv_path}')
    print(f'Images:  {out_dir}')


# ---------------------------------------------------------------------------
# Model / datamodule builders
# ---------------------------------------------------------------------------

def _build_unite(config, ckpt, distilled_io_ckpt):
    loss_fn = instantiate(config.model.loss_fn)
    model = instantiate(config.model, loss_fn=loss_fn)
    if ckpt:
        print(f'Loading UNITE checkpoint: {ckpt}')
        model._load_checkpoint(ckpt)
    if distilled_io_ckpt:
        print(f'Loading distilled IO checkpoint: {distilled_io_ckpt}')
        model._load_distilled_io_ckpt(distilled_io_ckpt)
    return model


def _build_distill(config, unite_ckpt):
    from weight_distill_unite import UniteDistillModule, load_teacher_patch_proj
    loss_fn = instantiate(config.model.loss_fn)
    model = instantiate(config.model, loss_fn=loss_fn)
    print(f'Loading UNITE body: {unite_ckpt}')
    model._load_checkpoint(unite_ckpt)
    model._freeze_body()
    teacher_conv = load_teacher_patch_proj(unite_ckpt)
    return UniteDistillModule(model=model, teacher_conv=teacher_conv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Overfit debug loop (no Lightning)')
    parser.add_argument('--mode', choices=['unite', 'distill'], required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt', default=None, help='UNITE .pt ckpt (unite mode)')
    parser.add_argument('--unite-ckpt', default=None, help='UNITE .pt ckpt (distill mode)')
    parser.add_argument('--distilled-io-ckpt', default=None)
    parser.add_argument('--n-batches', type=int, default=4, help='Batches to cache and overfit')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--vis-every', type=int, default=10,
                        help='Save image grid every N epochs (0 = off)')
    parser.add_argument('--out-dir', default='overfit_debug_out',
                        help='Directory for images and metrics CSV')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    OmegaConf.update(config, 'datamodule.overfit_batches', args.n_batches)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  |  mode: {args.mode}  |  batches: {args.n_batches}  |  epochs: {args.epochs}')

    if args.mode == 'unite':
        module = _build_unite(config, args.ckpt, args.distilled_io_ckpt)
        datamodule = instantiate(config.datamodule)
        optimizer = torch.optim.AdamW(
            [p for p in module.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=1e-4,
        )
    else:
        if not args.unite_ckpt:
            parser.error('--unite-ckpt is required for distill mode')
        from weight_distill_unite import make_rgb_datamodule
        module = _build_distill(config, args.unite_ckpt)
        datamodule = make_rgb_datamodule(config)
        datamodule.overfit_batches = args.n_batches
        optimizer = torch.optim.AdamW(
            list(module.model.patch_embed.parameters())
            + list(module.model.unpatchify.parameters())
            + list(module.model.modality_conditioner.parameters())
            + list(module.model.cond_proj.parameters()),
            lr=args.lr, weight_decay=1e-4,
        )

    norm_scheme = getattr(datamodule, 'norm_scheme', 'legacy')
    os.makedirs(args.out_dir, exist_ok=True)
    datamodule.setup('fit')
    loader = datamodule.train_dataloader()
    print(f'Cached {len(loader)} batch(es). Starting training...\n')

    run_loop(
        module=module,
        optimizer=optimizer,
        loader=loader,
        device=device,
        epochs=args.epochs,
        mode=args.mode,
        vis_every=args.vis_every,
        out_dir=args.out_dir,
        norm_scheme=norm_scheme,
    )


if __name__ == '__main__':
    main()
