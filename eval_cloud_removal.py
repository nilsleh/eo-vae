"""Evaluation script for Sen12MS-CR cloud removal diffusion model.

Loads pre-encoded latents, runs diffusion sampling, decodes predictions
to pixel space, and computes RMSE / PSNR / SSIM / SAM over all 13 S2 bands.

Usage:
    python eval_cloud_removal.py \
        --config configs_superres/sen12ms_cr_latent.yaml \
        --ckpt /path/to/diffusion.ckpt \
        --vae_ckpt /path/to/vae.ckpt \
        [--gpu 0] [--batch_size 8] [--output_dir results/cloud-removal-metrics]
"""

import argparse
import json
import os

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torchmetrics.functional import (
    mean_squared_error,
    peak_signal_noise_ratio,
    spectral_angle_mapper,
    structural_similarity_index_measure,
)
from tqdm import tqdm

from eo_vae.datasets.sen12ms_cr_dataset import SEN12MSCRS2Norm

# Image-space normalization stats — pulled directly from the dataset normalizer
# (TerraMesh NORM_STATS_LEGACY['S2L1C'], 13 bands, clip [0,10000] then z-score)
_s2_norm = SEN12MSCRS2Norm()
S2_MEAN = _s2_norm.mean.view(1, 13, 1, 1)  # [1, 13, 1, 1]
S2_STD = _s2_norm.std.view(1, 13, 1, 1)    # [1, 13, 1, 1]


def load_diffusion_model(conf, ckpt_path, device):
    model = instantiate(conf.lightning_module)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()
    return model


def load_vae(conf, vae_ckpt, device):
    vae = instantiate(conf.autoencoder, freeze_body=True)
    if vae_ckpt:
        ckpt = torch.load(vae_ckpt, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)
        vae.load_state_dict(state_dict, strict=False)
    vae.to(device).eval()
    return vae


@torch.no_grad()
def decode_latent(vae, z_norm, dm, wvs, device):
    """Denormalise latent then decode to pixel space (z-scored S2)."""
    mean = dm.train_dataset.s2_mean.to(device)
    std = dm.train_dataset.s2_std.to(device)
    z_raw = z_norm * std + mean
    return vae.decoder(z_raw, wvs)   # [B, 13, H, W] z-scored


def to_physical(img, device):
    """Convert z-scored S2 back to [0,1] by undoing z-score then dividing by 10000."""
    mean = S2_MEAN.to(device)
    std = S2_STD.to(device)
    phys = img * std + mean        # [0, 10000] range
    return torch.clamp(phys / 10000.0, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default=None, help='Diffusion model checkpoint')
    parser.add_argument('--vae_ckpt', type=str, default=None, help='EO-VAE checkpoint')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output_dir', type=str, default='results/cloud-removal-metrics')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    conf = OmegaConf.load(args.config)
    OmegaConf.register_new_resolver('eval', eval, replace=True)

    # Add vae_ckpt to conf so load_vae can fall back to conf.vae_ckpt if desired
    if args.vae_ckpt:
        OmegaConf.update(conf, 'vae_ckpt', args.vae_ckpt, merge=True)

    # ------------------------------------------------------------------
    # DataModule
    # ------------------------------------------------------------------
    print('Setting up datamodule...')
    dm = instantiate(conf.datamodule, batch_size=args.batch_size)
    dm.setup('test')
    dataloader = dm.test_dataloader()

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    print('Loading diffusion model...')
    diff_model = load_diffusion_model(conf, args.ckpt, device)

    print('Loading EO-VAE decoder...')
    vae = load_vae(conf, args.vae_ckpt, device)

    wvs_s2 = torch.tensor(
        [0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.945, 1.375, 1.610, 2.190],
        device=device,
    )

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    metrics = {'RMSE': [], 'PSNR': [], 'SSIM': [], 'SAM': []}

    print('Running evaluation...')
    with torch.no_grad():
        for batch in tqdm(dataloader):
            hr = batch['image_hr'].to(device)   # [B, 32, H, W] normalised target latent
            cond = batch['image_lr'].to(device)  # [B, 64, H, W] normalised condition

            # 1. Sample from diffusion model
            pred_norm = diff_model.sample(x1_shape=hr.shape, cond=cond)  # [B, 32, H, W]

            # 2. Decode prediction and GT to z-scored pixel space
            pred_pixels = decode_latent(vae, pred_norm, dm, wvs_s2, device)  # [B, 13, H, W]
            gt_pixels = decode_latent(vae, hr, dm, wvs_s2, device)            # [B, 13, H, W]

            # 3. Convert to physical [0, 1]
            pred_phys = to_physical(pred_pixels, device)
            gt_phys = to_physical(gt_pixels, device)

            pred_phys = pred_phys.contiguous()
            gt_phys = gt_phys.contiguous()

            metrics['RMSE'].append(
                mean_squared_error(pred_phys, gt_phys, squared=False).item()
            )
            metrics['PSNR'].append(
                peak_signal_noise_ratio(pred_phys, gt_phys, data_range=1.0).item()
            )
            metrics['SSIM'].append(
                structural_similarity_index_measure(pred_phys, gt_phys, data_range=1.0).item()
            )
            metrics['SAM'].append(
                spectral_angle_mapper(pred_phys, gt_phys).item()
            )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    final = {k: float(np.mean(v)) for k, v in metrics.items()}
    print('\n--- Cloud Removal Metrics ---')
    for k, v in final.items():
        print(f'  {k}: {v:.4f}')

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'metrics_cloud_removal.json')
    with open(out_path, 'w') as f:
        json.dump(final, f, indent=4)
    print(f'\nSaved to {out_path}')


if __name__ == '__main__':
    main()
