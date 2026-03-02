"""Latent Encoding Script for Sen12MS-CR Cloud Removal Training.

Encodes three modalities for each patch:
  - s2:        cloud-free Sentinel-2  (target)
  - s2_cloudy: cloudy Sentinel-2      (condition 1)
  - s1:        Sentinel-1 SAR         (condition 2)

Output layout:
  {output_root}/
    latent_stats.json
    model_config.yaml
    train/ {patch_id}.npz ...
    val/   {patch_id}.npz ...
    test/  {patch_id}.npz ...

Each .npz contains float16 arrays:
  s2:        [32, H/8, W/8]  -- raw latent (unnormalised)
  s2_cloudy: [32, H/8, W/8]
  s1:        [32, H/8, W/8]
"""

import argparse
import json
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from eo_vae.datasets.sen12ms_cr_dataset import SEN12MSCRDataModule
from eo_vae.utils.latent_encoding import RunningStatsButFast, encode_raw, load_eo_vae

# S2 L1C band wavelengths (µm): B01–B12
WVS_S2 = torch.tensor(
    [0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.945, 1.375, 1.610, 2.190]
)

# S1 C-band VV/VH wavelengths (µm)
WVS_S1 = torch.tensor([5.4, 5.6])


def encode_split(model, dataloader, output_dir, device, stats_s2, stats_s2c, stats_s1, split_name):
    """Encode one split and write per-patch .npz files.

    Args:
        model: EO-VAE model (eval mode).
        dataloader: DataLoader yielding dicts with 's1', 's2', 's2_cloudy', 'patch_id'.
        output_dir: Directory to write .npz files into.
        device: Torch device.
        stats_s2: RunningStatsButFast for cloud-free S2 latents.
        stats_s2c: RunningStatsButFast for cloudy S2 latents.
        stats_s1: RunningStatsButFast for S1 latents.
        split_name: Human-readable name for progress display.
    """
    os.makedirs(output_dir, exist_ok=True)
    wvs_s2 = WVS_S2.to(device)
    wvs_s1 = WVS_S1.to(device)

    print(f'\nEncoding {split_name} split -> {output_dir}')

    for batch in tqdm(dataloader, desc=split_name):
        s2 = batch['s2'].to(device)          # [B, 13, H, W]
        s2c = batch['s2_cloudy'].to(device)  # [B, 13, H, W]
        s1 = batch['s1'].to(device)          # [B, 2,  H, W]
        patch_ids = batch['patch_id']        # list[str]

        with torch.no_grad():
            z_s2 = encode_raw(model, s2, wvs_s2)    # [B, 32, H/8, W/8]
            z_s2c = encode_raw(model, s2c, wvs_s2)  # [B, 32, H/8, W/8]
            z_s1 = encode_raw(model, s1, wvs_s1)    # [B, 32, H/8, W/8]

        stats_s2(z_s2)
        stats_s2c(z_s2c)
        stats_s1(z_s1)

        for i, patch_id in enumerate(patch_ids):
            np.savez_compressed(
                os.path.join(output_dir, f'{patch_id}.npz'),
                s2=z_s2[i].cpu().to(torch.float16).numpy(),
                s2_cloudy=z_s2c[i].cpu().to(torch.float16).numpy(),
                s1=z_s1[i].cpu().to(torch.float16).numpy(),
            )


def main():
    parser = argparse.ArgumentParser(description='Encode Sen12MS-CR dataset to latent space')
    parser.add_argument('--sen12ms_root', type=str, required=True, help='Path to Sen12MS-CR root')
    parser.add_argument('--config', type=str, required=True, help='Path to EO-VAE config')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to EO-VAE checkpoint')
    parser.add_argument('--output_root', type=str, required=True, help='Root folder for output')
    parser.add_argument('--target_size', type=int, default=256, help='Resize images to N×N')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--region', type=str, default='all')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ------------------------------------------------------------------
    # 1. Dataset
    # ------------------------------------------------------------------
    print('\nSetting up SEN12MSCRDataModule...')
    dm = SEN12MSCRDataModule(
        root=args.sen12ms_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        region=args.region,
        normalize=True,
        target_size=(args.target_size, args.target_size),
    )
    dm.setup()

    # ------------------------------------------------------------------
    # 2. Model
    # ------------------------------------------------------------------
    model = load_eo_vae(args.config, args.ckpt, device)

    # ------------------------------------------------------------------
    # 3. Running statistics (accumulated over all splits)
    # ------------------------------------------------------------------
    stats_s2 = RunningStatsButFast((32,), dims=[0, 2, 3]).to(device)
    stats_s2c = RunningStatsButFast((32,), dims=[0, 2, 3]).to(device)
    stats_s1 = RunningStatsButFast((32,), dims=[0, 2, 3]).to(device)

    # ------------------------------------------------------------------
    # 4. Encode all splits
    # ------------------------------------------------------------------
    os.makedirs(args.output_root, exist_ok=True)

    splits = {
        'train': dm.train_dataloader(),
        'val': dm.val_dataloader(),
        'test': dm.test_dataloader(),
    }
    for split_name, loader in splits.items():
        out_dir = os.path.join(args.output_root, split_name)
        encode_split(
            model=model,
            dataloader=loader,
            output_dir=out_dir,
            device=device,
            stats_s2=stats_s2,
            stats_s2c=stats_s2c,
            stats_s1=stats_s1,
            split_name=split_name,
        )

    # ------------------------------------------------------------------
    # 5. Save statistics
    # ------------------------------------------------------------------
    s2_stats = stats_s2.get_stats_dict()
    s2c_stats = stats_s2c.get_stats_dict()
    s1_stats = stats_s1.get_stats_dict()

    print('\n' + '=' * 60)
    print('LATENT STATISTICS (accumulated over all splits)')
    print('=' * 60)
    for name, st in [('s2', s2_stats), ('s2_cloudy', s2c_stats), ('s1', s1_stats)]:
        print(f'\n{name}:')
        print(f'  mean range: [{st["mean"].min():.4f}, {st["mean"].max():.4f}]')
        print(f'  std  range: [{st["std"].min():.4f},  {st["std"].max():.4f}]')

    stats_path = os.path.join(args.output_root, 'latent_stats.json')
    stats_to_save = {
        's2':        {k: v.tolist() for k, v in s2_stats.items()},
        's2_cloudy': {k: v.tolist() for k, v in s2c_stats.items()},
        's1':        {k: v.tolist() for k, v in s1_stats.items()},
    }
    with open(stats_path, 'w') as f:
        json.dump(stats_to_save, f, indent=4)
    print(f'\nSaved latent statistics to {stats_path}')

    conf_out_path = os.path.join(args.output_root, 'model_config.yaml')
    OmegaConf.save(OmegaConf.load(args.config), conf_out_path)
    print(f'Saved model config to {conf_out_path}')

    print('\n' + '=' * 60)
    print('ENCODING COMPLETE')
    print(f'Latents saved to: {args.output_root}/')
    print('=' * 60)


if __name__ == '__main__':
    main()
