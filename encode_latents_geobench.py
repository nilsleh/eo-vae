"""Latent Encoding Script for GeobenchV2 Linear Probing.

Encodes geobench_v2 S2 imagery to spatial latents [B, 32, H/8, W/8].
Pooling is intentionally deferred to the dataset/training stage for flexibility.

Output layout:
  {output_root}/
    latent_stats.json
    train/ {idx:06d}.npz ...
    val/   {idx:06d}.npz ...
    test/  {idx:06d}.npz ...

Each .npz contains float16 arrays:
  feature: [32, H/8, W/8]  -- raw spatial VAE latent (unnormalised)
  label:   [num_classes]    -- multi-hot label vector
"""

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from eo_vae.utils.latent_encoding import RunningStatsButFast, encode_raw, load_eo_vae

# Per-dataset config: wavelengths (µm) and the batch image key emitted by the datamodule
DATASET_CONFIG = {
    'ben': {
        'wvs': torch.tensor(
            [0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.945, 1.610, 2.190]
        ),
        'image_key': 'image_s2',
    },
    'treesat': {
        'wvs': torch.tensor([0.665, 0.560, 0.490, 0.842]),
        'image_key': 'image_aerial',
    },
}


def get_geobench_dataloaders(root, dataset_name, batch_size, num_workers):
    """Build train/val/test dataloaders from geobench_v2.

    Args:
        dataset_name: 'ben' or 'treesat'.
        batch_size: Batch size.
        num_workers: DataLoader workers.

    Returns:
        dict with 'train', 'val', 'test' DataLoaders.
    """
    if dataset_name == 'ben':
        from geobench_v2.datamodules.benv2 import GeoBenchBENV2DataModule  # noqa: PLC0415

        dm = GeoBenchBENV2DataModule(
            root=os.path.join(root, 'benv2'),
            # download=True,
            # band_order=[B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12],
            band_order = {"s2": [
                "B01",
                "B02",
                "B03",
                "B04",
                "B05",
                "B06",
                "B07",
                "B08",
                "B8A",
                "B09",
                "B11",
                "B12",
            ],
            },
            batch_size=batch_size,
            num_workers=num_workers,
        )
    elif dataset_name == 'treesat':
        from geobench_v2.datamodules.treesatai import GeoBenchTreeSatAIDataModule  # noqa: PLC0415

        dm = GeoBenchTreeSatAIDataModule(
            root=os.path.join(root, 'treesatai'),
            band_order={"aerial": ['r', 'g', 'b', 'nir']},
            # download=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}. Choose ben or treesat.')

    dm.setup(stage='fit')
    dm.setup(stage='test')

    return {
        'train': dm.train_dataloader(),
        'val': dm.val_dataloader(),
        'test': dm.test_dataloader(),
    }


def extract_image_and_label(batch, image_key: str):
    """Extract image tensor [B, C, H, W] and multi-hot label [B, num_classes] from a geobench_v2 batch.

    Args:
        batch: Dict returned by the geobench_v2 datamodule.
        image_key: Batch key for the image tensor, e.g. 'image_s2' or 'image_aerial'.

    Returns:
        image: FloatTensor [B, C, H, W]
        label: FloatTensor [B, num_classes]
    """
    img = batch[image_key]
    label = batch['label']

    # resize img to 224x224
    img = torch.nn.functional.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)

    # Ensure label is 2-D [B, num_classes]; single-label datasets return [B]
    if label.dim() == 1:
        label = label.unsqueeze(1)

    return img, label


def encode_split(model, dataloader, wvs, image_key, output_dir, device, stats, split_name):
    """Encode one split and write per-sample .npz files.

    Args:
        model: EO-VAE model (eval mode).
        dataloader: DataLoader yielding geobench_v2 batches.
        wvs: Wavelength tensor for this dataset.
        image_key: Batch key for the image tensor.
        output_dir: Directory to write .npz files into.
        device: Torch device.
        stats: RunningStatsButFast for spatial latents [B, 32, H, W].
        split_name: Human-readable name for progress display.
    """
    os.makedirs(output_dir, exist_ok=True)
    wvs = wvs.to(device)
    idx = 0

    print(f'\nEncoding {split_name} split -> {output_dir}')

    for batch in tqdm(dataloader, desc=split_name):
        img, label = extract_image_and_label(batch, image_key)
        img = img.to(device)   # [B, C, H, W]
        label = label.cpu()    # [B, num_classes]

        with torch.no_grad():
            z = encode_raw(model, img, wvs)  # [B, 32, H/8, W/8]

        stats(z)

        for i in range(z.shape[0]):
            np.savez_compressed(
                os.path.join(output_dir, f'{idx:06d}.npz'),
                feature=z[i].cpu().to(torch.float16).numpy(),   # [32, H/8, W/8]
                label=label[i].numpy().astype(np.float16),       # [num_classes]
            )
            idx += 1

    print(f'  Saved {idx} samples to {output_dir}')


def main():
    parser = argparse.ArgumentParser(description='Encode geobench_v2 dataset to spatial VAE latents')
    parser.add_argument('--dataset', type=str, required=True, choices=['ben', 'treesat'],
                        help='Dataset: ben (BigEarthNetV2) or treesat (TreeSatAI)')
    parser.add_argument('--config', type=str, required=True, help='Path to EO-VAE config')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to EO-VAE checkpoint')
    parser.add_argument('--output_root', type=str, required=True,
                        help='Root folder for output latents')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing the geobench_v2 dataset(s)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device string, e.g. cuda or cuda:0 or cpu')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    cfg = DATASET_CONFIG[args.dataset]
    wvs = cfg['wvs']
    image_key = cfg['image_key']

    # ------------------------------------------------------------------
    # 1. Model
    # ------------------------------------------------------------------
    model = load_eo_vae(args.config, args.ckpt, device)

    # ------------------------------------------------------------------
    # 2. Data
    # ------------------------------------------------------------------
    print(f'\nLoading geobench_v2 dataset: {args.dataset}')
    loaders = get_geobench_dataloaders(
        root=args.data_root,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ------------------------------------------------------------------
    # 3. Running statistics over spatial latents (reduce B, H, W)
    # ------------------------------------------------------------------
    stats = RunningStatsButFast((32,), dims=[0, 2, 3]).to(device)

    # ------------------------------------------------------------------
    # 4. Encode all splits
    # ------------------------------------------------------------------
    output_root = os.path.join(args.output_root, args.dataset)
    os.makedirs(output_root, exist_ok=True)


    for split_name, loader in loaders.items():
        out_dir = os.path.join(output_root, split_name)
        encode_split(
            model=model,
            dataloader=loader,
            wvs=wvs,
            image_key=image_key,
            output_dir=out_dir,
            device=device,
            stats=stats,
            split_name=split_name,
        )

    # ------------------------------------------------------------------
    # 5. Save statistics
    # ------------------------------------------------------------------
    feat_stats = stats.get_stats_dict()

    print('\n' + '=' * 60)
    print('LATENT STATISTICS (spatial, accumulated over all splits)')
    print('=' * 60)
    print(f'  mean range: [{feat_stats["mean"].min():.4f}, {feat_stats["mean"].max():.4f}]')
    print(f'  std  range: [{feat_stats["std"].min():.4f},  {feat_stats["std"].max():.4f}]')

    stats_path = os.path.join(output_root, 'latent_stats.json')
    with open(stats_path, 'w') as f:
        json.dump({k: v.tolist() for k, v in feat_stats.items()}, f, indent=4)
    print(f'\nSaved latent statistics to {stats_path}')

    print('\n' + '=' * 60)
    print('ENCODING COMPLETE')
    print(f'Latents saved to: {output_root}/')
    print('=' * 60)


if __name__ == '__main__':
    main()