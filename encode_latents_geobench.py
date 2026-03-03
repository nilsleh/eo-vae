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

# BigEarthNetV2 S2 12-band wavelengths (µm): B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12
WVS_BEN = torch.tensor(
    [0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.945, 1.610, 2.190]
)

# TreeSatAI RGBN wavelengths (µm): R, G, B, NIR
WVS_TREESAT = torch.tensor([0.665, 0.560, 0.490, 0.842])

DATASET_WVS = {
    'ben': WVS_BEN,
    'treesat': WVS_TREESAT,
}


def get_geobench_dataloaders(dataset_name, batch_size, num_workers):
    """Build train/val/test dataloaders from geobench_v2.

    Args:
        dataset_name: 'ben' or 'treesat'.
        batch_size: Batch size.
        num_workers: DataLoader workers.

    Returns:
        dict with 'train', 'val', 'test' DataLoaders.
    """
    import geobench  # noqa: PLC0415

    if dataset_name == 'ben':
        task = next(geobench.task_iterator(benchmark_name='classification_v1', dataset_name='bigearthnet'))
    elif dataset_name == 'treesat':
        task = next(geobench.task_iterator(benchmark_name='classification_v1', dataset_name='treesat_aerial'))
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}. Choose ben or treesat.')

    loaders = {}
    for split in ('train', 'valid', 'test'):
        dataset = task.get_dataset(split=split)
        out_key = 'val' if split == 'valid' else split
        loaders[out_key] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    return loaders


def extract_image_and_label(batch, dataset_name):
    """Extract image tensor [B, C, H, W] and multi-hot label [B, num_classes] from geobench batch.

    Handles both Sample-list and dict-style batches returned by geobench.

    Returns:
        image: FloatTensor [B, C, H, W]
        label: FloatTensor [B, num_classes]
    """
    if isinstance(batch, dict):
        img = batch.get('image', batch.get('x'))
        label = batch.get('label', batch.get('y'))
        img = img.float() if not img.is_floating_point() else img
        label = label.float() if not label.is_floating_point() else label
        return img, label

    # geobench_v2 Sample objects
    images, labels = [], []
    for sample in batch:
        bands = np.stack([b.data.squeeze() for b in sample.bands], axis=0)
        images.append(bands)
        lbl = sample.label
        if hasattr(lbl, 'data'):
            lbl = lbl.data
        labels.append(np.array(lbl, dtype=np.float32))

    image = torch.from_numpy(np.stack(images, axis=0)).float()
    label = torch.from_numpy(np.stack(labels, axis=0)).float()
    return image, label


def encode_split(model, dataloader, wvs, output_dir, device, stats, split_name, dataset_name):
    """Encode one split and write per-sample .npz files.

    Args:
        model: EO-VAE model (eval mode).
        dataloader: DataLoader yielding geobench samples.
        wvs: Wavelength tensor for this dataset.
        output_dir: Directory to write .npz files into.
        device: Torch device.
        stats: RunningStatsButFast for spatial latents [B, 32, H, W].
        split_name: Human-readable name for progress display.
        dataset_name: 'ben' or 'treesat'.
    """
    os.makedirs(output_dir, exist_ok=True)
    wvs = wvs.to(device)
    idx = 0

    print(f'\nEncoding {split_name} split -> {output_dir}')

    for batch in tqdm(dataloader, desc=split_name):
        img, label = extract_image_and_label(batch, dataset_name)
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
    parser = argparse.ArgumentParser(description='Encode geobench dataset to spatial VAE latents')
    parser.add_argument('--dataset', type=str, required=True, choices=['ben', 'treesat'],
                        help='Dataset: ben (BigEarthNetV2) or treesat (TreeSatAI)')
    parser.add_argument('--config', type=str, required=True, help='Path to EO-VAE config')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to EO-VAE checkpoint')
    parser.add_argument('--output_root', type=str, required=True,
                        help='Root folder for output latents')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device string, e.g. cuda or cuda:0 or cpu')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    wvs = DATASET_WVS[args.dataset]

    # ------------------------------------------------------------------
    # 1. Model
    # ------------------------------------------------------------------
    model = load_eo_vae(args.config, args.ckpt, device)

    # ------------------------------------------------------------------
    # 2. Data
    # ------------------------------------------------------------------
    print(f'\nLoading geobench dataset: {args.dataset}')
    loaders = get_geobench_dataloaders(
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
    os.makedirs(args.output_root, exist_ok=True)

    for split_name, loader in loaders.items():
        out_dir = os.path.join(args.output_root, split_name)
        encode_split(
            model=model,
            dataloader=loader,
            wvs=wvs,
            output_dir=out_dir,
            device=device,
            stats=stats,
            split_name=split_name,
            dataset_name=args.dataset,
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

    stats_path = os.path.join(args.output_root, 'latent_stats.json')
    with open(stats_path, 'w') as f:
        json.dump({k: v.tolist() for k, v in feat_stats.items()}, f, indent=4)
    print(f'\nSaved latent statistics to {stats_path}')

    print('\n' + '=' * 60)
    print('ENCODING COMPLETE')
    print(f'Latents saved to: {args.output_root}/')
    print('=' * 60)


if __name__ == '__main__':
    main()
