"""Sen12MS-CR Latent Dataset and DataModule for cloud removal diffusion training.

Loads pre-encoded .npz latent files produced by encode_latents_sen12ms.py.
Each .npz contains float16 arrays: 's2', 's2_cloudy', 's1' of shape [32, H, W].

Returned batch keys (required by DiffusionSuperRes):
  image_hr: [32, H, W]  -- cloud-free S2 latent (target)
  image_lr: [64, H, W]  -- cat([z_s2_cloudy, z_s1]) (conditions)
  wvs:      [13]        -- S2 wavelengths (for VAE decoding at eval)
"""

import json
import os
from glob import glob

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

# S2 L1C band wavelengths (µm): B01–B12
S2_WAVELENGTHS = [0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.945, 1.375, 1.610, 2.190]


class SEN12MSCRLatentDataset(Dataset):
    """Dataset of pre-encoded Sen12MS-CR latents.

    Args:
        root: Root directory containing split subdirs and latent_stats.json.
        split: One of 'train', 'val', 'test'.
        normalize: If True, z-score each modality with its own per-channel stats.
    """

    def __init__(self, root: str, split: str, normalize: bool = True):
        assert split in ('train', 'val', 'test'), f'Invalid split: {split}'
        self.root = root
        self.split = split
        self.normalize = normalize

        split_dir = os.path.join(root, split)
        self.files = sorted(glob(os.path.join(split_dir, '*.npz')))
        if not self.files:
            import warnings
            warnings.warn(f'No .npz files found in {split_dir}')

        # Load normalization stats
        stats_path = os.path.join(root, 'latent_stats.json')
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f'Latent stats not found: {stats_path}')

        with open(stats_path) as f:
            stats = json.load(f)

        self.s2_mean = torch.tensor(stats['s2']['mean'], dtype=torch.float32).view(-1, 1, 1)
        self.s2_std = torch.tensor(stats['s2']['std'], dtype=torch.float32).view(-1, 1, 1)
        self.s2c_mean = torch.tensor(stats['s2_cloudy']['mean'], dtype=torch.float32).view(-1, 1, 1)
        self.s2c_std = torch.tensor(stats['s2_cloudy']['std'], dtype=torch.float32).view(-1, 1, 1)
        self.s1_mean = torch.tensor(stats['s1']['mean'], dtype=torch.float32).view(-1, 1, 1)
        self.s1_std = torch.tensor(stats['s1']['std'], dtype=torch.float32).view(-1, 1, 1)

        self.wvs = torch.tensor(S2_WAVELENGTHS, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        with np.load(self.files[idx]) as data:
            z_s2 = torch.from_numpy(data['s2'].astype(np.float32))
            z_s2c = torch.from_numpy(data['s2_cloudy'].astype(np.float32))
            z_s1 = torch.from_numpy(data['s1'].astype(np.float32))

        if self.normalize:
            z_s2 = (z_s2 - self.s2_mean) / self.s2_std
            z_s2c = (z_s2c - self.s2c_mean) / self.s2c_std
            z_s1 = (z_s1 - self.s1_mean) / self.s1_std

        # Concatenate conditions channel-wise: [32 cloudy_s2 + 32 s1] = [64, H, W]
        cond = torch.cat([z_s2c, z_s1], dim=0)

        return {
            'image_hr': z_s2,    # [32, H, W] -- target
            'image_lr': cond,    # [64, H, W] -- conditions
            'wvs': self.wvs,     # [13]
        }


class SEN12MSCRLatentDataModule(LightningDataModule):
    """LightningDataModule for pre-encoded Sen12MS-CR latents.

    Args:
        root: Root directory with latent_stats.json and train/val/test subdirs.
        batch_size: Training batch size.
        eval_batch_size: Eval batch size (defaults to batch_size * 2).
        num_workers: Number of DataLoader workers.
        normalize: Whether to z-score normalize latents.
    """

    def __init__(
        self,
        root: str,
        batch_size: int = 16,
        eval_batch_size: int = None,
        num_workers: int = 4,
        normalize: bool = True,
    ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size * 2
        self.num_workers = num_workers
        self.normalize = normalize

    def setup(self, stage=None):
        self.train_dataset = SEN12MSCRLatentDataset(self.root, 'train', normalize=self.normalize)
        self.val_dataset = SEN12MSCRLatentDataset(self.root, 'val', normalize=self.normalize)
        self.test_dataset = SEN12MSCRLatentDataset(self.root, 'test', normalize=self.normalize)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
