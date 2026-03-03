"""Geobench Latent Dataset and DataModule for linear probing.

Loads pre-encoded spatial latents produced by encode_latents_geobench.py.
Each .npz contains float16 arrays:
  feature: [32, H/8, W/8]  -- raw spatial VAE latent (unnormalised)
  label:   [num_classes]    -- multi-hot label vector

Pooling is applied in __getitem__ so different strategies can be tested without
re-encoding. Default: global average pool → [32].
"""

import json
import os
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class GeobenchLatentDataset(Dataset):
    """Dataset of pre-encoded geobench spatial latents.

    Args:
        root: Root directory containing split subdirs and latent_stats.json.
        split: One of 'train', 'val', 'test'.
        normalize: If True, z-score normalise latents with per-channel stats.
        pool: Pooling strategy applied after normalisation.
            'avg' (default): global average pool → [32]
            'max': global max pool → [32]
            'flatten': no pooling, return [32, H, W]
    """

    def __init__(
        self,
        root: str,
        split: str,
        normalize: bool = True,
        pool: str = 'avg',
    ):
        assert split in ('train', 'val', 'test'), f'Invalid split: {split}'
        assert pool in ('avg', 'max', 'flatten'), f'Invalid pool: {pool}'
        self.normalize = normalize
        self.pool = pool

        split_dir = os.path.join(root, split)
        self.files = sorted(glob(os.path.join(split_dir, '*.npz')))
        if not self.files:
            import warnings
            warnings.warn(f'No .npz files found in {split_dir}')

        stats_path = os.path.join(root, 'latent_stats.json')
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f'Latent stats not found: {stats_path}')

        with open(stats_path) as f:
            stats = json.load(f)

        # Per-channel stats over spatial dims; shape [32, 1, 1] for broadcasting
        self.mean = torch.tensor(stats['mean'], dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(stats['std'], dtype=torch.float32).view(-1, 1, 1)

        # Infer num_classes from first sample
        with np.load(self.files[0]) as d:
            self.num_classes = d['label'].shape[0]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        with np.load(self.files[idx]) as data:
            feature = torch.from_numpy(data['feature'].astype(np.float32))  # [32, H, W]
            label = torch.from_numpy(data['label'].astype(np.float32))      # [num_classes]

        if self.normalize:
            feature = (feature - self.mean) / self.std

        if self.pool == 'avg':
            feature = F.adaptive_avg_pool2d(feature.unsqueeze(0), 1).squeeze()  # [32]
        elif self.pool == 'max':
            feature = F.adaptive_max_pool2d(feature.unsqueeze(0), 1).squeeze()  # [32]
        # 'flatten': keep [32, H, W]

        return {'feature': feature, 'label': label}


class GeobenchLatentDataModule(LightningDataModule):
    """LightningDataModule for pre-encoded geobench spatial latents.

    Args:
        root: Root directory with latent_stats.json and train/val/test subdirs.
        batch_size: Training batch size.
        eval_batch_size: Eval batch size (defaults to batch_size * 2).
        num_workers: Number of DataLoader workers.
        normalize: Whether to z-score normalise latents.
        pool: Pooling strategy; see GeobenchLatentDataset.
    """

    def __init__(
        self,
        root: str,
        batch_size: int = 256,
        eval_batch_size: int = None,
        num_workers: int = 4,
        normalize: bool = True,
        pool: str = 'avg',
    ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size * 2
        self.num_workers = num_workers
        self.normalize = normalize
        self.pool = pool

    def setup(self, stage=None):
        self.train_dataset = GeobenchLatentDataset(
            self.root, 'train', normalize=self.normalize, pool=self.pool
        )
        self.val_dataset = GeobenchLatentDataset(
            self.root, 'val', normalize=self.normalize, pool=self.pool
        )
        self.test_dataset = GeobenchLatentDataset(
            self.root, 'test', normalize=self.normalize, pool=self.pool
        )
        self.num_classes = self.train_dataset.num_classes

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
