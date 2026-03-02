"""SEN12MS-CR Dataset and DataModule for mono-temporal cloud removal data.

Handles the original SEN12MS-CR directory format:
  {root}/ROIs1158_spring/s1_106/ROIs1158_spring_s1_106_p1.tif
  {root}/ROIs1158_spring/s2_106/ROIs1158_spring_s2_106_p1.tif
  {root}/ROIs1158_spring/s2_cloudy_106/ROIs1158_spring_s2_cloudy_106_p1.tif
"""

import os
import random

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningDataModule
from omegaconf import ListConfig
from torch.utils.data import DataLoader, Dataset


def read_tif(path):
    return rasterio.open(path)


def read_img(tif):
    return tif.read().astype(np.float32)

# =============================================================================
# Constants
# =============================================================================

ROI_SEASON_MAP = {
    'ROIs1158': 'ROIs1158_spring',
    'ROIs1868': 'ROIs1868_summer',
    'ROIs1970': 'ROIs1970_fall',
    'ROIs2017': 'ROIs2017_winter',
}

# All scenes per ROI collection (from SEN12MSCRTS reference)
ALL_ROIS = {
    'ROIs1158': ['106'],
    'ROIs1868': ['17', '36', '56', '73', '85', '100', '114', '119', '121', '126', '127', '139', '142', '143'],
    'ROIs1970': ['20', '21', '35', '40', '57', '65', '71', '82', '83', '91', '112', '116', '119', '128', '132', '133', '135', '139', '142', '144', '149'],
    'ROIs2017': ['8', '22', '25', '32', '49', '61', '63', '69', '75', '103', '108', '115', '116', '117', '130', '140', '146'],
}

# S2 L1C band wavelengths (µm): B01, B02, B03, B04, B05, B06, B07, B08, B08A, B09, B10, B11, B12
S2_WAVELENGTHS = [0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.945, 1.375, 1.610, 2.190]

# S1 C-band VV/VH wavelengths (µm)
S1_WAVELENGTHS = [5.4, 5.6]

# Hardcoded scene-level train/val/test splits (transcribed from SEN12MSCRTS).
# Format: 'ROIsXXXX/scene_id'. For 'all' region, train is computed as all scenes
# not in test or val.
SPLITS = {
    'all': {
        'test': [
            'ROIs1868/119', 'ROIs1970/139', 'ROIs2017/108', 'ROIs2017/63',
            'ROIs1158/106', 'ROIs1868/73', 'ROIs2017/32', 'ROIs1868/100',
            'ROIs1970/132', 'ROIs2017/103', 'ROIs1868/142', 'ROIs1970/20', 'ROIs2017/140',
        ],
        'val': ['ROIs2017/22', 'ROIs1970/65', 'ROIs2017/117', 'ROIs1868/127', 'ROIs1868/17'],
    },
    'africa': {
        'test': ['ROIs2017/32', 'ROIs2017/140'],
        'val': ['ROIs2017/22'],
        'train': ['ROIs1970/21', 'ROIs1970/35', 'ROIs1970/40', 'ROIs2017/8', 'ROIs2017/61', 'ROIs2017/75'],
    },
    'america': {
        'test': ['ROIs1158/106', 'ROIs1970/132'],
        'val': ['ROIs1970/65'],
        'train': ['ROIs1868/36', 'ROIs1868/85', 'ROIs1970/82', 'ROIs1970/142', 'ROIs2017/49', 'ROIs2017/116'],
    },
    'asiaEast': {
        'test': ['ROIs1868/73', 'ROIs1868/119', 'ROIs1970/139'],
        'val': ['ROIs2017/117'],
        'train': ['ROIs1868/114', 'ROIs1868/126', 'ROIs1868/143', 'ROIs1970/116', 'ROIs1970/135', 'ROIs2017/25'],
    },
    'asiaWest': {
        'test': ['ROIs1868/100'],
        'val': ['ROIs1868/127'],
        'train': ['ROIs1970/57', 'ROIs1970/83', 'ROIs1970/112', 'ROIs2017/69', 'ROIs1970/115', 'ROIs1970/130'],
    },
    'europa': {
        'test': ['ROIs2017/63', 'ROIs2017/103', 'ROIs2017/108', 'ROIs1868/142', 'ROIs1970/20'],
        'val': ['ROIs1868/17'],
        'train': [
            'ROIs1868/56', 'ROIs1868/121', 'ROIs1868/139',
            'ROIs1970/71', 'ROIs1970/91', 'ROIs1970/119', 'ROIs1970/128',
            'ROIs1970/133', 'ROIs1970/144', 'ROIs1970/149',
            'ROIs2017/146',
        ],
    },
}

# =============================================================================
# Normalizer modules
# =============================================================================


class SEN12MSCRS2Norm(nn.Module):
    """S2 L1C normalization: clip to [0, 10000] then z-score.

    Statistics from TerraMesh NORM_STATS_LEGACY['S2L1C'] (13 bands).
    """

    def __init__(self):
        super().__init__()
        self.register_buffer(
            'mean',
            torch.tensor([
                2475.625, 2260.839, 2143.561, 2230.225, 2445.427,
                2992.950, 3257.843, 3171.695, 3440.958, 1567.433,
                561.076, 2562.809, 1924.178,
            ]).view(13, 1, 1),
        )
        self.register_buffer(
            'std',
            torch.tensor([
                1761.905, 1804.267, 1661.263, 1932.020, 1918.007,
                1812.421, 1795.179, 1734.280, 1780.039, 1082.531,
                512.077, 1350.580, 1177.511,
            ]).view(13, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, min=0.0, max=10000.0)
        return (x - self.mean) / self.std


class SEN12MSCRS1Norm(nn.Module):
    """S1 SAR normalization: clip to [-25, 0] dB then z-score.

    Statistics from TerraMesh NORM_STATS_LEGACY['S1RTC'] (VV, VH).
    """

    def __init__(self):
        super().__init__()
        self.register_buffer(
            'mean',
            torch.tensor([-10.793, -17.198]).view(2, 1, 1),
        )
        self.register_buffer(
            'std',
            torch.tensor([4.278, 4.346]).view(2, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, min=-25.0, max=0.0)
        return (x - self.mean) / self.std


# =============================================================================
# Augmentation
# =============================================================================


def _apply_augmentation(x: torch.Tensor) -> torch.Tensor:
    """D4 symmetry group augmentation for a single image (C, H, W)."""
    if random.random() > 0.5:
        x = torch.flip(x, dims=[-1])
    if random.random() > 0.5:
        x = torch.flip(x, dims=[-2])
    k = random.randint(0, 3)
    if k > 0:
        x = torch.rot90(x, k, dims=[-2, -1])
    return x


# =============================================================================
# Dataset
# =============================================================================


class SEN12MSCRDataset(Dataset):
    """PyTorch Dataset for the mono-temporal SEN12MS-CR dataset.

    Returns S1, S2 (cloud-free), and S2_cloudy triplets with optional
    normalization, resizing, and D4 augmentation.

    Args:
        root: Path to the SEN12MS-CR root directory.
        split: One of 'train', 'val', 'test', 'all'.
        region: One of 'all', 'africa', 'america', 'asiaEast', 'asiaWest', 'europa'.
        normalize: Whether to apply z-score normalization.
        target_size: Optional (H, W) tuple to resize images to.
        augment: Whether to apply random D4 augmentation.
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        region: str = 'all',
        normalize: bool = True,
        target_size=None,
        augment: bool = False,
    ):
        assert split in ('train', 'val', 'test', 'all'), f'Invalid split: {split}'
        assert region in SPLITS, f'Invalid region: {region}'

        self.root = root
        self.split = split
        self.region = region
        self.normalize = normalize
        self.augment = augment

        if isinstance(target_size, (list, ListConfig)):
            target_size = tuple(target_size)
        self.target_size = target_size

        self.s2_norm = SEN12MSCRS2Norm()
        self.s1_norm = SEN12MSCRS1Norm()

        self._split_rois = self._get_split_rois(split, region)
        self.paths = self._index_paths()

        if not self.paths:
            import warnings
            warnings.warn(
                f'No samples found for split={split}, region={region} in {root}. '
                'Check that the directory structure matches: '
                '{root}/ROIsXXXX_season/s1_NNN/ROIsXXXX_season_s1_NNN_pM.tif'
            )

    def _get_split_rois(self, split: str, region: str) -> list:
        """Return list of 'ROIsXXXX/scene_id' strings for the given split."""
        region_splits = SPLITS[region]

        if split == 'all':
            train = self._compute_train(region_splits)
            return train + region_splits['test'] + region_splits['val']

        if split == 'train':
            return self._compute_train(region_splits)

        return region_splits[split]

    def _compute_train(self, region_splits: dict) -> list:
        if 'train' in region_splits:
            return region_splits['train']
        # For 'all' region: train = every known ROI not in test or val
        all_rois = [
            f'{col}/{scene}'
            for col, scenes in ALL_ROIS.items()
            for scene in scenes
        ]
        excluded = set(region_splits['test'] + region_splits['val'])
        return [r for r in all_rois if r not in excluded]

    def _index_paths(self) -> list:
        """Build list of {'s1', 's2', 's2_cloudy'} path dicts for all patches."""
        paths = []
        for roi_str in self._split_rois:
            roi_col, scene_id = roi_str.split('/')
            season = ROI_SEASON_MAP[roi_col]

            s1_dir = os.path.join(self.root, season, f's1_{scene_id}')
            s2_dir = os.path.join(self.root, season, f's2_{scene_id}')
            s2c_dir = os.path.join(self.root, season, f's2_cloudy_{scene_id}')

            if not os.path.isdir(s1_dir):
                continue

            for fname in sorted(os.listdir(s1_dir)):
                if not fname.endswith('.tif'):
                    continue
                s1_path = os.path.join(s1_dir, fname)
                # Derive paired filenames: ROIsXXXX_season_s1_NNN_pM.tif
                #   → ROIsXXXX_season_s2_NNN_pM.tif
                #   → ROIsXXXX_season_s2_cloudy_NNN_pM.tif
                s2_fname = fname.replace('_s1_', '_s2_')
                s2c_fname = fname.replace('_s1_', '_s2_cloudy_')
                s2_path = os.path.join(s2_dir, s2_fname)
                s2c_path = os.path.join(s2c_dir, s2c_fname)

                if not (os.path.isfile(s2_path) and os.path.isfile(s2c_path)):
                    continue

                # e.g. 'ROIs1158_spring_s1_106_p1.tif' -> 'ROIs1158_spring_106_p1'
                patch_id = fname.replace('_s1_', '_').replace('.tif', '')
                paths.append({
                    's1': s1_path,
                    's2': s2_path,
                    's2_cloudy': s2c_path,
                    'patch_id': patch_id,
                })

        return paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        entry = self.paths[idx]

        s1 = torch.from_numpy(read_img(read_tif(entry['s1'])))        # (2, H, W)
        s2 = torch.from_numpy(read_img(read_tif(entry['s2'])))        # (13, H, W)
        s2_cloudy = torch.from_numpy(read_img(read_tif(entry['s2_cloudy'])))  # (13, H, W)

        if self.normalize:
            s1 = self.s1_norm(s1)
            s2 = self.s2_norm(s2)
            s2_cloudy = self.s2_norm(s2_cloudy)

        if self.target_size is not None:
            s1 = F.interpolate(s1.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)
            s2 = F.interpolate(s2.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)
            s2_cloudy = F.interpolate(s2_cloudy.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)

        if self.augment:
            # Stack all bands, apply identical augmentation, then split
            stacked = torch.cat([s1, s2, s2_cloudy], dim=0)  # (28, H, W)
            stacked = _apply_augmentation(stacked)
            s1, s2, s2_cloudy = stacked[:2], stacked[2:15], stacked[15:]

        return {
            's1': s1,                                       # (2, H, W)
            's2': s2,                                       # (13, H, W)
            's2_cloudy': s2_cloudy,                         # (13, H, W)
            's1_wvs': torch.tensor(S1_WAVELENGTHS),         # (2,)
            's2_wvs': torch.tensor(S2_WAVELENGTHS),         # (13,)
            'patch_id': entry['patch_id'],                  # str
        }


# =============================================================================
# DataModule
# =============================================================================


class SEN12MSCRDataModule(LightningDataModule):
    """Lightning DataModule for SEN12MS-CR mono-temporal cloud removal.

    Args:
        root: Path to the SEN12MS-CR root directory.
        batch_size: Training batch size.
        eval_batch_size: Eval batch size (defaults to batch_size * 2).
        num_workers: Number of DataLoader workers.
        region: One of 'all', 'africa', 'america', 'asiaEast', 'asiaWest', 'europa'.
        normalize: Whether to apply z-score normalization.
        target_size: Optional [H, W] to resize images.
    """

    def __init__(
        self,
        root: str,
        batch_size: int = 16,
        eval_batch_size: int = None,
        num_workers: int = 4,
        region: str = 'all',
        normalize: bool = True,
        target_size=None,
    ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size * 2
        self.num_workers = num_workers
        self.region = region
        self.normalize = normalize

        if isinstance(target_size, (list, ListConfig)):
            target_size = tuple(target_size)
        self.target_size = target_size

    def setup(self, stage=None):
        self.train_dataset = SEN12MSCRDataset(
            self.root, split='train', region=self.region,
            normalize=self.normalize, target_size=self.target_size, augment=True,
        )
        self.val_dataset = SEN12MSCRDataset(
            self.root, split='val', region=self.region,
            normalize=self.normalize, target_size=self.target_size, augment=False,
        )
        self.test_dataset = SEN12MSCRDataset(
            self.root, split='test', region=self.region,
            normalize=self.normalize, target_size=self.target_size, augment=False,
        )

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
