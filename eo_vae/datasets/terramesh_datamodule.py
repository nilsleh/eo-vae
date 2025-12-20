import os
import random

import numpy as np
import torch
from lightning import LightningDataModule
from omegaconf import ListConfig
from torch.utils.data import DataLoader

from .terramesh import build_terramesh_dataset

# Wavelength definitions for different modalities (in micrometers)
WAVELENGTHS = {
    'S2RGB': [0.665, 0.56, 0.49],  # R, G, B
    'S1RTC': [5.4, 5.6],  # VV, VH
    'S2L2A': [
        0.443,
        0.490,
        0.560,
        0.665,
        0.705,
        0.740,
        0.783,
        0.842,
        0.865,
        1.610,
        2.190,
        0.945,
    ],  # 12 bands
    'S2L1C': [
        0.443,
        0.490,
        0.560,
        0.665,
        0.705,
        0.740,
        0.783,
        0.842,
        0.865,
        0.945,
        1.375,
        1.610,
        2.190,
    ],  # 13 bands
}

NORM_STATS = {
    'S2L1C': {
        'mean': [
            2357.090,
            2137.398,
            2018.799,
            2082.998,
            2295.663,
            2854.548,
            3122.860,
            3040.571,
            3306.491,
            1473.849,
            506.072,
            2472.840,
            1838.943,
        ],
        'std': [
            1673.639,
            1722.641,
            1602.205,
            1873.138,
            1866.055,
            1779.839,
            1776.496,
            1724.114,
            1771.041,
            1079.786,
            512.404,
            1340.879,
            1172.435,
        ],
    },
    'S2L2A': {
        'mean': [
            1390.461,
            1503.332,
            1718.211,
            1853.926,
            2199.116,
            2779.989,
            2987.025,
            3083.248,
            3132.235,
            3162.989,
            2424.902,
            1857.665,
        ],
        'std': [
            2131.157,
            2163.666,
            2059.311,
            2152.477,
            2105.179,
            1912.773,
            1842.326,
            1893.568,
            1775.656,
            1814.907,
            1436.282,
            1336.155,
        ],
    },
    'S2RGB': {'mean': [110.349, 99.507, 75.843], 'std': [69.905, 53.708, 53.378]},
    'S1GRD': {'mean': [-12.577, -20.265], 'std': [5.179, 5.872]},
    'S1RTC': {'mean': [-10.93, -17.329], 'std': [4.391, 4.459]},
    'NDVI': {'mean': [0.327], 'std': [0.322]},
    'DEM': {'mean': [651.663], 'std': [928.168]},
}


def normalize_image(image: torch.Tensor, modality: str) -> torch.Tensor:
    """Normalize image tensor using mean/std for the modality."""
    if modality not in NORM_STATS:
        raise ValueError(f'Unknown modality {modality} for normalization.')

    mean = torch.tensor(NORM_STATS[modality]['mean'], device=image.device).view(
        -1, 1, 1
    )
    std = torch.tensor(NORM_STATS[modality]['std'], device=image.device).view(-1, 1, 1)
    return (image - mean) / (std + 1e-8)


# def normalize_image(image: torch.Tensor, modality: str) -> torch.Tensor:
#     """Normalize image tensor using robust scaling: x = (x - (mean - 3*std)) / (6*std)."""
#     if modality not in NORM_STATS:
#         raise ValueError(f"Unknown modality {modality} for normalization.")

#     mean = torch.tensor(NORM_STATS[modality]["mean"], device=image.device).view(
#         -1, 1, 1
#     )
#     std = torch.tensor(NORM_STATS[modality]["std"], device=image.device).view(-1, 1, 1)

#     # Compute xmin and xmax
#     xmin = mean - 3 * std
#     xmax = mean + 3 * std

#     # Apply normalization: x = (x - xmin) / (xmax - xmin)
#     return (image - xmin) / (xmax - xmin + 1e-8)


def single_modality_collate_fn(modalities, normalize=True, target_size=(224, 224)):
    """Collate function that randomly selects ONE modality per batch.

    Args:
        modalities: List of modalities to randomly select from (e.g., ['S2RGB', 'S2L2A'])
        normalize: Whether to apply 0-1 normalization
        target_size: Target image size for interpolation

    Returns:
        Callable collate function
    """

    def collate(batch):
        """Process batch by selecting a single random modality.

        Args:
            batch: Dict with keys like 'S2RGB', 'S2L2A', or 'image' if single modality.

        Returns:
            Dict with keys 'image' and 'wvs'
        """
        # Handle single modality case where key is "image"
        if len(modalities) == 1:
            selected_modality = modalities[0]
            if 'image' in batch:
                images = batch['image']
            else:
                raise ValueError(
                    f"Expected 'image' key for single modality {selected_modality}, but found: {batch.keys()}"
                )
        else:
            # Randomly select one modality for this batch that is also in the batch, or keep trying
            available_modalities = [mod for mod in modalities if mod in batch]
            if not available_modalities:
                raise ValueError(
                    f'None of the specified modalities {modalities} found in batch keys {batch.keys()}'
                )
            selected_modality = random.choice(available_modalities)
            images = batch[selected_modality]

        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if normalize:
            images = normalize_image(images, selected_modality)

        if target_size is not None and images.shape[-2:] != target_size:
            images = torch.nn.functional.interpolate(
                images, size=target_size, mode='bilinear', align_corners=False
            )

        # Get wavelengths for the selected modality
        wvs = torch.FloatTensor(WAVELENGTHS[selected_modality])

        return {
            'image': images,
            'wvs': wvs,
            'modality': selected_modality,  # Optional: track which modality was used
        }

    return collate


def deterministic_modality_collate_fn(modality, normalize=True, target_size=(224, 224)):
    """Collate function that always uses the SAME modality.

    Useful for validation where you want consistent modality.

    Args:
        modality: Single modality name (e.g., 'S2L2A')
        normalize: Whether to apply 0-1 normalization
        target_size: Target image size for interpolation

    Returns:
        Callable collate function
    """

    def collate(batch):
        # Handle case where single modality uses "image" key
        if modality in batch:
            images = batch[modality]
        elif 'image' in batch:
            images = batch['image']
        else:
            raise ValueError(
                f'Modality {modality} not found in batch. Available: {batch.keys()}'
            )

        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if normalize:
            images = normalize_image(images, modality)

        if target_size is not None and images.shape[-2:] != target_size:
            images = torch.nn.functional.interpolate(
                images, size=target_size, mode='bilinear', align_corners=False
            )

        wvs = torch.FloatTensor(WAVELENGTHS[modality])

        return {'image': images, 'wvs': wvs, 'modality': modality}

    return collate


class TerraMeshDataModule(LightningDataModule):
    """Lightning DataModule for TerraMesh streaming datasets with multi-modality support."""

    def __init__(
        self,
        data_path,
        modalities,
        batch_size=8,
        eval_batch_size=16,
        num_workers=4,
        train_collate_mode='random',  # "random" or specific modality name
        val_collate_mode='S2L2A',  # Deterministic modality for validation
        normalize=True,
        target_size=(224, 224),
        **kwargs,
    ):
        """Initialize the TerraMeshDataModule.

        Args:
            data_path (str): Path to the TerraMesh dataset.
            modalities (list): List of modalities to load (e.g., ['S2RGB', 'S2L2A']).
            batch_size (int): Batch size for training.
            eval_batch_size (int): Batch size for validation.
            num_workers (int): Number of worker processes for data loading.
            train_collate_mode (str): Either "random" to randomly select modality per batch,
                                     or a specific modality name for deterministic selection.
            val_collate_mode (str): Modality to use for validation (should be deterministic).
            normalize (bool): Whether to apply 0-1 normalization.
            target_size (tuple): Target image size (H, W).
            **kwargs: Additional keyword arguments for the dataset loader.
        """
        super().__init__()
        self.data_path = data_path
        self.modalities = modalities
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.train_collate_mode = train_collate_mode
        self.val_collate_mode = val_collate_mode
        self.normalize = normalize
        if isinstance(target_size, ListConfig):
            target_size = tuple(target_size)
        self.target_size = target_size
        self.kwargs = kwargs

        # Make norm_stats accessible
        self.norm_stats = NORM_STATS

        # Validate modalities have wavelength definitions
        for mod in modalities:
            if mod not in WAVELENGTHS:
                raise ValueError(
                    f'Modality {mod} not supported. Available: {list(WAVELENGTHS.keys())}'
                )

        # Create collate functions
        if train_collate_mode == 'random':
            self.train_collate_fn = single_modality_collate_fn(
                modalities=modalities, normalize=normalize, target_size=target_size
            )
        else:
            self.train_collate_fn = deterministic_modality_collate_fn(
                modality=train_collate_mode,
                normalize=normalize,
                target_size=target_size,
            )

        self.val_collate_fn = deterministic_modality_collate_fn(
            modality=val_collate_mode, normalize=normalize, target_size=target_size
        )

    def setup(self, stage=None):
        """Set up the train and validation datasets."""
        train_urls = os.path.join(
            self.data_path,
            'train',
            f'[{",".join(self.modalities)}]',
            'majortom_shard_{000001..000025}.tar',
        )
        self.train_dataset = build_terramesh_dataset(
            path=self.data_path,
            urls=train_urls,
            modalities=self.modalities,
            split='train',
            batch_size=self.batch_size,
            shuffle=True,
            **self.kwargs,
        )
        val_urls = os.path.join(
            self.data_path,
            'val',
            f'[{",".join(self.modalities)}]',
            'majortom_shard_{000001..000008}.tar',
        )
        self.val_dataset = build_terramesh_dataset(
            path=self.data_path,
            modalities=self.modalities,
            split='val',
            urls=val_urls,
            batch_size=self.eval_batch_size,
            shuffle=False,
            **self.kwargs,
        )

    def train_dataloader(self):
        """Return the training DataLoader with random modality selection."""
        return DataLoader(
            self.train_dataset,
            batch_size=None,  # Batching handled by webdataset
            num_workers=self.num_workers,
            collate_fn=self.train_collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Return the validation DataLoader with deterministic modality."""
        return DataLoader(
            self.val_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            collate_fn=self.val_collate_fn,
            pin_memory=True,
        )
