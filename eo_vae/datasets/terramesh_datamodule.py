import os
import random
import numpy as np
import torch
from lightning import LightningDataModule
from omegaconf import ListConfig
from torch.utils.data import DataLoader
from .terramesh import build_terramesh_dataset

# Wavelength definitions (unchanged)
WAVELENGTHS = {
    'S2RGB': [0.665, 0.56, 0.49],
    'S1RTC': [5.4, 5.6],
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
    ],
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
    ],
}

# Updated with your computed stats
NORM_STATS = {
    'S2L2A': {
        'mean': [
            1375.648,
            1489.600,
            1709.087,
            1831.752,
            2186.075,
            2794.358,
            3008.528,
            3096.780,
            3155.180,
            3169.651,
            2415.761,
            1838.622,
        ],
        'std': [
            2101.107,
            2138.673,
            2033.628,
            2118.186,
            2061.646,
            1869.234,
            1801.386,
            1841.173,
            1734.404,
            1751.174,
            1375.131,
            1284.165,
        ],
        'robust_min': 0.0,
        'robust_max': 4000.0,  # Based on histogram peaks/land-surface range
    },
    'S1RTC': {
        'mean': [-10.793, -17.198],
        'std': [4.278, 4.346],
        'robust_min': -25.0,
        'robust_max': 0.0,
    },
    'S2L1C': {
        'mean': [
            2475.625,
            2260.839,
            2143.561,
            2230.225,
            2445.427,
            2992.950,
            3257.843,
            3171.695,
            3440.958,
            1567.433,
            561.076,
            2562.809,
            1924.178,
        ],
        'std': [
            1761.905,
            1804.267,
            1661.263,
            1932.020,
            1918.007,
            1812.421,
            1795.179,
            1734.280,
            1780.039,
            1082.531,
            512.077,
            1350.580,
            1177.511,
        ],
        'robust_min': 0.0,
        'robust_max': 4500.0,
    },
    'S2RGB': {
        'mean': [110.349, 99.507, 75.843],
        'std': [69.905, 53.708, 53.378],
        'robust_min': 0.0,
        'robust_max': 255.0,
    },
    'DEM': {
        'mean': [651.663],
        'std': [928.168],
        'robust_min': 0.0,
        'robust_max': 2500.0,
    },
}


def normalize_image(
    image: torch.Tensor, modality: str, method='zscore'
) -> torch.Tensor:
    """Normalize image tensor using chosen method."""
    if modality not in NORM_STATS:
        raise ValueError(f'Unknown modality {modality} for normalization.')

    stats = NORM_STATS[modality]
    device = image.device

    if method == 'zscore':
        mean = torch.tensor(stats['mean'], device=device).view(-1, 1, 1)
        std = torch.tensor(stats['std'], device=device).view(-1, 1, 1)
        return (image - mean) / (std + 1e-8)

    elif method == 'robust':
        v_min = stats['robust_min']
        v_max = stats['robust_max']
        # Linear stretch mapping [min, max] to [-1, 1]
        image = (image - v_min) / (v_max - v_min + 1e-8) * 2.0 - 1.0
        # Optional: clamp to avoid extreme cloud/glint outliers destabilizing gradients
        # return torch.clamp(image, -2.5, 5.0)
        return torch.clamp(image, -1.0, 1.0)

    else:
        raise ValueError(f'Normalization method {method} not supported.')


def unnormalize_image(
    image: torch.Tensor, modality: str, method='zscore'
) -> torch.Tensor:
    """Inverse normalization to recover original units (DN or dB)."""
    if modality not in NORM_STATS:
        return image

    stats = NORM_STATS[modality]
    device = image.device

    if method == 'zscore':
        mean = torch.tensor(stats['mean'], device=device).view(-1, 1, 1)
        std = torch.tensor(stats['std'], device=device).view(-1, 1, 1)
        return (image * std) + mean

    elif method == 'robust':
        v_min = stats['robust_min']
        v_max = stats['robust_max']
        # Reverse linear stretch: [-1, 1] back to [min, max]
        return ((image + 1.0) / 2.0) * (v_max - v_min) + v_min

    return image


def single_modality_collate_fn(
    modalities, normalize=True, norm_method='zscore', target_size=(224, 224)
):
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
            images = normalize_image(images, selected_modality, method=norm_method)

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


def deterministic_modality_collate_fn(
    modality, normalize=True, norm_method='zscore', target_size=(224, 224)
):
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
            images = normalize_image(images, modality, method=norm_method)

        if target_size is not None and images.shape[-2:] != target_size:
            images = torch.nn.functional.interpolate(
                images, size=target_size, mode='bilinear', align_corners=False
            )

        wvs = torch.FloatTensor(WAVELENGTHS[modality])

        return {'image': images, 'wvs': wvs, 'modality': modality}

    return collate


class TerraMeshDataModule(LightningDataModule):
    def __init__(
        self,
        data_path,
        modalities,
        batch_size=8,
        eval_batch_size=16,
        num_workers=4,
        train_collate_mode='random',
        val_collate_mode='S2L2A',
        normalize=True,
        norm_method='z-score',
        target_size=(224, 224),
        **kwargs,
    ):
        super().__init__()
        self.data_path = data_path
        self.modalities = modalities
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        self.norm_method = norm_method

        self.kwargs = kwargs

        if isinstance(target_size, ListConfig):
            target_size = tuple(target_size)
        self.target_size = target_size

        # Make norm_stats accessible
        self.norm_stats = NORM_STATS

        for mod in modalities:
            if mod not in WAVELENGTHS:
                raise ValueError(
                    f'Modality {mod} not supported. Available: {list(WAVELENGTHS.keys())}'
                )

        # Create collate functions with the chosen norm_method
        if train_collate_mode == 'random':
            self.train_collate_fn = single_modality_collate_fn(
                modalities=modalities,
                normalize=normalize,
                norm_method=norm_method,
                target_size=target_size,
            )
        else:
            self.train_collate_fn = deterministic_modality_collate_fn(
                modality=train_collate_mode,
                normalize=normalize,
                norm_method=norm_method,
                target_size=target_size,
            )

        self.val_collate_fn = deterministic_modality_collate_fn(
            modality=val_collate_mode,
            normalize=normalize,
            norm_method=norm_method,
            target_size=target_size,
        )

    def setup(self, stage=None):
        """Set up the train and validation datasets."""
        if len(self.modalities) > 1:
            # Folder name format for multi-modality: [mod1,mod2,mod3]
            mod_path_segment = f'[{",".join(self.modalities)}]'
        else:
            # Folder name format for single modality: mod1
            mod_path_segment = self.modalities[0]

        train_urls = os.path.join(
            self.data_path,
            'train',
            mod_path_segment,
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
            mod_path_segment,
            'majortom_shard_{000001..000005}.tar',
        )

        test_urls = os.path.join(
            self.data_path,
            'val',
            mod_path_segment,
            'majortom_shard_{000006..000008}.tar',
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

        self.val_dataset = build_terramesh_dataset(
            path=self.data_path,
            modalities=self.modalities,
            split='val',
            urls=val_urls,
            batch_size=self.eval_batch_size,
            shuffle=False,
            **self.kwargs,
        )

        self.test_dataset = build_terramesh_dataset(
            path=self.data_path,
            modalities=self.modalities,
            split='val',
            urls=test_urls,
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

    def test_dataloader(self):
        """Return the test DataLoader with deterministic modality."""
        return DataLoader(
            self.test_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            collate_fn=self.val_collate_fn,
            pin_memory=True,
        )
