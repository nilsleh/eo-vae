import os
import random

import numpy as np
import torch
from lightning import LightningDataModule
from omegaconf import ListConfig
from terramesh import build_terramesh_dataset
from torch.utils.data import DataLoader

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

import torch.nn as nn


class Sentine2L2AlNorm(nn.Module):
    def __init__(self):
        super().__init__()
        # Use the "Clipped" stats (Max=10000) to prioritize land contrast
        self.register_buffer(
            'mean',
            torch.tensor(
                [
                    2435.8665,
                    2543.3806,
                    2767.4976,
                    2911.2197,
                    3253.6277,
                    3837.3064,
                    4048.8118,
                    4138.8438,
                    4197.8535,
                    4216.5630,
                    3484.7454,
                    2912.8987,
                ]
            ).view(12, 1, 1),
        )

        self.register_buffer(
            'std',
            torch.tensor(
                [
                    2078.0635,
                    2077.7659,
                    1985.1591,
                    2070.0999,
                    2001.9419,
                    1821.0082,
                    1768.0741,
                    1792.8589,
                    1710.2782,
                    1711.2448,
                    1433.0996,
                    1341.4229,
                ]
            ).view(12, 1, 1),
        )

    def forward(self, x):
        # 1. Harmonize: Fix the -1000 offset seen in histogram
        x = x + 1000.0

        # 2. Mask NoData (exact 0 after shift)
        nodata_mask = x <= 0

        # 3. Clip Saturation: Cut off the "Cloud Spike" at 15000
        # to preserve contrast in the "Land Hump" (0-4000)
        x = torch.clamp(x, min=0.0, max=10000.0)

        # 4. Standardize
        x = (x - self.mean) / self.std

        # 5. Fill NoData
        x = x.masked_fill(nodata_mask, 0.0)
        return x


def apply_batch_augmentations(images: torch.Tensor) -> torch.Tensor:
    """Applies random geometric augmentations (D4 Symmetry Group) to a batch of EO images.
    Input: (B, C, H, W)

    Includes:
    - Random Horizontal Flip
    - Random Vertical Flip
    - Random 90-degree Rotation (0, 90, 180, 270)
    """
    # 1. Random Horizontal Flip
    if random.random() > 0.5:
        images = torch.flip(images, dims=[-1])

    # 2. Random Vertical Flip
    if random.random() > 0.5:
        images = torch.flip(images, dims=[-2])

    # 3. Random 90-degree Rotation
    # k is number of times to rotate 90 degrees
    k = random.randint(0, 3)
    if k > 0:
        images = torch.rot90(images, k, dims=[-2, -1])

    return images


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
    modalities,
    normalize=True,
    norm_method='zscore',
    target_size=(224, 224),
    mode='train',
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
            images = torch.from_numpy(images).float()
        if normalize:
            images = normalize_image(images, selected_modality, method=norm_method)

        # if target_size is not None and images.shape[-2:] != target_size:
        #     images = torch.nn.functional.interpolate(
        #         images.float(), size=target_size, mode='bilinear', align_corners=False
        #     )

        # augmentation
        # if mode == "train":
        #     images = apply_batch_augmentations(images)

        # Get wavelengths for the selected modality
        wvs = torch.FloatTensor(WAVELENGTHS[selected_modality])

        return {
            'image': images,
            'wvs': wvs,
            'modality': selected_modality,  # Optional: track which modality was used
        }

    return collate


def deterministic_modality_collate_fn(
    modality, normalize=True, norm_method='zscore', target_size=(224, 224), mode='train'
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

        # if target_size is not None and images.shape[-2:] != target_size:
        #     images = torch.nn.functional.interpolate(
        #         images, size=target_size, mode='bilinear', align_corners=False
        #     )
        # augmentation
        # if mode == "train":
        #     images = apply_batch_augmentations(images)

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
                mode='train',
            )
        else:
            self.train_collate_fn = deterministic_modality_collate_fn(
                modality=train_collate_mode,
                normalize=normalize,
                norm_method=norm_method,
                target_size=target_size,
                mode='train',
            )

        self.val_collate_fn = deterministic_modality_collate_fn(
            modality=val_collate_mode,
            normalize=normalize,
            norm_method=norm_method,
            target_size=target_size,
            mode='eval',
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
            shardshuffle=2000,
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
            # 1. FIX: Disable the duplicate data stream
            # The code normally expects two data sources. Since you only have MajorTom,
            # we set probabilities to [100%, 0%] to ignore the second empty/duplicate pipeline.
            probs=[1.0, 0.0],
            # 2. FIX: Increase shuffle buffer
            # Since the buffer stores compressed Zarr files, we can increase this.
            # 500 samples * 4 workers = mixes ~2000 samples globally.
            # This covers ~20% of a shard, significantly reducing the "fluctuation" bumps.
            shardshuffle=1000,
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


import torch
import torch.nn as nn


class ClippedRunningStats(nn.Module):
    def __init__(self, shape, dims, shift_offset=1000.0, saturation_threshold=10000.0):
        """Computes stats for Shifted Sentinel-2 Data (Baseline 04.00).

        Logic:
          1. Shifts input by +1000.
          2. Treats exact 0 as NoData.
          3. Clips saturation at 10000.
        """
        super().__init__()

        # Handle shape argument (int or tuple)
        if isinstance(shape, (tuple, list)):
            self.num_channels = shape[0]
            self.shape_tuple = shape
        else:
            self.num_channels = shape
            self.shape_tuple = (shape,)

        self.dims = dims
        self.shift_offset = shift_offset
        self.saturation_threshold = saturation_threshold

        # Initialize Buffers
        self.register_buffer('mean', torch.zeros(self.shape_tuple))
        self.register_buffer('var', torch.ones(self.shape_tuple))
        self.register_buffer('std', torch.ones(self.shape_tuple))
        self.register_buffer('count', torch.zeros(self.shape_tuple))
        self.register_buffer('min', torch.full(self.shape_tuple, float('inf')))
        self.register_buffer('max', torch.full(self.shape_tuple, float('-inf')))

    def update(self, x):
        with torch.no_grad():
            # --- STEP 1: HARMONIZE ---
            # Restore physical meaning.
            # -1000 (NoData) becomes 0.
            # -800 (Dark Pixel) becomes 200.
            x_shifted = x + self.shift_offset

            # --- STEP 2: DEFINE VALID DATA ---
            # We treat 0 as the 'NoData' marker.
            # Real data (even dark shadows) is typically > 0 (e.g. 10 or 20).
            valid_mask = x_shifted > 0

            # --- STEP 3: CLIP SATURATION ---
            # We clamp clouds to 10000 to prevent outliers from skewing the mean.
            # We apply min=0 to ensure no lingering negatives mess up the math.
            x_fixed = torch.clamp(x_shifted, min=0.0, max=self.saturation_threshold)

            # --- STEP 4: WELFORD UPDATE (Standard) ---

            # Permute dimensions to isolate channels
            all_dims = list(range(x.ndim))
            channel_dim = [d for d in all_dims if d not in self.dims]
            permute_order = channel_dim + self.dims

            x_flat = x_fixed.permute(permute_order).reshape(self.num_channels, -1)
            mask_flat = valid_mask.permute(permute_order).reshape(self.num_channels, -1)

            batch_mean = torch.zeros_like(self.mean)
            batch_var = torch.zeros_like(self.var)
            batch_count = torch.zeros_like(self.count)

            for c in range(self.num_channels):
                # Only select VALID pixels
                valid_pixels = x_flat[c][mask_flat[c]]
                n = valid_pixels.numel()

                if n > 0:
                    batch_mean[c] = valid_pixels.mean()
                    batch_var[c] = valid_pixels.var(unbiased=False) if n > 1 else 0.0
                    batch_count[c] = n

                    # Update Min/Max
                    current_min = valid_pixels.min()
                    current_max = valid_pixels.max()
                    if current_min < self.min[c]:
                        self.min[c] = current_min
                    if current_max > self.max[c]:
                        self.max[c] = current_max

            # Merge with running stats
            delta = batch_mean - self.mean
            total_count = self.count + batch_count
            denom = total_count.clamp(min=1)

            new_mean = self.mean + delta * (batch_count / denom)
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + delta**2 * self.count * batch_count / denom

            update_locs = batch_count > 0
            self.mean[update_locs] = new_mean[update_locs]
            self.var[update_locs] = (M2 / denom)[update_locs]
            self.std[update_locs] = torch.sqrt(self.var[update_locs] + 1e-8)
            self.count += batch_count

    def forward(self, x):
        self.update(x)
        return x


import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class StreamingHistogram(nn.Module):
    def __init__(self, num_channels, min_val=-1200, max_val=1000, bins=2200):
        """Accumulates histogram counts for a specific value range.
        Default range [-1200, 1000] with 2200 bins gives us ~1 unit precision per bin.
        """
        super().__init__()
        self.num_channels = num_channels
        self.min_val = min_val
        self.max_val = max_val
        self.bins = bins

        # Store counts: Shape (Channels, Bins)
        self.register_buffer('hist_counts', torch.zeros(num_channels, bins))

        # Create bin edges for plotting later
        self.bin_edges = torch.linspace(min_val, max_val, steps=bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

    def update(self, x):
        """Updates the histogram with a new batch of data."""
        with torch.no_grad():
            # Flatten: (B, C, H, W) -> (C, -1)
            if x.ndim == 4:
                x_flat = x.permute(1, 0, 2, 3).reshape(self.num_channels, -1)
            else:
                x_flat = x.reshape(self.num_channels, -1)

            for c in range(self.num_channels):
                # torch.histc only works on 1D tensors and doesn't do batching well
                # So we loop over channels (fast enough since C is small, e.g. 12)
                batch_hist = torch.histc(
                    x_flat[c].float(),
                    bins=self.bins,
                    min=self.min_val,
                    max=self.max_val,
                )
                self.hist_counts[c] += batch_hist

    def plot(self, channel_names=None):
        """Plots the accumulated histograms."""
        counts_np = self.hist_counts.cpu().numpy()
        centers_np = self.bin_centers.cpu().numpy()

        plt.figure(figsize=(15, 8))

        for c in range(self.num_channels):
            label = channel_names[c] if channel_names else f'Ch {c}'

            # Use log scale for Y because NoData spikes are usually huge
            # compared to normal data
            plt.plot(centers_np, counts_np[c], label=label, alpha=0.7)

        plt.yscale('log')
        plt.title("Data Distribution: The 'Problem Zone' (Negative Values)")
        plt.xlabel('Pixel Value')
        plt.ylabel('Count (Log Scale)')
        plt.grid(True, which='both', ls='-', alpha=0.2)
        plt.legend()

        # Draw lines to help analysis
        plt.axvline(
            x=-100, color='r', linestyle='--', alpha=0.5, label='Expected NoData (-999)'
        )
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.5, label='Zero')

        plt.savefig('streaming_histogram.png')
        plt.show()


if __name__ == '__main__':
    from tqdm import tqdm

    dm = TerraMeshDataModule(
        data_path='/mnt/SSD2/nils/datasets/terramesh',
        modalities=['S2L2A'],
        batch_size=8,
        num_workers=4,
        norm_method='zscore',
        normalize=False,
    )

    dm.setup('fit')
    train_loader = dm.train_dataloader()

    out = nex(iter(train_loader))

    # L2A stats
    stats = ClippedRunningStats(
        (12,), dims=[0, 2, 3], shift_offset=1000.0, saturation_threshold=10000.0
    )
    # L1C stats
    stats = ClippedRunningStats(
        (13,), dims=[0, 2, 3], shift_offset=0.0, saturation_threshold=12000.0
    )

    hist_computer = StreamingHistogram(
        num_channels=2, min_val=-100, max_val=100, bins=500
    )

    max_num = 300
    i = 0
    for batch in tqdm(train_loader):
        hist_computer.update(batch['image'])
        if i >= max_num:
            break
        i += 1
    hist_computer.plot()

    print('Computed Stats:')
    print('Mean:', stats.mean)
    print('Std:', stats.std)
    print('Min:', stats.min)
    print('Max:', stats.max)
