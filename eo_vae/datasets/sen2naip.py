"""Sen2Naip Dataset for Super Resolution."""

import logging

logging.getLogger('rasterio._env').setLevel(logging.ERROR)

import os
from collections.abc import Callable, Sequence
from glob import glob
from typing import ClassVar, Sequence, Any
from torch import Generator

import hashlib
import pyproj
from shapely.wkt import loads as wkt_loads

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
import json
import re
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import random_split

from torchgeo.datasets.geo import NonGeoDataset
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.datasets.utils import Path, array_to_tensor


def assign_spatial_split(
    df: pd.DataFrame,
    lon_col: str = 'lon',
    lat_col: str = 'lat',
    n_blocks_x: int = 12,
    n_blocks_y: int = 8,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Split dataset into spatial blocks (grid) and randomly assign blocks to splits.
    Implementing the 'random' pattern strategy.
    """
    if df.empty:
        return df

    # 1. Calculate Bounds and Grid
    minx, maxx = df[lon_col].min(), df[lon_col].max()
    miny, maxy = df[lat_col].min(), df[lat_col].max()

    # Buffer to avoiding edge issues
    buffer_x = (maxx - minx) * 0.001
    buffer_y = (maxy - miny) * 0.001
    minx -= buffer_x
    maxx += buffer_x
    miny -= buffer_y
    maxy += buffer_y

    x_step = (maxx - minx) / n_blocks_x
    y_step = (maxy - miny) / n_blocks_y

    # Assign Grid Indices
    block_x = ((df[lon_col] - minx) / x_step).astype(int).clip(0, n_blocks_x - 1)
    block_y = ((df[lat_col] - miny) / y_step).astype(int).clip(0, n_blocks_y - 1)
    df['block_id'] = block_y * n_blocks_x + block_x

    # 2. Block-wise Splitting (Random)
    # Generate all possible block IDs for the grid
    total_grid_blocks = n_blocks_x * n_blocks_y
    all_blocks = np.arange(total_grid_blocks)

    rs = np.random.RandomState(random_state)
    rs.shuffle(all_blocks)

    # Determine number of blocks for each split (e.g. 10% val, 10% test)
    n_test_blocks = max(1, int(total_grid_blocks * 0.1))
    n_val_blocks = max(1, int(total_grid_blocks * 0.1))

    # Assign sets of blocks
    test_blocks = set(all_blocks[:n_test_blocks])
    val_blocks = set(all_blocks[n_test_blocks : n_test_blocks + n_val_blocks])
    # Remaining are train

    def get_split_label(bid):
        if bid in test_blocks:
            return 'test'
        if bid in val_blocks:
            return 'val'
        return 'train'

    df['split'] = df['block_id'].map(get_split_label)
    return df


class Sen2NaipCrossSensor(NonGeoDataset):
    """Sen2Naip Cross Sensor dataset for super resolution."""

    # filename = "sen2naipv2-histmatch.0000.part.taco"
    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Sen2Naip dataset instance.

        Args:
            root: Root directory where the dataset should be stored.
            split: Dataset split to load. Must be one of 'train' or 'val'.
            transforms: A function/transform that takes input sample and its target as entry
                and returns a transformed version.
            download: Whether to download the dataset if it is not found on disk.
            checksum: Whether to verify the integrity of the dataset after download.
        """
        self.root = root
        self.split = split
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self.aois = sorted(
            glob(os.path.join(self.root, '**', 'metadata.json'), recursive=True)
        )

        self.metadata_df = self._create_spatial_split(self.aois)

    def _create_spatial_split(self, aoi_paths: list[str]) -> pd.DataFrame:
        """Parse metadata and create a deterministic spatial split."""
        data = []
        transformers = {}
        target_crs = pyproj.CRS('EPSG:4326')

        for path in aoi_paths:
            with open(path) as f:
                meta = json.load(f)

            # Extract geometry info
            wkt = meta.get('proj:geometry')
            epsg = meta.get('proj:epsg')
            if not wkt or not epsg:
                continue

            # Get or create transformer for this EPSG
            if epsg not in transformers:
                source_crs = pyproj.CRS(f'EPSG:{epsg}')
                transformers[epsg] = pyproj.Transformer.from_crs(
                    source_crs, target_crs, always_xy=True
                ).transform

            # Parse geometry and reproject centroid to WGS84
            geom = wkt_loads(wkt)
            minx, miny, maxx, maxy = geom.bounds
            cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
            lon, lat = transformers[epsg](cx, cy)

            data.append(
                {
                    'aoi_id': meta.get('id', os.path.basename(os.path.dirname(path))),
                    'lr_path': os.path.join(os.path.dirname(path), 'lr.tif'),
                    'hr_path': os.path.join(os.path.dirname(path), 'hr.tif'),
                    'lon': lon,
                    'lat': lat,
                }
            )

        df = pd.DataFrame(data)

        df = assign_spatial_split(df, lon_col='lon', lat_col='lat')

        # # plot the splits to check
        # fig: Figure = plt.figure(figsize=(8, 6))
        # ax = fig.add_subplot(1, 1, 1)
        # colors = {'train': 'blue', 'val': 'orange', 'test': 'green'}
        # for split in ['train', 'val', 'test']:
        #     subset = df[df['split'] == split]
        #     ax.scatter(subset['lon'], subset['lat'], c=colors[split], label=split, alpha=0.5)
        # ax.set_xlabel('Longitude')
        # ax.set_ylabel('Latitude')
        # ax.set_title('AOI Spatial Split')
        # ax.legend()
        # fig.savefig('sen2naip_spatial_split.png', dpi=300)

        # import pdb
        # pdb.set_trace()

        df = df[df['split'] == self.split].reset_index(drop=True)
        return df

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.metadata_df)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Return dataset sample at this index."""

        row = self.metadata_df.iloc[idx]

        lr_path: str = row['lr_path']
        hr_path: str = row['hr_path']

        with rasterio.open(lr_path) as lr_src, rasterio.open(hr_path) as hr_src:
            lr_data: np.ndarray = lr_src.read()
            hr_data: np.ndarray = hr_src.read()

        # resize hr image to 512x512 from 520x520
        # resize lr image to 128x128 from 121x121
        hq_image = F.interpolate(
            array_to_tensor(hr_data).float().unsqueeze(0),
            size=512,
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)
        lq_image = F.interpolate(
            array_to_tensor(lr_data).float().unsqueeze(0),
            size=128,
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)

        sample = {'image_lr': lq_image, 'image_hr': hq_image, 'aoi': row['aoi_id']}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


class RunningStatsButFast(torch.nn.Module):
    def __init__(self, shape, dims):
        """Initializes the RunningStatsButFast method.

        A PyTorch module that can be put on the GPU and calculate the multidimensional
        mean and variance of inputs online in a numerically stable way. This is useful
        for calculating the channel-wise mean and variance of a big dataset because you
        don't have to load the entire dataset into memory.

        Uses the "Parallel algorithm" from: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        Similar implementation here: https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py#L5

        Access the mean, variance, and standard deviation of the inputs with the
        `mean`, `var`, and `std` attributes.

        Example:
        ```
        rs = RunningStatsButFast((12,), [0, 2, 3])
        for inputs, _ in dataloader:
            rs(inputs)
        print(rs.mean)
        print(rs.var)
        print(rs.std)
        ```

        Args:
            shape: The shape of resulting mean and variance. For example, if you
                are calculating the mean and variance over the 0th, 2nd, and 3rd
                dimensions of inputs of size (64, 12, 256, 256), this should be 12.
            dims: The dimensions of your input to calculate the mean and variance
                over. In the above example, this should be [0, 2, 3].
        """
        super(RunningStatsButFast, self).__init__()
        self.register_buffer('mean', torch.zeros(shape))
        self.register_buffer('var', torch.ones(shape))
        self.register_buffer('std', torch.ones(shape))
        self.register_buffer('count', torch.zeros(1))
        self.register_buffer('min', torch.zeros(shape))
        self.register_buffer('max', torch.zeros(shape))
        self.dims = dims

    def update(self, x):
        with torch.no_grad():
            batch_mean = torch.mean(x, dim=self.dims)
            batch_var = torch.var(x, dim=self.dims)
            batch_min = torch.amin(x, dim=self.dims)
            batch_max = torch.amax(x, dim=self.dims)
            batch_count = torch.tensor(x.shape[self.dims[0]], dtype=torch.float)

            n_ab = self.count + batch_count
            m_a = self.mean * self.count
            m_b = batch_mean * batch_count
            M2_a = self.var * self.count
            M2_b = batch_var * batch_count

            delta = batch_mean - self.mean

            self.mean = (m_a + m_b) / (n_ab)
            # we don't subtract -1 from the denominator to match the standard Numpy/PyTorch variances
            self.var = (M2_a + M2_b + delta**2 * self.count * batch_count / (n_ab)) / (
                n_ab
            )
            self.count += batch_count
            self.std = torch.sqrt(self.var + 1e-8)

            self.min = torch.min(self.min, batch_min)
            self.max = torch.max(self.max, batch_max)

    def forward(self, x):
        self.update(x)
        return x


# from torch.utils.data import DataLoader
# from tqdm import tqdm
# # compute the mean and std of the image_lr and image_hr over the entire train dataset
# ds = Sen2NaipCrossSensor(root='/mnt/SSD2/nils/datasets/sen2naip/cross-sensor/cross-sensor', split='train')

# rs_lr = RunningStatsButFast((4, ), dims=[0, 2, 3])
# rs_hr = RunningStatsButFast((4, ), dims=[0, 2, 3])

# dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)

# for batch in tqdm(dl):
#     rs_lr(batch['image_lr'])
#     rs_hr(batch['image_hr'])

# print('LR Stats:')
# print('Mean:', rs_lr.mean)
# print('Std:', rs_lr.std)
# print('Min:', rs_lr.min)
# print('Max:', rs_lr.max)

# print('HR Stats:')
# print('Mean:', rs_hr.mean)
# print('Std:', rs_hr.std)
# print('Min:', rs_hr.min)
# print('Max:', rs_hr.max)


class Sen2NaipCrossSensorLatent(NonGeoDataset):
    """Sen2Naip latent encodings dataset."""

    valid_splits = ['train', 'val']

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        latent_scale_factor: float = 1.0,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Sen2Naip dataset instance.

        Args:
            root: Root directory where the dataset should be stored.
            split: Dataset split to load. Must be one of 'train' or 'val'.
            latent_scale_factor: Scale factor to apply to the latent encodings.
            transforms: A function/transform that takes input sample and its target as entry
                and returns a transformed version.
            download: Whether to download the dataset if it is not found on disk.
            checksum: Whether to verify the integrity of the dataset after download.
        """

        assert split in self.valid_splits, f'Split must be one of {self.valid_splits}'

        self.root = root
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self.latent_scale_factor = latent_scale_factor

        self.aois = glob(os.path.join(self.root, split, '*.npz'))

        self.metadata_df = pd.DataFrame(self.aois, columns=['path'])

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.metadata_df)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Return dataset sample at this index.

        Args:
            idx: Index of the dataset sample.

        Returns:
            A dataset sample containing the lr and hr image
        """
        row = self.metadata_df.iloc[idx]

        path = row['path']

        with np.load(path) as data:
            hr_latent = torch.from_numpy(data['hr_latent'])
            lr_latent = torch.from_numpy(data['lr_latent'])

        sample = {
            'image_hr': hr_latent * self.latent_scale_factor,
            'image_lr': lr_latent * self.latent_scale_factor,
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


def sen2naip_cross_sensor_collate_fn(
    batch: Sequence[dict[str, Tensor]],
) -> dict[str, Tensor]:
    """Collate function for the Sen2NaipCrossSensor dataset."""
    # LR Stats
    lr_mean = torch.tensor([1302.9685, 1085.2820, 764.7739, 2769.4824]).view(1, 4, 1, 1)
    lr_std = torch.tensor([780.8768, 513.2825, 414.3385, 793.6396]).view(1, 4, 1, 1)

    # HR Stats
    hr_mean = torch.tensor([125.1176, 121.9117, 100.0240, 143.8500]).view(1, 4, 1, 1)
    hr_std = torch.tensor([39.8066, 30.3501, 28.9109, 28.8952]).view(1, 4, 1, 1)

    images_hr = torch.stack([sample['image_hr'] for sample in batch])
    # Z-score normalization for HR
    new_images_hr = (images_hr - hr_mean) / hr_std

    images_lr = torch.stack([sample['image_lr'] for sample in batch])

    # Z-score normalization for LR
    images_lr = (images_lr - lr_mean) / lr_std

    # Interpolate low res image to high res image size
    new_images_lr = F.interpolate(
        images_lr, size=images_hr.shape[-2:], mode='bicubic', align_corners=False
    )

    return {
        'image_lr': new_images_lr,
        'image_hr': new_images_hr,
        'aoi': [sample['aoi'] for sample in batch],
    }


class Sen2NaipCrossSensorDataModule(NonGeoDataModule):
    std = torch.tensor([1])

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new Original WorldStrat DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.Sen2NaipCrossSensor`.
        """
        super().__init__(Sen2NaipCrossSensor, batch_size, num_workers, **kwargs)

        self.collate_fn = sen2naip_cross_sensor_collate_fn

    def setup(self, stage: str) -> None:
        """Set up datasets."""
        self.train_dataset = self.dataset_class(**self.kwargs, split='train')
        self.val_dataset = self.dataset_class(**self.kwargs, split='val')
        self.test_dataset = self.dataset_class(**self.kwargs, split='test')

    def on_after_batch_transfer(
        self, batch: dict[str, Tensor], dataloader_idx: int
    ) -> dict[str, Tensor]:
        """Hook to modify batch after transfer to device."""
        # IMPORTANT to not use torchgeo default for now
        return batch


class Sen2NaipLatentCrossSensorDataModule(NonGeoDataModule):
    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new Original WorldStrat DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.Sen2NaipCrossSensor`.
        """
        super().__init__(Sen2NaipCrossSensorLatent, batch_size, num_workers, **kwargs)

    def setup(self, stage: str) -> None:
        """Set up datasets."""
        self.train_dataset = self.dataset_class(split='train', **self.kwargs)
        self.val_dataset = self.dataset_class(split='val', **self.kwargs)

    def on_after_batch_transfer(
        self, batch: dict[str, Tensor], dataloader_idx: int
    ) -> dict[str, Tensor]:
        """Hook to modify batch after transfer to device."""
        # IMPORTANT to not use torchgeo default for now
        return batch
