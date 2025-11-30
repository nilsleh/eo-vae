# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SeasoNet datamodule."""

from typing import Any

import kornia.augmentation as K
import torch
from torchgeo.datamodules.geo import NonGeoDataModule
from torchgeo.datasets import SeasoNet


class SeasoNetWithWaves(SeasoNet):
    """SeasoNet dataset with waves."""

    all_bands = ('10m_RGB', '10m_IR', '20m', '60m')
    band_nums = {'10m_RGB': 3, '10m_IR': 1, '20m': 6, '60m': 2}

    # https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/bands/
    wvs = torch.tensor(
        [0.66, 0.56, 0.49, 0.842, 0.705, 0.74, 0.783, 0.865, 1.61, 2.19, 0.945, 1.375]
    )

    def __getitem__(self, index: int):
        """Return the image and waves at the given index."""
        sample = super().__getitem__(index)
        sample['wvs'] = self.wvs.float()
        # import pdb
        # pdb.set_trace()
        sample['image'] = sample['image'] / 3000
        return sample


class SeasoNetDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the SeasoNet dataset."""

    # Computed on summer season
    means = torch.tensor(
        [
            723.2311,
            700.6459,
            459.1584,
            2828.0876,
            1135.0892,
            2236.9810,
            2690.9607,
            2959.0400,
            2100.0281,
            1318.8518,
            353.0888,
            2952.8635,
        ]
    )
    stds = torch.tensor(
        [
            635.5509,
            409.1886,
            323.9882,
            1138.0142,
            653.9771,
            828.4099,
            1037.7228,
            1118.3160,
            1045.8048,
            867.0016,
            216.1276,
            1061.7631,
        ]
    )

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, size: int = 256, **kwargs: Any
    ) -> None:
        """Initialize a new SeasoNetDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            size: resize images of input size 1000x1000 to size x size
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.SeasoNet`.
        """
        if kwargs.get('bands', None) == ['10m_RGB']:
            self.mean, self.std = self.means[:3], self.stds[:3]
        else:
            self.mean, self.std = self.means[:], self.stds[:]

        # kwargs['transforms'] = K.AugmentationSequential(
        #     K.Normalize(mean=self.mean, std=self.std), data_keys=None, keepdim=True
        # )
        # add normalization to dataset level
        super().__init__(SeasoNetWithWaves, batch_size, num_workers, **kwargs)

        self.train_aug = K.AugmentationSequential(
            K.Resize(size),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=None,
            keepdim=True,
        )

        self.aug = K.AugmentationSequential(
            K.Resize(size), data_keys=None, keepdim=True
        )

        self.size = size
