import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningDataModule
from omegaconf import ListConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# =============================================================================
# Constants
# =============================================================================

NAIP_WAVELENGTHS = [0.665, 0.56, 0.49, 0.842]  # R, G, B, NIR in µm


def _apply_batch_augmentations(images: torch.Tensor) -> torch.Tensor:
    """D4 symmetry group augmentations: random h/v flips and 90-degree rotations."""
    if random.random() > 0.5:
        images = torch.flip(images, dims=[-1])
    if random.random() > 0.5:
        images = torch.flip(images, dims=[-2])
    k = random.randint(0, 3)
    if k > 0:
        images = torch.rot90(images, k, dims=[-2, -1])
    return images


NAIP_MEAN = [102.7197, 112.3209,  92.9662, 140.7744]  # R, G, B, NIR
NAIP_STD = [43.1395, 32.7670, 28.5774, 48.9748]


# =============================================================================
# Dataset
# =============================================================================


class SatlasNAIPDataset(Dataset):
    """Dataset for Satlas NAIP tiles.

    Each item loads a matched (tci, ir) pair and returns a 4-channel RGBN tensor.

    Args:
        file_list: List of (tci_path, ir_path) string tuples.
        normalize: Whether to apply z-score normalization.
        target_size: Optional (H, W) to resize tiles via bilinear interpolation.
    """

    def __init__(self, file_list, normalize=True, target_size=None):
        self.file_list = file_list
        self.normalize = normalize
        self.target_size = target_size
        self.wvs = torch.FloatTensor(NAIP_WAVELENGTHS)

        if normalize:
            self.mean = torch.tensor(NAIP_MEAN).view(4, 1, 1)
            self.std = torch.tensor(NAIP_STD).view(4, 1, 1)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        tci_path, ir_path = self.file_list[idx]

        # Load TCI (RGB, uint8) → [3, H, W] float
        tci = Image.open(tci_path).convert('RGB')
        tci = torch.from_numpy(np.array(tci)).permute(2, 0, 1).float()

        # Load IR (single channel, uint8) → [1, H, W] float
        ir = Image.open(ir_path).convert('L')
        ir = torch.from_numpy(np.array(ir)).unsqueeze(0).float()

        # Stack to RGBN order → [4, H, W]
        image = torch.cat([tci, ir], dim=0)

        if self.target_size is not None:
            image = F.interpolate(
                image.unsqueeze(0),
                size=self.target_size,
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)

        if self.normalize:
            image = (image - self.mean) / self.std

        return {'image': image, 'wvs': self.wvs, 'modality': 'NAIP'}


# =============================================================================
# DataModule
# =============================================================================


class SatlasNAIPDataModule(LightningDataModule):
    """Lightning DataModule for Satlas NAIP aerial imagery.

    Scans a root directory for scenes, splits at scene level to avoid spatial
    leakage, and returns batches in the same format as TerraMeshDataModule:
    ``{'image': Tensor[B, 4, H, W], 'wvs': Tensor[4], 'modality': 'NAIP'}``.

    Directory layout expected::

        {data_path}/
          {scene_id}/
            tci/
              {x}_{y}.png   # uint8 RGB
            ir/
              {x}_{y}.png   # uint8 grayscale (NIR)

    Args:
        data_path: Root directory containing scene subdirectories.
        batch_size: Training batch size.
        eval_batch_size: Validation/test batch size.
        num_workers: DataLoader worker processes.
        normalize: Apply z-score normalization using NAIP_MEAN/NAIP_STD.
        target_size: Resize tiles to (H, W) before batching. None keeps native size.
        val_fraction: Fraction of scenes held out for validation (default 0.1).
        seed: Random seed for reproducible scene-level split.
    """

    def __init__(
        self,
        data_path,
        batch_size=16,
        eval_batch_size=32,
        num_workers=4,
        normalize=True,
        target_size=None,
        val_fraction=0.1,
        seed=42,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        self.val_fraction = val_fraction
        self.seed = seed

        if isinstance(target_size, (list, tuple, ListConfig)):
            target_size = tuple(target_size)
        self.target_size = target_size

    def setup(self, stage=None):
        root = Path(self.data_path) / "naip"

        # Discover scenes that have both tci/ and ir/ subdirectories
        scenes = sorted(
            d
            for d in root.iterdir()
            if d.is_dir() and (d / 'tci').is_dir() and (d / 'ir').is_dir()
        )

        if not scenes:
            raise ValueError(f'No valid NAIP scenes found in {self.data_path}')

        # Scene-level split to prevent spatial leakage
        rng = random.Random(self.seed)
        scenes = list(scenes)
        rng.shuffle(scenes)

        n_val = max(1, int(len(scenes) * self.val_fraction))
        val_scenes = scenes[:n_val]
        train_scenes = scenes[n_val:]

        self.train_dataset = SatlasNAIPDataset(
            self._collect_pairs(train_scenes), self.normalize, self.target_size
        )
        self.val_dataset = SatlasNAIPDataset(
            self._collect_pairs(val_scenes), self.normalize, self.target_size
        )

    def _collect_pairs(self, scenes):
        """Collect matched (tci_path, ir_path) pairs from a list of scene dirs."""
        pairs = []
        for scene_dir in scenes:
            tci_dir = scene_dir / 'tci'
            ir_dir = scene_dir / 'ir'
            for tci_path in sorted(tci_dir.glob('*.png')):
                ir_path = ir_dir / tci_path.name
                if ir_path.exists():
                    pairs.append((str(tci_path), str(ir_path)))
        return pairs

    def _train_collate(self, batch):
        images = torch.stack([b['image'] for b in batch])
        images = _apply_batch_augmentations(images)
        return {'image': images, 'wvs': batch[0]['wvs'], 'modality': 'NAIP'}

    def _eval_collate(self, batch):
        images = torch.stack([b['image'] for b in batch])
        return {'image': images, 'wvs': batch[0]['wvs'], 'modality': 'NAIP'}

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._train_collate,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._eval_collate,
            pin_memory=True,
        )

    def test_dataloader(self):
        return self.val_dataloader()






# dm = SatlasNAIPDataModule(
#     data_path='/mnt/SSD2/nils/datasets/satlas_pretrain/ds/',
#     batch_size=16,
#     eval_batch_size=32,
#     num_workers=4,
#     normalize=False,
#     target_size=(256, 256),
#     val_fraction=0.1,
#     seed=42,
# )
# dm.setup()
# train_batch = next(iter(dm.train_dataloader()))

# import pdb
# pdb.set_trace()
