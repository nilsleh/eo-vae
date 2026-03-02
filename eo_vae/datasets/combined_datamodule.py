import random

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from eo_vae.datasets.satlas_naip import SatlasNAIPDataModule, _apply_batch_augmentations
from eo_vae.datasets.terramesh_datamodule import TerraMeshDataModule


class CombinedEOIterableDataset(IterableDataset):
    """Interleaves TerraMesh and NAIP batches within a single epoch.

    Epoch length equals the TerraMesh epoch. For each TerraMesh raw batch,
    a biased coin flip decides whether to yield a NAIP batch instead.

    Args:
        terramesh_dataset: TerraMesh IterableDataset yielding pre-batched raw dicts.
        naip_dataset: SatlasNAIPDataset (map-style) for random-index NAIP sampling.
        terramesh_collate_fn: Collate function applied to raw TerraMesh batches.
        naip_batch_size: Number of NAIP samples per NAIP batch.
        p_naip: Probability of yielding a NAIP batch instead of TerraMesh.
    """

    def __init__(
        self,
        terramesh_dataset,
        naip_dataset,
        terramesh_collate_fn,
        naip_batch_size,
        p_naip,
    ):
        self.terramesh_dataset = terramesh_dataset
        self.naip_dataset = naip_dataset
        self.terramesh_collate_fn = terramesh_collate_fn
        self.naip_batch_size = naip_batch_size
        self.p_naip = p_naip
        self.naip_len = len(naip_dataset)

    def _naip_collate(self, samples):
        images = torch.stack([s['image'] for s in samples])
        images = _apply_batch_augmentations(images)
        return {'image': images, 'wvs': samples[0]['wvs'], 'modality': 'NAIP'}

    def __iter__(self):
        worker_info = get_worker_info()
        seed = worker_info.seed if worker_info is not None else None
        rng = random.Random(seed)

        for raw_tm_batch in self.terramesh_dataset:
            if rng.random() < self.p_naip:
                indices = [rng.randrange(self.naip_len) for _ in range(self.naip_batch_size)]
                samples = [self.naip_dataset[i] for i in indices]
                yield self._naip_collate(samples)
            else:
                yield self.terramesh_collate_fn(raw_tm_batch)


class CombinedEODataModule(LightningDataModule):
    """DataModule that interleaves TerraMesh and Satlas NAIP batches during training.

    Validation uses TerraMesh S2L2A only (unchanged from TerraMeshDataModule).

    Args:
        terramesh: Keyword arguments for TerraMeshDataModule (excluding num_workers).
        naip: Keyword arguments for SatlasNAIPDataModule (excluding num_workers).
        p_naip: Fraction of training batches drawn from NAIP (default 0.5).
        num_workers: DataLoader worker processes shared across both sources.
    """

    def __init__(self, terramesh, naip, p_naip=0.5, num_workers=4):
        super().__init__()
        self.p_naip = p_naip
        self.num_workers = num_workers

        self.terramesh_dm = TerraMeshDataModule(**terramesh, num_workers=num_workers)
        self.naip_dm = SatlasNAIPDataModule(**naip, num_workers=num_workers)

    def setup(self, stage=None):
        self.terramesh_dm.setup(stage)
        self.naip_dm.setup(stage)

    def train_dataloader(self):
        combined_ds = CombinedEOIterableDataset(
            terramesh_dataset=self.terramesh_dm.train_dataset,
            naip_dataset=self.naip_dm.train_dataset,
            terramesh_collate_fn=self.terramesh_dm.train_collate_fn,
            naip_batch_size=self.naip_dm.batch_size,
            p_naip=self.p_naip,
        )
        return DataLoader(
            combined_ds,
            batch_size=None,
            num_workers=self.num_workers,
            collate_fn=None,
            pin_memory=True,
        )

    def val_dataloader(self):
        return self.terramesh_dm.val_dataloader()



#  data_path: "/mnt/SSD2/nils/datasets/terramesh"
#   modalities: ["S2L2A", "S1RTC", "S2RGB"]
#   batch_size: 8
#   eval_batch_size: 16
#   num_workers: 4
#   train_collate_mode: "random"
#   val_collate_mode: "S2L2A"
#   normalize: true
#   norm_method: zscore
#   norm_scheme: legacy
#   return_metadata: True
#   target_size: [256, 256]

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

# combined = CombinedEODataModule(
#     terramesh={
#         "data_path": "/mnt/SSD2/nils/datasets/terramesh",
#         "modalities": ["S2L2A", "S1RTC", "S2RGB"],
#         "batch_size": 8,
#         "eval_batch_size": 16,
#         "normalize": True,
#         "norm_method": "zscore",
#         "norm_scheme": "legacy",
#         "return_metadata": True,
#         "target_size": (256, 256),
#     },
#     naip={
#         "data_path": "/mnt/SSD2/nils/datasets/satlas_pretrain/ds/",
#         "batch_size": 16,
#         "eval_batch_size": 32,
#         "normalize": True,
#         "target_size": (256, 256),
#         "val_fraction": 0.1,
#         "seed": 42,
#     },
#     p_naip=0.25,
#     num_workers=4,
# )

# combined.setup()
# train_batch = next(iter(combined.train_dataloader()))

# import pdb
# pdb.set_trace()

# print(0)