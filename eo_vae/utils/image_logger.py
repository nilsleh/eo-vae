import os

import matplotlib.pyplot as plt
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only

WAVELENGTHS = {
    'S2RGB': [0.665, 0.56, 0.49],  # R, G, B
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

RGB_INDICES = {
    'S2RGB': [0, 1, 2],  # Already RGB
    'S2L2A': [3, 2, 1],  # R: 0.665 (B04), G: 0.560 (B03), B: 0.490 (B02)
    'S2L1C': [3, 2, 1],  # Same as S2L2A
}


class ImageLogger(Callback):
    def __init__(self, max_images=8, save_dir='images'):
        super().__init__()
        self.max_images = max_images
        self.save_dir = save_dir

    @rank_zero_only
    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        """Called automatically by Lightning when the validation loop starts.
        We strictly intercept the very first batch (batch_idx == 0).
        """
        if batch_idx == 0:
            self.log_local(
                self.save_dir,
                batch,
                pl_module,
                global_step=trainer.global_step,
                split='val',
            )

    def log_local(self, save_dir, batch, pl_module, global_step, split='val'):
        root = os.path.join(save_dir, 'image_log', split)
        os.makedirs(root, exist_ok=True)

        # No need for pl_module.eval() or torch.no_grad() context managers here;
        # Lightning puts the model in eval mode and disables grads during the val loop automatically.

        # images = pl_module.get_input(batch, pl_module.image_key)
        images = batch['image']
        wvs = batch['wvs']
        modality = batch.get('modality', 'S2RGB')

        # 1. Forward Pass
        with torch.no_grad():
            reconstruction = pl_module(
                images,
                wvs,
                # use_refiner=True if pl_module.training_mode == 'flow-refine' else False,
            )
            if isinstance(reconstruction, tuple):
                reconstruction = reconstruction[0]

        # 2. Slice batch
        N = min(images.shape[0], self.max_images)
        inputs = images[:N]
        recons = reconstruction[:N]

        # 3. Compute MSE on Raw Z-Score Data
        diff = torch.abs(inputs - recons).mean(dim=1, keepdim=True)

        # 4. Un-normalize to Physical Units
        # Access stats from the active trainer/datamodule
        dm = pl_module.trainer.datamodule

        if hasattr(dm, 'norm_stats') and modality in dm.norm_stats:
            stats = dm.norm_stats[modality]
            # Move stats to same device as images
            mean = torch.tensor(stats['mean'], device=pl_module.device).view(
                1, -1, 1, 1
            )
            std = torch.tensor(stats['std'], device=pl_module.device).view(1, -1, 1, 1)

            inputs_phys = inputs * std + mean
            recons_phys = recons * std + mean
        else:
            inputs_phys = inputs
            recons_phys = recons

        # 5. Helper: Physical -> Visual [0, 1]
        def to_vis(x, rgb_indices):
            # Select RGB channels
            x = x[:, rgb_indices, :, :]
            b, c, h, w = x.shape
            x_flat = x.view(b, c, -1)

            vis_batch = []
            for i in range(b):
                img = x_flat[i]
                # Robust scaling (2% - 98%) per image
                low = torch.quantile(img, 0.02)
                high = torch.quantile(img, 0.98)
                img_norm = (x[i] - low) / (high - low + 1e-5)
                vis_batch.append(torch.clamp(img_norm, 0, 1))

            return torch.stack(vis_batch)

        rgb_indices = RGB_INDICES.get(
            modality, [0, 1, 2]
        )  # Default to first 3 if unknown
        inputs_vis = to_vis(inputs_phys, rgb_indices)
        recons_vis = to_vis(recons_phys, rgb_indices)

        # Normalize Error Map (use mean diff across selected channels)
        diff = torch.abs(
            inputs_phys[:, rgb_indices, :, :] - recons_phys[:, rgb_indices, :, :]
        ).mean(dim=1, keepdim=True)
        diff_vis = (diff - diff.min()) / (diff.max() - diff.min() + 1e-5)
        diff_vis = diff_vis.repeat(1, 3, 1, 1)

        # 6. Grid Construction
        rows = []
        for i in range(N):
            rows.append(torch.cat((inputs_vis[i], recons_vis[i], diff_vis[i]), dim=2))

        grid = torch.cat(rows, dim=1)

        # 7. Plot & Save
        filename = f'global_step_{global_step:04}_{modality}.png'
        path = os.path.join(root, filename)

        grid_np = grid.permute(1, 2, 0).cpu().numpy()

        plt.figure(figsize=(10, N * 3))
        plt.imshow(grid_np)
        plt.axis('off')
        plt.title(
            f'Global Step: {global_step} | Modality: {modality}\nInput (Phys) | Recon (Phys) | MSE (Z-Score)'
        )
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
