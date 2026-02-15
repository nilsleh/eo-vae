"""Latent Encoding Script for Super-Resolution Training.

Pipeline Design:
================
1. Load raw Sen2NAIP data (already Z-score normalized by dataloader)
2. Encode to raw latents (32 channels) using encoder()
3. Compute running statistics on latents (on-the-fly)
4. Save raw latents to disk (unnormalized - we normalize later using computed stats)
5. After all data processed, save final statistics

Post-processing:
- Latents are saved RAW (not normalized)
- Statistics file is saved at the end
- During training, normalize latents using: (z - mean) / std
- During eval, denormalize using: z * std + mean
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

from eo_vae.datasets.sen2naip import Sen2NaipCrossSensorDataModule
from eo_vae.models.autoencoder_flux import FluxAutoencoderKL

# =============================================================================
# RUNNING STATISTICS
# =============================================================================


class RunningStatsButFast(torch.nn.Module):
    """A PyTorch module that calculates multidimensional mean and variance online
    in a numerically stable way.

    Uses the "Parallel algorithm" from:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

    Example:
        rs = RunningStatsButFast((32,), [0, 2, 3])  # For latents [B, 32, H, W]
        for batch in dataloader:
            z = encode(batch)
            rs(z)
        print(rs.mean, rs.std)
    """

    def __init__(self, shape, dims):
        """Args:
        shape: Shape of resulting mean/variance (e.g., (32,) for 32 channels)
        dims: Dimensions to reduce over (e.g., [0, 2, 3] for batch, height, width)
        """
        super(RunningStatsButFast, self).__init__()
        self.register_buffer('mean', torch.zeros(shape))
        self.register_buffer('var', torch.ones(shape))
        self.register_buffer('std', torch.ones(shape))
        self.register_buffer('count', torch.zeros(1))
        self.register_buffer('min', torch.full(shape, float('inf')))
        self.register_buffer('max', torch.full(shape, float('-inf')))
        self.dims = dims

    def update(self, x):
        with torch.no_grad():
            batch_mean = torch.mean(x, dim=self.dims)
            batch_var = torch.var(x, dim=self.dims)
            batch_min = torch.amin(x, dim=self.dims)
            batch_max = torch.amax(x, dim=self.dims)

            # Count number of elements per channel
            batch_count = 1.0
            for d in self.dims:
                batch_count *= x.shape[d]
            batch_count = torch.tensor(batch_count, dtype=torch.float, device=x.device)

            n_ab = self.count + batch_count
            m_a = self.mean * self.count
            m_b = batch_mean * batch_count
            M2_a = self.var * self.count
            M2_b = batch_var * batch_count

            delta = batch_mean - self.mean

            self.mean = (m_a + m_b) / n_ab
            self.var = (
                M2_a + M2_b + delta**2 * self.count * batch_count / (n_ab + 1e-8)
            ) / n_ab
            self.count = n_ab
            self.std = torch.sqrt(self.var + 1e-8)

            self.min = torch.minimum(self.min, batch_min)
            self.max = torch.maximum(self.max, batch_max)

    def forward(self, x):
        self.update(x)
        return x

    def get_stats_dict(self):
        """Return statistics as a dictionary for saving."""
        return {
            'mean': self.mean.cpu(),
            'std': self.std.cpu(),
            'var': self.var.cpu(),
            'min': self.min.cpu(),
            'max': self.max.cpu(),
            'count': self.count.cpu(),
        }


# =============================================================================
# MODEL LOADING
# =============================================================================


def load_eo_vae(config_path, ckpt_path, device):
    """Loads EO-VAE from config and checkpoint."""
    print(f'Loading EO-VAE from config: {config_path}')
    conf = OmegaConf.load(config_path)
    model = instantiate(conf.model)

    if ckpt_path:
        print(f'Loading EO-VAE checkpoint from {ckpt_path}')
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=False)

    model.to(device).eval()
    return model


# =============================================================================
# ENCODING FUNCTIONS
# =============================================================================


@torch.no_grad()
def encode_raw(model, img, wvs):
    """Encode image to RAW latent space (no shuffle, no BatchNorm).

    Args:
        model: EO-VAE model
        img: Input image [B, C, H, W] (already normalized by dataloader)
        wvs: Wavelength vector

    Returns:
        z_raw: Raw latent [B, 32, H/8, W/8]
    """
    if isinstance(model, FluxAutoencoderKL) or hasattr(model, 'encoder'):
        # Get encoder output (moments: mean + logvar concatenated)
        moments = model.encoder(img, wvs)
        # Extract mean (mode of the distribution)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        return mean  # [B, 32, H/8, W/8]

    raise ValueError(f'Unknown model type: {type(model)}')


@torch.no_grad()
def encode_spatial_norm(model, img, wvs):
    """Encode image to Spatially Normalized latent space.
    Uses VAE's internal frozen BN stats.

    Returns: [B, 32, H/8, W/8] normalized
    """
    if hasattr(model, 'encode_spatial_normalized'):
        return model.encode_spatial_normalized(img, wvs)
    raise ValueError('Model does not support encode_spatial_normalized method')


@torch.no_grad()
def decode_raw(model, z, wvs):
    """Decode RAW latent to image (no unshuffle, no inverse BatchNorm).

    Args:
        z: Raw latent [B, 32, H/8, W/8]
        wvs: Wavelength vector

    Returns:
        Reconstructed image [B, C, H, W]
    """
    if isinstance(model, FluxAutoencoderKL) or hasattr(model, 'decoder'):
        return model.decoder(z, wvs)

    raise ValueError(f'Unknown model type: {type(model)}')


@torch.no_grad()
def decode_spatial_norm(model, z, wvs):
    """Decode Spatially Normalized latent to image.
    Uses VAE's internal inverse BN.
    """
    if hasattr(model, 'decode_spatial_normalized'):
        return model.decode_spatial_normalized(z, wvs)
    raise ValueError('Model does not support decode_spatial_normalized method')


# =============================================================================
# VISUALIZATION
# =============================================================================


@torch.no_grad()
def visualize_reconstruction(
    model,
    dataloader,
    device,
    output_path,
    wvs_lr,
    wvs_hr,
    latent_stats_lr,
    latent_stats_hr,
    lr_img_mean,
    lr_img_std,
    hr_img_mean,
    hr_img_std,
    encode_fn=encode_raw,
    decode_fn=decode_raw,
):
    """Visualize reconstruction quality using the specified encode/decode pipeline.
    Tests: decode(encode(x)) ≈ x
    """
    print('Visualizing reconstruction...')

    try:
        batch = next(iter(dataloader))
    except StopIteration:
        print('Dataloader is empty!')
        return

    lr_img = batch['image_lr'].to(device)
    hr_img = batch['image_hr'].to(device)

    # Encode
    z_lr = encode_fn(model, lr_img, wvs_lr)
    z_hr = encode_fn(model, hr_img, wvs_hr)

    # Decode
    rec_lr = decode_fn(model, z_lr, wvs_lr)
    rec_hr = decode_fn(model, z_hr, wvs_hr)

    # Compute RMSE
    rmse_lr = (
        torch.sqrt(torch.mean((rec_lr - lr_img) ** 2, dim=(1, 2, 3))).cpu().numpy()
    )
    rmse_hr = (
        torch.sqrt(torch.mean((rec_hr - hr_img) ** 2, dim=(1, 2, 3))).cpu().numpy()
    )

    print(f'  Mean RMSE - LR: {rmse_lr.mean():.4f}, HR: {rmse_hr.mean():.4f}')

    def prep_img_for_plot(tensor, img_mean, img_std, max_val):
        """Convert normalized image tensor to plottable RGB."""
        t = tensor.clone()
        # Denormalize from Z-score
        t = t * img_std + img_mean
        # Take RGB channels
        t = t[:, :3, :, :]
        # Scale to [0, 1]
        t = t / max_val
        t = torch.clamp(t, 0, 1)
        return t.permute(0, 2, 3, 1).cpu().numpy()

    # Prepare for plotting
    lr_orig_plot = prep_img_for_plot(lr_img, lr_img_mean, lr_img_std, 3000.0)
    lr_rec_plot = prep_img_for_plot(rec_lr, lr_img_mean, lr_img_std, 3000.0)
    hr_orig_plot = prep_img_for_plot(hr_img, hr_img_mean, hr_img_std, 255.0)
    hr_rec_plot = prep_img_for_plot(rec_hr, hr_img_mean, hr_img_std, 255.0)

    n_samples = min(8, lr_img.shape[0])
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    plt.suptitle('Reconstruction Check (Latent stats verify method)', fontsize=16)

    cols = ['LR Original', 'LR Reconstructed', 'HR Original', 'HR Reconstructed']
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    for i in range(n_samples):
        axes[i, 0].imshow(lr_orig_plot[i])
        axes[i, 1].imshow(lr_rec_plot[i])
        axes[i, 2].imshow(hr_orig_plot[i])
        axes[i, 3].imshow(hr_rec_plot[i])

        row_label = f'LR RMSE: {rmse_lr[i]:.4f}\nHR RMSE: {rmse_hr[i]:.4f}'
        axes[i, 0].set_ylabel(
            row_label, rotation=0, labelpad=60, va='center', fontsize=10
        )

        for j, ax in enumerate(axes[i]):
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved reconstruction check to {output_path}')


# =============================================================================
# MAIN ENCODING LOOP
# =============================================================================


def encode_split(
    model,
    dataloader,
    output_dir,
    device,
    wvs_lr,
    wvs_hr,
    stats_lr,
    stats_hr,
    split_name,
    encode_fn=encode_raw,
):
    """Encode a single data split (train/val/test).

    Pipeline per batch:
    1. Load normalized images from dataloader
    2. Encode (raw or spatial-normalized)
    3. Update running statistics
    4. Save latents to disk
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f'\nEncoding {split_name} split...')
    print(f'  Output directory: {output_dir}')

    for batch in tqdm(dataloader, desc=split_name):
        # 1. Get normalized images from dataloader
        lr_img = batch['image_lr'].to(device)
        hr_img = batch['image_hr'].to(device)
        aois = batch['aoi']

        # 2. Encode
        z_lr = encode_fn(model, lr_img, wvs_lr)
        z_hr = encode_fn(model, hr_img, wvs_hr)

        # 3. Update running statistics
        stats_lr(z_lr)
        stats_hr(z_hr)

        # 4. Save latents to disk
        for i, aoi_id in enumerate(aois):
            np.savez_compressed(
                os.path.join(output_dir, f'{aoi_id}.npz'),
                lr_latent=z_lr[i].cpu().numpy(),  # [32, H/8, W/8]
                hr_latent=z_hr[i].cpu().numpy(),  # [32, H/8, W/8]
                lr_image=lr_img[i].cpu().numpy(),  # [4, H, W] - normalized
                hr_image=hr_img[i].cpu().numpy(),  # [4, H, W] - normalized
            )


def main():
    parser = argparse.ArgumentParser(
        description='Encode Sen2NAIP dataset to latent space'
    )
    parser.add_argument(
        '--sen2naip_root', type=str, required=True, help='Path to Sen2NAIP dataset'
    )
    parser.add_argument(
        '--config', type=str, required=True, help='Path to EO-VAE config'
    )
    parser.add_argument(
        '--ckpt', type=str, default=None, help='Path to EO-VAE checkpoint'
    )
    parser.add_argument(
        '--output_root', type=str, required=True, help='Root folder for output latents'
    )
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument(
        '--skip_visualization',
        action='store_true',
        default=False,
        help='Skip reconstruction visualization',
    )
    parser.add_argument(
        '--use_spatial_norm',
        action='store_true',
        default=False,
        help='Use VAE-based spatial normalization (32 ch, shuffled-then-normed-then-unshuffled)',
    )
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Select encoding mode
    if args.use_spatial_norm:
        print('Using SPATIAL NORMALIZED encoding (VAE-based)')
        encode_func = encode_spatial_norm
        decode_func = decode_spatial_norm
        subfolder_name = 'eo-vae-spatial-norm'
    else:
        print('Using RAW encoding (requires external stats for normalization)')
        encode_func = encode_raw
        decode_func = decode_raw
        subfolder_name = 'eo-vae-legacy'

    # ==========================================================================
    # 1. Setup Dataset
    # ==========================================================================
    print('\nSetting up Sen2Naip DataModule...')
    dm = Sen2NaipCrossSensorDataModule(
        root=args.sen2naip_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.setup('fit')

    # ==========================================================================
    # 2. Load EO-VAE
    # ==========================================================================
    model = load_eo_vae(args.config, args.ckpt, device)

    # Wavelengths for encoding (R, G, B, NIR)
    wvs_lr = torch.tensor([0.665, 0.56, 0.49, 0.842], device=device)
    wvs_hr = torch.tensor([0.665, 0.56, 0.49, 0.842], device=device)

    # Image normalization stats (for visualization only)
    lr_img_mean = torch.tensor(
        [1302.9685, 1085.2820, 764.7739, 2769.4824], device=device
    ).view(1, 4, 1, 1)
    lr_img_std = torch.tensor(
        [780.8768, 513.2825, 414.3385, 793.6396], device=device
    ).view(1, 4, 1, 1)
    hr_img_mean = torch.tensor(
        [125.1176, 121.9117, 100.0240, 143.8500], device=device
    ).view(1, 4, 1, 1)
    hr_img_std = torch.tensor([39.8066, 30.3501, 28.9109, 28.8952], device=device).view(
        1, 4, 1, 1
    )

    # ==========================================================================
    # 3. Initialize Running Statistics
    # ==========================================================================
    # Latents have shape [B, 32, H/8, W/8]
    # We want per-channel statistics, so reduce over dims [0, 2, 3]
    num_latent_channels = 32
    stats_lr = RunningStatsButFast((num_latent_channels,), dims=[0, 2, 3]).to(device)
    stats_hr = RunningStatsButFast((num_latent_channels,), dims=[0, 2, 3]).to(device)

    # ==========================================================================
    # 4. Create Output Directory
    # ==========================================================================
    os.makedirs(args.output_root, exist_ok=True)

    if not args.skip_visualization:
        visualize_reconstruction(
            model,
            dm.val_dataloader(),
            device,
            os.path.join(args.output_root, 'reconstruction_check.png'),
            wvs_lr,
            wvs_hr,
            stats_lr,
            stats_hr,
            lr_img_mean,
            lr_img_std,
            hr_img_mean,
            hr_img_std,
            encode_fn=encode_func,
            decode_fn=decode_func,
        )

    # Remove pdb in production
    # import pdb; pdb.set_trace()

    # ==========================================================================
    # 5. Encode All Splits
    # ==========================================================================
    splits = {
        'train': dm.train_dataloader(),
        'val': dm.val_dataloader(),
        'test': dm.test_dataloader(),
    }
    save_dir = os.path.join(args.output_root, subfolder_name)
    os.makedirs(save_dir, exist_ok=True)

    for split_name, loader in splits.items():
        out_dir = os.path.join(save_dir, split_name)
        encode_split(
            model=model,
            dataloader=loader,
            output_dir=out_dir,
            device=device,
            wvs_lr=wvs_lr,
            wvs_hr=wvs_hr,
            stats_lr=stats_lr,
            stats_hr=stats_hr,
            split_name=split_name,
            encode_fn=encode_func,
        )

    # ==========================================================================
    # 6. Save Statistics
    # ==========================================================================
    print('\n' + '=' * 60)
    print('LATENT STATISTICS (computed over all splits)')
    print('=' * 60)

    lr_stats = stats_lr.get_stats_dict()
    hr_stats = stats_hr.get_stats_dict()

    print('\nLR Latent Stats:')
    print(f'  Mean: [{lr_stats["mean"].min():.4f}, {lr_stats["mean"].max():.4f}]')
    print(f'  Std:  [{lr_stats["std"].min():.4f}, {lr_stats["std"].max():.4f}]')
    print(f'  Min:  [{lr_stats["min"].min():.4f}, {lr_stats["min"].max():.4f}]')
    print(f'  Max:  [{lr_stats["max"].min():.4f}, {lr_stats["max"].max():.4f}]')

    print('\nHR Latent Stats:')
    print(f'  Mean: [{hr_stats["mean"].min():.4f}, {hr_stats["mean"].max():.4f}]')
    print(f'  Std:  [{hr_stats["std"].min():.4f}, {hr_stats["std"].max():.4f}]')
    print(f'  Min:  [{hr_stats["min"].min():.4f}, {hr_stats["min"].max():.4f}]')
    print(f'  Max:  [{hr_stats["max"].min():.4f}, {hr_stats["max"].max():.4f}]')

    # Extract stats and save them as lists to json file
    stats_path = os.path.join(save_dir, 'latent_stats.json')
    stats_to_save = {
        'lr_latent': {k: v.tolist() for k, v in lr_stats.items()},
        'hr_latent': {k: v.tolist() for k, v in hr_stats.items()},
    }
    import json

    with open(stats_path, 'w') as f:
        json.dump(stats_to_save, f, indent=4)
    print(f'\nSaved latent statistics to {stats_path}')

    # also save model config for reference
    conf_out_path = os.path.join(save_dir, 'model_config.yaml')
    OmegaConf.save(OmegaConf.load(args.config), conf_out_path)
    print(f'Saved model config to {conf_out_path}')

    # Print for easy copy-paste into code
    print('\n' + '=' * 60)
    print('COPY-PASTE INTO YOUR CODE:')
    print('=' * 60)
    print('LATENT_STATS_LR = {')
    print(f"    'mean': torch.tensor({lr_stats['mean'].tolist()}),")
    print(f"    'std': torch.tensor({lr_stats['std'].tolist()}),")
    print('}')
    print('\nLATENT_STATS_HR = {')
    print(f"    'mean': torch.tensor({hr_stats['mean'].tolist()}),")
    print(f"    'std': torch.tensor({hr_stats['std'].tolist()}),")
    print('}')

    # ==========================================================================
    # 7. Visualization (optional)
    # ==========================================================================
    if not args.skip_visualization:
        visualize_reconstruction(
            model,
            dm.val_dataloader(),
            device,
            os.path.join(args.output_root, 'reconstruction_check.png'),
            wvs_lr,
            wvs_hr,
            stats_lr,
            stats_hr,
            lr_img_mean,
            lr_img_std,
            hr_img_mean,
            hr_img_std,
        )

    # ==========================================================================
    # Summary
    # ==========================================================================
    print('\n' + '=' * 60)
    print('ENCODING COMPLETE')
    print('=' * 60)
    print(f'Latents saved to: {save_dir}/')
    print(f'Statistics saved to: {stats_path}')
    print('\nLatent format:')
    print('  - Shape: [32, H/8, W/8]')
    print('  - Type: RAW (not normalized)')
    print('\nTo normalize during training:')
    print(
        "  z_norm = (z_raw - stats['mean'].view(1,-1,1,1)) / stats['std'].view(1,-1,1,1)"
    )
    print('\nTo denormalize during eval:')
    print(
        "  z_raw = z_norm * stats['std'].view(1,-1,1,1) + stats['mean'].view(1,-1,1,1)"
    )


if __name__ == '__main__':
    main()
