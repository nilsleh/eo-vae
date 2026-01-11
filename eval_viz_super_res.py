import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

# Import Models and Stats
from eo_vae.models.super_res import DiffusionSuperRes
from eo_vae.models.autoencoder_flux import FluxAutoencoderKL
from eo_vae.datasets.sen2naip import LATENT_STATS


def normalize_vis(img):
    """
    Robust normalization for visualization to [0,1].
    Handles [C, H, W] tensors.
    """
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)

    img = img.float()
    if img.ndim == 3:
        if img.shape[0] > 3:
            img = img[:3]  # Take first 3 channels (R,G,B usually)

    # Flatten spatial dims to compute quantiles per image
    img_flat = img.reshape(img.shape[0], -1)
    low = torch.quantile(img_flat, 0.02, dim=1, keepdim=True).unsqueeze(-1)
    high = torch.quantile(img_flat, 0.98, dim=1, keepdim=True).unsqueeze(-1)

    # Avoid div by zero
    scale = high - low
    scale[scale == 0] = 1e-6

    img = (img - low) / scale
    img = torch.clamp(img, 0, 1)

    # [C, H, W] -> [H, W, C]
    return img.permute(1, 2, 0).cpu().numpy()


def denormalize_latents(latents, model_name='eo-vae', device='cpu'):
    """Reverses the Z-Score normalization applied by the dataset."""
    stats = LATENT_STATS[model_name]
    mean = stats['mean'].to(device).view(1, -1, 1, 1)
    std = stats['std'].to(device).view(1, -1, 1, 1)
    return latents * std + mean


def load_models(conf, sr_ckpt_path, device):
    # 1. Load VAE
    print(f'Loading VAE from {conf.vae_ckpt}')
    try:
        # Fix: Add map_location='cpu' to avoid CUDA device mismatch errors
        vae = FluxAutoencoderKL.load_from_checkpoint(conf.vae_ckpt, map_location='cpu')
    except Exception as e:
        print(
            f'Warning: Direct PL checkpoint load failed ({e}). Trying instantiation + load_state_dict...'
        )
        # Fallback if config has enough info
        vae = instantiate(conf.autoencoder)
        checkpoint = torch.load(conf.vae_ckpt, map_location='cpu')
        state_dict = (
            checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        )
        vae.load_state_dict(state_dict, strict=False)

    vae.to(device).eval()

    # 2. Load SR Model
    print(f'Loading SR Model from {sr_ckpt_path}')
    sr_model = instantiate(conf.lightning_module)
    sr_checkpoint = torch.load(sr_ckpt_path, map_location='cpu')
    sr_model.load_state_dict(sr_checkpoint['state_dict'])
    sr_model.to(device).eval()

    return sr_model, vae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, required=True, help='Path to experiment config.yaml'
    )
    parser.add_argument(
        '--ckpt', type=str, default=None, help='Path to SR model checkpoint'
    )
    # REMOVED: parser.add_argument('--out_dir', ...)
    parser.add_argument(
        '--num_batches', type=int, default=1, help='Number of batches to visualize'
    )
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load Config
    conf = OmegaConf.load(args.config)

    # Determine Output Directory from Config
    exp_dir = conf.experiment.save_dir
    vis_dir = os.path.join(exp_dir, 'vis_results')
    os.makedirs(vis_dir, exist_ok=True)
    print(f'Saving visualizations to {vis_dir}')

    # Determine SR Checkpoint
    if args.ckpt is None:
        # Try finding 'last.ckpt' relative to config location
        base_dir = os.path.dirname(args.config)
        candidates = [
            os.path.join(base_dir, 'checkpoints', 'last.ckpt'),
            os.path.join(base_dir, 'last.ckpt'),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                args.ckpt = cand
                break

        if args.ckpt is None:
            raise FileNotFoundError(
                f'Could not find a checkpoint in {base_dir}. Please specify --ckpt.'
            )

    # Load Models
    sr_model, vae = load_models(conf, args.ckpt, device)

    # Instantiate DataModule per Config
    print('Setting up DataModule from config...')
    dm = instantiate(conf.datamodule)
    dm.setup('val')
    dataloader = dm.val_dataloader()

    # Latent Model Name (for getting stats)
    latent_model_name = conf.datamodule.get('latent_model', 'eo-vae')

    # Wavelengths for decoding (S2RGB + NIR typically for 4-channel HR)
    # Assuming standard Sen2Naip HR bands: [Red, Green, Blue, NIR]
    # Wavelengths should match what the VAE trained on for those channels
    # S2RGB wvs: [0.665, 0.56, 0.49]
    # NIR w: 0.842
    wvs = torch.tensor([0.665, 0.56, 0.49, 0.842], device=device)

    print(f'Starting visualization for {args.num_batches} batches...')

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        if batch_idx >= args.num_batches:
            break

        # Get data (Latents are Z-Score normalized by Dataset)
        lr_latent = batch['image_lr'].to(device)
        hr_latent = batch['image_hr'].to(device)

        # Get original pixels (Dataset provides these if using Sen2NaipLatentCrossSensorDataModule)
        # Note: Sen2NaipLatent returns keys 'lr_image' and 'hr_image' from .npz
        raw_lr = batch['orig_image_lr']  # .npz key from extract_latents.py
        raw_hr = batch['orig_image_hr']  # .npz key from extract_latents.py

        B = lr_latent.shape[0]

        # 1. Inference (Latent Super Res)
        with torch.no_grad():
            # Sample latent prediction [B, C, H, W]
            pred_latent = sr_model.sample(x1_shape=hr_latent.shape, cond=lr_latent)

            # 2. Denormalize Latents (Z-Score -> Raw Latent Space)
            z_pred_raw = denormalize_latents(pred_latent, latent_model_name, device)

            # 3. Decode -> Pixels
            # Manually invoke decoder parts to skip unshuffle/unnorm if using FluxAutoencoderKL
            # If FluxAutoencoderKL has a .decode() that does everything, check implementation.
            # Usually .decode() calls .post_process() which does unshuffle.
            # BUT our latents are spatial (never shuffled into channels).
            # We just need the raw decoder network.

            # FluxAutoencoderKL.decoder is usually the Decoder() module.
            if hasattr(vae, 'decoder'):
                # Ensure wvs batch size matches
                # wvs_b = wvs.repeat(B, 1)

                # The decoder module typically expects [B, z_channels, H, W] and returns [B, out_ch, H, W]
                # It does NOT handle the un-patchifying. That is handled by post_process/decode wrapper.
                # Since extract_latents saved patchified latents as spatial tensors, we are good.
                pred_pixel = vae.decoder(z_pred_raw, wvs)
            else:
                # Fallback for standard VAEs
                pred_pixel = vae.decode(z_pred_raw)

        # 4. Visualization
        # Columns: [Raw LR] [Latent LR] [Latent HR GT] [Latent Pred] [Pixel Pred] [Raw HR]

        fig, axes = plt.subplots(B, 6, figsize=(24, 4 * B))
        if B == 1:
            axes = axes[None, :]

        for i in range(B):
            # Column 1: Input LR Original (Upscale for vis)
            axes[i, 0].imshow(normalize_vis(raw_lr[i]))
            axes[i, 0].set_title('Input LR Original')
            axes[i, 0].axis('off')

            # Column 2: Latent Input (Visualizing first 3 channels)
            axes[i, 1].imshow(normalize_vis(lr_latent[i]))
            axes[i, 1].set_title('Latent Input')
            axes[i, 1].axis('off')

            # Column 3: Latent (GT)
            axes[i, 2].imshow(normalize_vis(hr_latent[i]))
            axes[i, 2].set_title('Latent (GT)')
            axes[i, 2].axis('off')

            # Column 4: Prediction (Latent)
            axes[i, 3].imshow(normalize_vis(pred_latent[i]))
            axes[i, 3].set_title('Prediction (Latent)')
            axes[i, 3].axis('off')

            # Column 5: Pixel Prediction (Decoded)
            axes[i, 4].imshow(normalize_vis(pred_pixel[i]))
            axes[i, 4].set_title('Pixel Prediction')
            axes[i, 4].axis('off')

            # Column 6: Original High Resolution Image
            axes[i, 5].imshow(normalize_vis(raw_hr[i]))
            axes[i, 5].set_title('Original HR')
            axes[i, 5].axis('off')

        out_path = os.path.join(vis_dir, f'batch_{batch_idx}.png')
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
