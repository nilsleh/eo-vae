import argparse
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from hydra.utils import instantiate
from diffusers import AutoencoderKL
from torchmetrics.functional import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
    spectral_angle_mapper,
)
from eo_vae.models.autoencoder_flux import FluxAutoencoderKL
from eo_vae.datasets.sen2naip import LATENT_STATS


def denormalize_latents(latents, model_name='eo-vae', device='cpu'):
    """Reverses the Z-Score normalization applied by the dataset to latents."""
    stats = LATENT_STATS[model_name]
    mean = stats['mean'].to(device).view(1, -1, 1, 1)
    std = stats['std'].to(device).view(1, -1, 1, 1)
    return latents * std + mean


def load_flux_vae(device):
    """Loads the original Flux Autoencoder."""
    model_name = 'black-forest-labs/FLUX.1-dev'  # Fallback to public weights
    print(f'Loading Flux VAE: {model_name}')
    try:
        vae = AutoencoderKL.from_pretrained(model_name, subfolder='vae')
        vae.to(device).eval()
        return vae
    except Exception as e:
        print(f'Failed to load Flux VAE: {e}')
        return None


def load_eo_vae(conf, device):
    """Loads EO-VAE from config/checkpoint."""
    print(f'Loading EO-VAE from {conf.vae_ckpt}')
    try:
        vae = FluxAutoencoderKL.load_from_checkpoint(conf.vae_ckpt, map_location='cpu')
    except Exception:
        vae = instantiate(conf.autoencoder)
        checkpoint = torch.load(conf.vae_ckpt, map_location='cpu')
        state_dict = (
            checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        )
        vae.load_state_dict(state_dict, strict=False)
    vae.to(device).eval()
    return vae


def process_model(model_name, config_path, ckpt_path, device, args):
    print(f'--- Processing Model: {model_name} ---')
    conf = OmegaConf.load(config_path)

    # 1. Instantiate DataModule (Specific to this model config)
    print('Setting up DataModule...')
    dm = instantiate(conf.datamodule)
    dm.setup('test')  # Use Test Set for Metrics
    dataloader = dm.test_dataloader()

    # 2. Load SR Model
    print(f'Loading SR Model from {ckpt_path}')
    sr_model = instantiate(conf.lightning_module)
    if ckpt_path:
        sr_checkpoint = torch.load(ckpt_path, map_location='cpu')
        sr_model.load_state_dict(sr_checkpoint['state_dict'])
    sr_model.to(device).eval()

    # 3. Determine Model Type & Decoder
    model_type = 'pixel'
    if 'autoencoder' in conf:
        latent_model_key = conf.datamodule.get('latent_model', 'eo-vae')
        if 'flux' in str(latent_model_key).lower():
            model_type = 'flux-vae'
            decoder_model = load_flux_vae(device)
        else:
            model_type = 'eo-vae'
            decoder_model = load_eo_vae(conf, device)
    else:
        decoder_model = None

    print(f'Identified Model Type: {model_type}')

    # 4. Helpers needed for decoding
    wvs = torch.tensor([0.665, 0.56, 0.49, 0.842], device=device)

    metrics = {'RMSE': [], 'PSNR': [], 'SSIM': [], 'SAM': []}

    # 5. Evaluation Loop
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Inputs
            lr = batch['image_lr'].to(device)
            hr_target = batch['image_hr'].to(device)

            # Ground Truth (Normalized Pixel Space)
            if 'hr_image' in batch:
                # Latent datasets usually provide 'hr_image' from npz (Normalized)
                gt_norm = batch['hr_image'].to(device)
            elif model_type == 'pixel':
                # Pixel datasets: 'image_hr' is the normalized image
                gt_norm = hr_target
            else:
                # Fallback: assume hr_target is what we have
                # If explicit hr_image missing in latent mode, we try defaults (dangerous if latent)
                if 'orig_image_hr' in batch:
                    # Some dataloaders return orig image
                    gt_norm = batch['orig_image_hr'].to(device)
                else:
                    gt_norm = hr_target

            # --- Inference and Decode ---
            if model_type == 'pixel':
                # Direct prediction: [B, C, H, W] (Normalized)
                pred_norm = sr_model.sample(lr.shape, cond=lr)

            else:
                # Latent Models: Sample & Decode
                pred_latent = sr_model.sample(x1_shape=hr_target.shape, cond=lr)

                # Decode to Normalized Pixel Space
                if model_type == 'eo-vae':
                    # We usually still need to reverse the *latent* normalization
                    # so the decoder gets the raw distribution it learned.
                    z_raw = denormalize_latents(pred_latent, 'eo-vae', device)
                    pred_norm = decoder_model.decoder(z_raw, wvs)

                elif model_type == 'flux-vae':
                    # Decode directly
                    pred_dec = decoder_model.decode(pred_latent).sample
                    pred_norm = pred_dec

            # --- Metrics (On Normalized Data) ---

            # Ensure dimensions match
            if pred_norm.shape[1] != gt_norm.shape[1]:
                min_c = min(pred_norm.shape[1], gt_norm.shape[1])
                pred_eval = pred_norm[:, :min_c]
                gt_eval = gt_norm[:, :min_c]
            else:
                pred_eval, gt_eval = pred_norm, gt_norm

            # Ensure contiguous memory for torch metrics
            pred_eval = pred_eval.contiguous()
            gt_eval = gt_eval.contiguous()

            # Data Range for PSNR/SSIM (Dynamic based on batch)
            data_range = gt_eval.max() - gt_eval.min()

            rmse_val = mean_squared_error(pred_eval, gt_eval, squared=False)
            psnr_val = peak_signal_noise_ratio(
                pred_eval, gt_eval, data_range=data_range
            )
            ssim_val = structural_similarity_index_measure(
                pred_eval, gt_eval, data_range=data_range
            )
            sam_val = spectral_angle_mapper(pred_eval, gt_eval)

            metrics['RMSE'].append(rmse_val.item())
            metrics['PSNR'].append(psnr_val.item())
            metrics['SSIM'].append(ssim_val.item())
            metrics['SAM'].append(sam_val.item())

    # Aggregate
    final_results = {k: float(np.mean(v)) for k, v in metrics.items()}
    print(f'Results for {model_name}: {final_results}')

    return final_results


def main():
    parser = argparse.ArgumentParser()
    # Input format: "Name=/path/to/config.yaml:/path/to/checkpoint.ckpt"
    parser.add_argument(
        '--models',
        nargs='+',
        required=True,
        help='List of models: Name=config_path:ckpt_path',
    )
    parser.add_argument('--output_dir', type=str, default='./results/sr-metrics')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}

    for entry in args.models:
        # Parse Entry
        try:
            name, path_str = entry.split('=')
            if ':' in path_str:
                config_path, ckpt_path = path_str.split(':')
            else:
                config_path = path_str
                # Try default ckpt location
                ckpt_path = os.path.join(
                    os.path.dirname(config_path), 'checkpoints', 'last.ckpt'
                )
                if not os.path.exists(ckpt_path):
                    ckpt_path = os.path.join(os.path.dirname(config_path), 'last.ckpt')
        except ValueError:
            print(f'Skipping invalid entry: {entry}. Format must be Name=config:ckpt')
            continue

        if not os.path.exists(config_path) or (
            ckpt_path and not os.path.exists(ckpt_path)
        ):
            print(f'Files not found for {name}: {config_path} | {ckpt_path}')
            continue

        # Process and collect results
        all_results[name] = process_model(name, config_path, ckpt_path, device, args)

    # Save all results to a single JSON file
    out_file = os.path.join(args.output_dir, 'all_metrics.json')
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=4, sort_keys=True)

    print(f'Saved all results to {out_file}')


if __name__ == '__main__':
    main()
