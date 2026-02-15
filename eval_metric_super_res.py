import argparse
import json
import os

import numpy as np
import torch
from diffusers import AutoencoderKL
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torchmetrics.functional import (
    mean_squared_error,
    peak_signal_noise_ratio,
    spectral_angle_mapper,
    structural_similarity_index_measure,
)
from tqdm import tqdm

# from torchmetrics.image import (
# )

# Stats from encode_latents.py
HR_MEAN = torch.tensor([125.1176, 121.9117, 100.0240, 143.8500]).view(1, 4, 1, 1)
HR_STD = torch.tensor([39.8066, 30.3501, 28.9109, 28.8952]).view(1, 4, 1, 1)


def batch_denorm_rgb(img, img_mean, img_std, max_val):
    """Batch denormalization to RGB [0,1] for metrics."""
    # img: [B, C, H, W]
    if img.shape[1] == 4:
        img = img[:, :3, :, :]

        img_mean = img_mean.to(img.device, img.dtype)[:, :3, :, :]
        img_std = img_std.to(img.device, img.dtype)[:, :3, :, :]
    elif img.shape[1] == 3:
        img_mean = img_mean.to(img.device, img.dtype)[:, :3, :, :]
        img_std = img_std.to(img.device, img.dtype)[:, :3, :, :]
    else:
        img_mean = img_mean.to(img.device, img.dtype)
        img_std = img_std.to(img.device, img.dtype)

    img = img * img_std + img_mean

    # Scale
    img = img / max_val
    return torch.clamp(img, 0, 1)


def denormalize_latents(latents, dm, device='cpu'):
    """Reverse Z-score normalization: z_raw = z_norm * std + mean"""
    # Undo latent scaling first
    latents = latents / dm.train_dataset.latent_scale_factor

    if dm.train_dataset.normalize:
        mean = dm.train_dataset.hr_mean.to(device).view(1, -1, 1, 1)
        std = dm.train_dataset.hr_std.to(device).view(1, -1, 1, 1)
        return latents * std + mean
    else:
        # If we didn't normalize externally, we assume the model output is already
        # in the raw/spatial_norm latent space (depending on training setup)
        return latents


def decode_latents(model, latents, wvs, conf):
    """Smart decoding based on config settings.
    Handles 'raw' vs 'spatial_norm' decoding paths.
    """
    use_spatial_decode = os.path.basename(conf.datamodule.root) == 'eo-vae-spatial-norm'

    if use_spatial_decode and hasattr(model, 'decode_spatial_normalized'):
        # Correct path for batch-norm/spatial-norm latents
        return model.decode_spatial_normalized(latents, wvs)
    elif hasattr(model, 'decode_raw'):
        # Fast path for raw latents
        return model.decode_raw(latents, wvs)
    else:
        # Fallback (may be slow/incorrect if model expects reshaped input)
        return model.decoder(latents, wvs)


def load_flux_vae(device):
    """Loads the original Flux Autoencoder."""
    model_name = 'black-forest-labs/FLUX.2-dev'  # Fallback to public weights
    print(f'Loading Flux VAE: {model_name}')
    vae = AutoencoderKL.from_pretrained(model_name, subfolder='vae')
    vae.to(device).eval()
    return vae


def load_eo_vae(conf, device):
    """Load EO-VAE decoder."""
    print(f'Loading EO-VAE from {conf.vae_ckpt}')
    vae = instantiate(conf.autoencoder, freeze_body=True)
    checkpoint = torch.load(conf.vae_ckpt, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    vae.load_state_dict(state_dict, strict=False)
    vae.to(device).eval()
    return vae


def infer_model_type(name, conf):
    """Infer model type from name or config."""
    name_lower = name.lower()
    if 'pixel' in name_lower:
        return 'pixel'
    elif 'eo-vae' in name_lower or 'eovae' in name_lower:
        return 'eo-vae'
    elif 'flux' in name_lower:
        return 'flux-vae'

    # Fallback to config
    latent_model = conf.datamodule.get('latent_model', 'pixel')
    if 'eo-vae' in str(latent_model).lower():
        return 'eo-vae'
    elif 'flux' in str(latent_model).lower():
        return 'flux-vae'
    return 'pixel'


def process_model(model_name, config_path, ckpt_path, device, args):
    print(f'--- Processing Model: {model_name} ---')
    conf = OmegaConf.load(config_path)
    model_type = infer_model_type(model_name, conf)
    print(f'Identified Model Type: {model_type}')

    # 1. Instantiate DataModule & Get Stats (matching eval_viz)
    print('Setting up DataModule...')
    dm = instantiate(conf.datamodule, batch_size=8)
    dm.setup('test')
    dataloader = dm.test_dataloader()

    # 2. Load SR Model
    print(f'Loading SR Model from {ckpt_path}')
    sr_model = instantiate(conf.lightning_module)
    if ckpt_path:
        sr_checkpoint = torch.load(ckpt_path, map_location='cpu')
        sr_model.load_state_dict(sr_checkpoint['state_dict'])
    sr_model.to(device).eval()

    # 3. Load Decoder if needed
    decoder = None
    wvs = torch.tensor([0.665, 0.56, 0.49, 0.842], device=device)

    if model_type == 'eo-vae':
        decoder = load_eo_vae(conf, device)
    elif model_type == 'flux-vae':
        decoder = load_flux_vae(device)

    # 4. Metrics
    metrics = {'RMSE': [], 'PSNR': [], 'SSIM': [], 'SAM': []}

    # 5. Evaluation Loop
    print('Running inference and metrics...')
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Inputs
            lr = batch['image_lr'].to(device)
            hr = batch['image_hr'].to(device)

            # Prediction
            if model_type == 'pixel':
                # Pixel model prediction
                cond_in = batch['orig_image_lr'].to(device)
                pred = sr_model.sample(cond_in.shape, cond=cond_in)
            else:
                # Latent Models: Sample & Decode
                # 1. Predict Z-scored latents
                pred_latent_norm = sr_model.sample(x1_shape=hr.shape, cond=lr)

                # 2. Denormalize to raw latent space
                pred_latent_raw = denormalize_latents(pred_latent_norm, dm, device)

                # 3. Decode to pixels
                if model_type == 'eo-vae':
                    pred = decode_latents(decoder, pred_latent_raw, wvs, conf)
                else:  # flux-vae
                    pred = decoder.decode(pred_latent_raw).sample

            # --- Prepare for Metrics ---
            # Denormalize both GT and Pred to RGB [0, 1] using fixed stats
            # This matches the visualization check which is deemed "correct"

            pred_eval = batch_denorm_rgb(pred, HR_MEAN, HR_STD, 255.0)
            # if model_type == "pixel":
            #     gt_eval = batch_denorm_rgb(batch["image_hr"], HR_MEAN, HR_STD, 255.0).to(device)
            # else:
            gt_eval = batch_denorm_rgb(
                batch['orig_image_hr'], HR_MEAN, HR_STD, 255.0
            ).to(device)

            # Ensure contiguous memory for torch metrics
            pred_eval = pred_eval.contiguous()
            gt_eval = gt_eval.contiguous()

            # For PSNR/SSIM on [0,1] data, data_range is 1.0 (or dynamic max-min)
            # Since we clamped to [0,1], 1.0 is appropriate if we want standard definitions
            data_range = 1.0

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
