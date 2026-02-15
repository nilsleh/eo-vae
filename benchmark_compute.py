"""Single-model benchmarking script. Run this separately for each model
to ensure complete CUDA isolation between benchmarks.

Usage:
    python benchmark_single.py --name "Model Name" --config path/to/config.yaml --ckpt path/to/checkpoint.ckpt --gpu 0
"""

import argparse
import json
import os

import numpy as np
import torch
from diffusers import AutoencoderKL
from hydra.utils import instantiate
from omegaconf import OmegaConf

from eo_vae.datasets.sen2naip import LATENT_STATS


def count_parameters(model):
    if model is None:
        return 0
    return sum(p.numel() for p in model.parameters())


def denormalize_latents(latents, device):
    stats = LATENT_STATS['eo-vae']
    mean = stats['mean'].to(device).view(1, -1, 1, 1)
    std = stats['std'].to(device).view(1, -1, 1, 1)
    return latents * std + mean


def main():
    parser = argparse.ArgumentParser(description='Benchmark a single model')
    parser.add_argument('--name', type=str, required=True, help='Model name')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--ckpt', type=str, default='', help='Path to checkpoint')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_iterations', type=int, default=50)
    parser.add_argument('--output', type=str, default='', help='Output JSON file path')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')

    print(f'{"=" * 80}')
    print(f'Benchmarking: {args.name}')
    print(f'{"=" * 80}')

    # Load config and determine model type
    conf = OmegaConf.load(args.config)
    latent_model_key = conf.datamodule.get('latent_model', 'pixel')

    if 'flux' in str(latent_model_key).lower():
        model_type = 'flux-vae'
    elif 'eo-vae' in str(latent_model_key).lower():
        model_type = 'eo-vae'
    else:
        model_type = 'pixel'

    print(f'Model type: {model_type}')
    print(f'Device: {device}')

    # Setup data
    print('Loading data...')
    dm = instantiate(conf.datamodule, num_workers=0, batch_size=args.batch_size)
    dm.setup('fit')
    dm.setup('test')
    sample_batch = next(iter(dm.test_dataloader()))

    # Get input samples
    if 'orig_image_lr' in sample_batch:
        lr_sample = sample_batch['orig_image_lr'].to(device)
        hr_sample = sample_batch['orig_image_hr'].to(device)
    else:
        lr_sample = sample_batch['image_lr'].to(device)
        hr_sample = sample_batch['image_hr'].to(device)

    print(f'Input shape: {list(lr_sample.shape)}')
    print(f'Output shape: {list(hr_sample.shape)}')

    # Load SR model
    print('Loading SR model...')
    sr_model = instantiate(conf.lightning_module)
    if args.ckpt and os.path.exists(args.ckpt):
        checkpoint = torch.load(args.ckpt, map_location='cpu')
        sr_model.load_state_dict(checkpoint['state_dict'])
    sr_model.to(device).eval()
    sr_params = count_parameters(sr_model)
    print(f'SR model parameters: {sr_params:,}')

    # Load VAE if needed
    encoder_model = None
    decoder_model = None
    wvs = None
    encoder_params = 0
    decoder_params = 0
    latent_channels = None

    if model_type == 'flux-vae':
        print('Loading Flux VAE...')
        vae = AutoencoderKL.from_pretrained(
            'black-forest-labs/FLUX.2-dev', subfolder='vae'
        )
        vae.to(device).eval()
        encoder_model = vae
        decoder_model = vae
        total_vae = count_parameters(vae)
        encoder_params = total_vae // 2
        decoder_params = total_vae // 2
        latent_channels = vae.config.latent_channels
        print(f'Flux VAE parameters: {total_vae:,}')

    elif model_type == 'eo-vae':
        print('Loading EO-VAE...')
        vae = instantiate(conf.autoencoder)
        checkpoint = torch.load(conf.vae_ckpt, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        vae.load_state_dict(state_dict, strict=False)
        vae.to(device).eval()
        encoder_model = vae
        decoder_model = vae
        encoder_params = count_parameters(vae.encoder)
        decoder_params = count_parameters(vae.decoder)
        latent_channels = 16
        wvs = torch.tensor([0.665, 0.56, 0.49, 0.842], device=device)
        print(
            f'EO-VAE parameters: encoder={encoder_params:,}, decoder={decoder_params:,}'
        )

    total_params = sr_params + encoder_params + decoder_params
    print(f'Total parameters: {total_params:,}')

    # ===== WARMUP =====
    print('\nWarming up (5 iterations)...')
    with torch.no_grad():
        for _ in range(5):
            if model_type == 'pixel':
                _ = sr_model.sample(lr_sample.shape, cond=lr_sample)
            else:
                # Encode
                if model_type == 'flux-vae':
                    lr_latent = encoder_model.encode(lr_sample).latent_dist.sample()
                else:
                    lr_latent = encoder_model.encode(lr_sample, wvs).mode()
                    stats = LATENT_STATS['eo-vae']
                    mean = stats['mean'].to(device).view(1, -1, 1, 1)
                    std = stats['std'].to(device).view(1, -1, 1, 1)
                    lr_latent = (lr_latent - mean) / std

                # SR
                pred_latent = sr_model.sample(x1_shape=lr_latent.shape, cond=lr_latent)

                # Decode
                if model_type == 'flux-vae':
                    _ = decoder_model.decoder(pred_latent)
                else:
                    z_raw = denormalize_latents(pred_latent, device)
                    _ = decoder_model.decoder(z_raw, wvs)

            torch.cuda.synchronize(device)

    # ===== BENCHMARK =====
    print(f'Running {args.n_iterations} timed iterations...')

    # Use CUDA events for accurate timing
    start_evt = torch.cuda.Event(enable_timing=True)
    enc_evt = torch.cuda.Event(enable_timing=True)
    sr_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    encode_times = []
    sr_times = []
    decode_times = []
    peak_mems = []

    with torch.no_grad():
        for i in range(args.n_iterations):
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

            if model_type == 'pixel':
                start_evt.record()
                _ = sr_model.sample(lr_sample.shape, cond=lr_sample)
                end_evt.record()
                torch.cuda.synchronize(device)

                encode_times.append(0)
                sr_times.append(start_evt.elapsed_time(end_evt))
                decode_times.append(0)
            else:
                # Encode
                start_evt.record()
                if model_type == 'flux-vae':
                    lr_latent = encoder_model.encode(lr_sample).latent_dist.sample()
                else:
                    lr_latent = encoder_model.encode(lr_sample, wvs).mode()
                    stats = LATENT_STATS['eo-vae']
                    mean = stats['mean'].to(device).view(1, -1, 1, 1)
                    std = stats['std'].to(device).view(1, -1, 1, 1)
                    lr_latent = (lr_latent - mean) / std
                enc_evt.record()

                # SR
                pred_latent = sr_model.sample(x1_shape=lr_latent.shape, cond=lr_latent)
                sr_evt.record()

                # Decode
                if model_type == 'flux-vae':
                    _ = decoder_model.decoder(pred_latent)
                else:
                    z_raw = denormalize_latents(pred_latent, device)
                    _ = decoder_model.decoder(z_raw, wvs)
                end_evt.record()

                torch.cuda.synchronize(device)

                encode_times.append(start_evt.elapsed_time(enc_evt))
                sr_times.append(enc_evt.elapsed_time(sr_evt))
                decode_times.append(sr_evt.elapsed_time(end_evt))

            peak_mems.append(torch.cuda.max_memory_allocated(device) / 1e9)

            if (i + 1) % 10 == 0:
                print(f'  Iteration {i + 1}/{args.n_iterations}')

    # Calculate averages (skip first 2 as additional warmup)
    avg_encode = float(np.mean(encode_times[2:]))
    avg_sr = float(np.mean(sr_times[2:]))
    avg_decode = float(np.mean(decode_times[2:]))
    avg_total = avg_encode + avg_sr + avg_decode
    avg_mem = float(np.mean(peak_mems[2:]))
    throughput = (args.batch_size * 1000) / avg_total if avg_total > 0 else 0

    # Print results
    print(f'\n{"=" * 80}')
    print(f'RESULTS: {args.name}')
    print(f'{"=" * 80}')
    print(f'Encode:     {avg_encode:.2f} ms')
    print(f'SR Forward: {avg_sr:.2f} ms')
    print(f'Decode:     {avg_decode:.2f} ms')
    print(f'Total:      {avg_total:.2f} ms')
    print(f'Throughput: {throughput:.2f} imgs/sec')
    print(f'Peak Memory: {avg_mem:.3f} GB')

    # Build result dict
    result = {
        'name': args.name,
        'model_type': model_type,
        'architecture': {
            'input_shape': list(lr_sample.shape),
            'output_shape': list(hr_sample.shape),
            'latent_channels': latent_channels,
            'compression_ratio': '64:1' if model_type != 'pixel' else '1:1',
        },
        'parameters': {
            'sr_model': sr_params,
            'encoder': encoder_params,
            'decoder': decoder_params,
            'total': total_params,
        },
        'memory_gb': {'peak_memory': avg_mem},
        'timing_ms': {
            'encode': avg_encode,
            'sr_forward': avg_sr,
            'decode': avg_decode,
            'total': avg_total,
        },
        'throughput_imgs_per_sec': throughput,
    }

    # Save to file if output path provided
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'\nSaved results to {args.output}')

    # Also print JSON to stdout for easy parsing
    print(f'\nJSON_RESULT:{json.dumps(result)}')


if __name__ == '__main__':
    main()
