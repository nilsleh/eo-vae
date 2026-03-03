import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchmetrics.functional import (
    mean_squared_error,
    peak_signal_noise_ratio,
    spectral_angle_mapper,
    structural_similarity_index_measure,
)
from tqdm import tqdm

from eo_vae.datasets.sen12ms_cr_dataset import SEN12MSCRDataset

OmegaConf.register_new_resolver('eval', eval)

# Hardcoded norm stats from SEN12MSCRDataset normalizer modules
_S2_MEAN = [
    2475.625, 2260.839, 2143.561, 2230.225, 2445.427,
    2992.950, 3257.843, 3171.695, 3440.958, 1567.433,
    561.076, 2562.809, 1924.178,
]
_S2_STD = [
    1761.905, 1804.267, 1661.263, 1932.020, 1918.007,
    1812.421, 1795.179, 1734.280, 1780.039, 1082.531,
    512.077, 1350.580, 1177.511,
]
_S1_MEAN = [-10.793, -17.198]
_S1_STD = [4.278, 4.346]


def reconstruct(model, images, wvs):
    """Encode and decode with the same wavelengths (identity translation)."""
    z = model.encode_to_latent(images, wvs[0])
    z_norm = model._normalize_latent(z)
    return model.decode(z_norm, wvs[0])


def _unnorm_s2(tensor):
    """Reverse S2 z-score: (C, H, W) -> raw counts."""
    device = tensor.device
    mean = torch.tensor(_S2_MEAN, device=device).view(-1, 1, 1)
    std = torch.tensor(_S2_STD, device=device).view(-1, 1, 1)
    return tensor * std + mean


def _unnorm_s1(tensor):
    """Reverse S1 z-score: (C, H, W) -> dB values."""
    device = tensor.device
    mean = torch.tensor(_S1_MEAN, device=device).view(-1, 1, 1)
    std = torch.tensor(_S1_STD, device=device).view(-1, 1, 1)
    return tensor * std + mean


def to_display(tensor, modality):
    """Convert a single (C, H, W) tensor to (H, W, 3) numpy array for plotting."""
    tensor = tensor.cpu().detach()
    if modality in ('s2', 's2_cloudy'):
        raw = _unnorm_s2(tensor)
        img = raw[[3, 2, 1], :, :]  # B04 R, B03 G, B02 B
        img = torch.clamp(img, 0, 3000) / 3000.0
    else:  # s1
        raw = _unnorm_s1(tensor)
        vv, vh = raw[0], raw[1]
        ratio = vh - vv
        img = torch.stack([vv, vh, ratio])
        for c in range(3):
            mn, mx = img[c].min(), img[c].max()
            img[c] = (img[c] - mn) / (mx - mn + 1e-8)
    return img.permute(1, 2, 0).numpy()


def visualize_batch(images, recons, modality, output_dir, batch_idx):
    """Save a B-row × 3-col figure: input | reconstruction | MSE error map."""
    B = images.shape[0]
    fig, axes = plt.subplots(B, 3, figsize=(12, 4 * B))
    if B == 1:
        axes = axes[np.newaxis, :]
    plt.subplots_adjust(wspace=0.1, hspace=0.2)

    for b in range(B):
        inp_disp = to_display(images[b], modality)
        rec_disp = to_display(recons[b], modality)
        mse_map = torch.mean((recons[b].cpu() - images[b].cpu()) ** 2, dim=0).numpy()

        axes[b, 0].imshow(inp_disp)
        axes[b, 0].set_title('Input')
        axes[b, 0].axis('off')

        axes[b, 1].imshow(rec_disp)
        axes[b, 1].set_title('Reconstruction')
        axes[b, 1].axis('off')

        im = axes[b, 2].imshow(mse_map, cmap='hot')
        axes[b, 2].set_title('MSE Error')
        axes[b, 2].axis('off')
        plt.colorbar(im, ax=axes[b, 2], fraction=0.046, pad=0.04)

    plt.suptitle(f'Reconstruction: {modality}', fontsize=14)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'viz_{modality}_batch{batch_idx}.png')
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved visualization to {save_path}')


def main():
    parser = argparse.ArgumentParser(description='SEN12MS-CR reconstruction evaluation')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/sen12ms_reconstruction')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_batches', type=int, default=0, help='0 = full split')
    parser.add_argument('--viz_batch', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    print(f'Using device: {device}')

    # Load config and model
    cfg = OmegaConf.load(args.config)
    model = instantiate(cfg.model)
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    # Dataset and loader
    dataset = SEN12MSCRDataset(root=args.data_root, split=args.split, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    total = args.num_batches if args.num_batches > 0 else len(loader)
    print(f'Evaluating {args.split} split ({total} batches, {len(dataset)} samples)...')

    modalities = ['s1', 's2', 's2_cloudy']
    metrics = {m: {'RMSE': [], 'PSNR': [], 'SSIM': [], 'SAM': []} for m in modalities}

    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader), total=total):
            if args.num_batches > 0 and i >= args.num_batches:
                break

            s1 = batch['s1'].to(device)
            s2 = batch['s2'].to(device)
            s2_cloudy = batch['s2_cloudy'].to(device)
            s1_wvs = batch['s1_wvs'].to(device)
            s2_wvs = batch['s2_wvs'].to(device)

            recon_s1 = reconstruct(model, s1, s1_wvs)
            recon_s2 = reconstruct(model, s2, s2_wvs)
            recon_s2_cloudy = reconstruct(model, s2_cloudy, s2_wvs)

            if i == args.viz_batch:
                visualize_batch(s1.cpu(), recon_s1.cpu(), 's1', args.output_dir, i)
                visualize_batch(s2.cpu(), recon_s2.cpu(), 's2', args.output_dir, i)
                visualize_batch(
                    s2_cloudy.cpu(), recon_s2_cloudy.cpu(), 's2_cloudy', args.output_dir, i
                )

            for images, recon, mod in [
                (s1, recon_s1, 's1'),
                (s2, recon_s2, 's2'),
                (s2_cloudy, recon_s2_cloudy, 's2_cloudy'),
            ]:
                pred = torch.clamp(recon, -1, 1) * 0.5 + 0.5
                gt = torch.clamp(images, -1, 1) * 0.5 + 0.5
                pred = pred.contiguous()
                gt = gt.contiguous()

                metrics[mod]['RMSE'].append(mean_squared_error(pred, gt, squared=False).item())
                metrics[mod]['PSNR'].append(
                    peak_signal_noise_ratio(pred, gt, data_range=1.0).item()
                )
                metrics[mod]['SSIM'].append(
                    structural_similarity_index_measure(pred, gt, data_range=1.0).item()
                )
                metrics[mod]['SAM'].append(spectral_angle_mapper(pred, gt).item())

    results = {'split': args.split, 'num_batches': i + 1}
    for mod in modalities:
        results[mod] = {k: float(np.mean(v)) for k, v in metrics[mod].items()}

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f'metrics_{args.split}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved metrics to {out_path}')

    # Print summary table
    print(f'\n{"Modality":<12} {"RMSE":>8} {"PSNR":>8} {"SSIM":>8} {"SAM":>8}')
    print('-' * 48)
    for mod in modalities:
        m = results[mod]
        print(f'{mod:<12} {m["RMSE"]:>8.4f} {m["PSNR"]:>8.2f} {m["SSIM"]:>8.4f} {m["SAM"]:>8.4f}')


if __name__ == '__main__':
    main()
