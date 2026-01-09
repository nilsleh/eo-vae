import argparse
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from hydra.utils import instantiate
from omegaconf import OmegaConf
from einops import rearrange
from torchmetrics.functional import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
    spectral_angle_mapper,
)

# Import your specific normalization logic
from eo_vae.datasets.terramesh_datamodule import normalize_image, unnormalize_image
from terratorch.registry import FULL_MODEL_REGISTRY

OmegaConf.register_new_resolver('eval', eval)


# --- HELPER: Evaluation Space Standardization ---
def convert_space(img, modality, src_method, tgt_method):
    """
    Converts an image tensor from one normalization space to another.
    Critical for comparing Robust (clamped) vs Z-Score (unclamped) models.
    """
    if src_method == tgt_method:
        return img

    # 1. Unnormalize to Raw (Physical Space)
    # Note: If src_method is 'robust', this 'raw' will still be clamped/lossy,
    # but that is the desired behavior for a fair comparison against a robust baseline.
    raw = unnormalize_image(img, modality, method=src_method)

    # 2. Normalize to Target Space
    norm = normalize_image(raw, modality, method=tgt_method)
    return norm


# --- HELPER: Spectral Density ---
def compute_radial_psd(img_tensor):
    """Computes PSD on the standardized image space."""
    imgs = img_tensor.detach().cpu().numpy()
    imgs = imgs.reshape(-1, imgs.shape[-2], imgs.shape[-1])
    psd_list = []

    for img in imgs:
        h, w = img.shape
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        power_spectrum = np.abs(fshift) ** 2

        y, x = np.indices((h, w))
        center = np.array([(h - 1) / 2, (w - 1) / 2])
        r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2).astype(int)

        tbin = np.bincount(r.ravel(), power_spectrum.ravel())
        nr = np.bincount(r.ravel())
        radial_profile = tbin / (nr + 1e-8)
        psd_list.append(radial_profile)

    return np.mean(np.array(psd_list), axis=0)


# --- MODEL LOADING ---
def build_terramind_model(
    modality: str, base_ckpt_dir: str = '/mnt/SSD2/nils/eo-vae/checkpoints/terramind'
):
    if modality in ['S2L2A', 'S2RGB']:
        model_name = 'terramind_v1_tokenizer_s2l2a'
        ckpt_filename = 'TerraMind_Tokenizer_S2L2A.pt'
    elif modality == 'S1RTC':
        model_name = 'terramind_v1_tokenizer_s1rtc'
        ckpt_filename = 'TerraMind_Tokenizer_S1RTC.pt'
    else:
        raise ValueError(f'TerraMind not implemented for modality: {modality}')

    ckpt_path = os.path.join(base_ckpt_dir, ckpt_filename)
    print(f'Loading TerraMind: {model_name} | CKPT: {ckpt_path}')
    return FULL_MODEL_REGISTRY.build(model_name, pretrained=True, ckpt_path=ckpt_path)


def load_model_and_config(entry, device, modality):
    name, path_str = entry.split('=')
    cfg_path, ckpt_path = path_str.split(':') if ':' in path_str else (path_str, None)

    cfg = OmegaConf.load(cfg_path)

    if 'terramind' in name.lower():
        model = build_terramind_model(modality)
        if 'datamodule' not in cfg:
            cfg.datamodule = {}
        cfg.datamodule.norm_method = 'zscore'
    else:
        model = instantiate(cfg.model)
        if ckpt_path:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            state_dict = (
                checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            )
            model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    return name, model, cfg.datamodule.get('norm_method', 'zscore')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_config', type=str, required=True)
    parser.add_argument('--models', nargs='+', required=True, help='Name=config:ckpt')
    parser.add_argument('--modality', type=str, default='S2L2A')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_batches', type=int, default=10)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Base Data Loader (Defines the "Evaluation Space")
    base_cfg = OmegaConf.load(args.base_config)
    base_cfg.datamodule.val_collate_mode = args.modality

    # This is the normalization method of the Ground Truth we load
    # Likely 'robust' or 'minmax' for EO-VAE configs
    # eval_space_method = base_cfg.datamodule.get('norm_method', 'zscore')
    eval_space_method = 'zscore'
    print(f'Evaluation Space set to: {eval_space_method}')

    datamodule = instantiate(base_cfg.datamodule, eval_batch_size=16, num_workers=4)
    datamodule.setup('fit')
    loader = datamodule.val_dataloader()

    # 2. Load Models
    models = []
    for entry in args.models:
        models.append(load_model_and_config(entry, device, modality=args.modality))

    # 3. Containers
    metrics = {
        name: {'RMSE': [], 'PSNR': [], 'SSIM': [], 'SAM': []} for name, _, _ in models
    }
    psd_data = {'GT': []}
    for name, _, _ in models:
        psd_data[name] = []

    print(f'Starting evaluation loop...')

    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader), total=args.num_batches):
            if i >= args.num_batches:
                break

            # Ground Truth in Evaluation Space (e.g., Robust Normalized)
            gt_eval = batch['image'].to(device)
            wvs = batch['wvs'].to(device) if 'wvs' in batch else None

            # Track GT PSD (in Eval Space)
            psd_data['GT'].append(compute_radial_psd(gt_eval))

            for name, model, model_norm_method in models:
                # --- A. PREPARE INPUT ---
                # We must convert the Eval Space GT to the Model's expected Input Space
                # E.g., Robust GT -> Raw Clamped -> Z-Score Input
                inp = convert_space(
                    gt_eval,
                    args.modality,
                    src_method=eval_space_method,
                    tgt_method=model_norm_method,
                )

                # --- B. INFERENCE ---
                if 'TerraMind' in name:
                    recon_native = model(inp, timesteps=20)
                elif 'Refiner' in name:
                    recon_native = model(inp, wvs)
                else:
                    try:
                        z = model.encode(inp, wvs).mode()
                        if hasattr(model, 'normalize_latent'):
                            z_shuffled = rearrange(
                                z,
                                '... c (i pi) (j pj) -> ... (c pi pj) i j',
                                pi=model.ps[0],
                                pj=model.ps[1],
                            )
                            z = model.normalize_latent(z_shuffled)
                        recon_native = model.decode(z, wvs)
                    except:
                        recon_native = model(inp, wvs)[0]

                # --- C. STANDARDIZE OUTPUT ---
                # Convert Model Output back to Evaluation Space
                # E.g., Z-Score Output -> Raw -> Robust Eval Space
                recon_eval = convert_space(
                    recon_native,
                    args.modality,
                    src_method=model_norm_method,
                    tgt_method=eval_space_method,
                )

                # Track Model PSD
                psd_data[name].append(compute_radial_psd(recon_eval))

                # --- D. COMPUTE METRICS ---
                # Now both `gt_eval` and `recon_eval` are in the SAME distribution (e.g., Robust).
                # We do not unnormalize to raw. We compute metrics here.

                # For safety, ensure we are contiguous
                gt_eval = gt_eval.contiguous()
                recon_eval = recon_eval.contiguous()

                # Since we are in Robust space (often 0-1), we can compute metrics directly.
                # If Robust space is not 0-1 (e.g. -1 to 1), we still compare them directly,
                # but we set data_range appropriately.

                # Auto-detect data range from GT batch (robust approach)
                current_range = gt_eval.max() - gt_eval.min()

                rmse = mean_squared_error(recon_eval, gt_eval, squared=False)
                psnr = peak_signal_noise_ratio(
                    recon_eval, gt_eval, data_range=current_range
                )
                ssim = structural_similarity_index_measure(
                    recon_eval, gt_eval, data_range=current_range
                )
                sam = spectral_angle_mapper(recon_eval, gt_eval)

                metrics[name]['RMSE'].append(rmse.item())
                metrics[name]['PSNR'].append(psnr.item())
                metrics[name]['SSIM'].append(ssim.item())
                metrics[name]['SAM'].append(sam.item())

    # 4. Save Metrics
    final_results = {}
    for name in metrics:
        if len(metrics[name]['RMSE']) > 0:
            final_results[name] = {
                k: float(np.mean(v)) for k, v in metrics[name].items()
            }
            print(f'[{name}] {final_results[name]}')

    with open(os.path.join(args.output_dir, f'metrics_{args.modality}.json'), 'w') as f:
        json.dump(final_results, f, indent=4)

    # 5. --- PLOT POWER SPECTRAL DENSITY ---
    if len(psd_data['GT']) > 0:
        print('Generating PSD Plot...')

        means = {k: np.mean(np.array(v), axis=0) for k, v in psd_data.items()}
        gt_curve = means['GT']
        freqs = np.linspace(0, 0.5, len(gt_curve))

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]}
        )

        # Plot Absolute
        ax1.plot(
            freqs,
            gt_curve,
            color='black',
            linestyle='--',
            linewidth=2,
            label=f'Ground Truth ({eval_space_method})',
        )
        for name, curve in means.items():
            if name == 'GT':
                continue
            ax1.plot(freqs, curve, label=name, alpha=0.8)

        ax1.set_ylabel('Power Spectrum')
        ax1.set_yscale('log')
        ax1.grid(True, which='both', alpha=0.2)
        ax1.legend()
        ax1.set_title(f'PSD in Common Evaluation Space: {eval_space_method}')

        # Plot Ratio
        ax2.axhline(1.0, color='black', linestyle='--', linewidth=1.5)
        for name, curve in means.items():
            if name == 'GT':
                continue
            valid_mask = gt_curve > 1e-10
            ratio = np.ones_like(curve)
            ratio[valid_mask] = curve[valid_mask] / gt_curve[valid_mask]
            ax2.plot(freqs, ratio, label=name, alpha=0.8)

        ax2.set_ylabel('Ratio (Model / GT)')
        ax2.set_xlabel('Spatial Frequency (cycles/pixel)')
        ax2.set_ylim(0.1, 10.0)
        ax2.set_yscale('log')
        ax2.grid(True, which='both', alpha=0.2)

        plt.tight_layout()
        plt.savefig(
            os.path.join(args.output_dir, f'psd_plot_{args.modality}.png'), dpi=300
        )
        plt.close()


if __name__ == '__main__':
    main()
