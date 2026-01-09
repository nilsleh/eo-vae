import argparse
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from einops import rearrange

# Import your specific normalization logic
from eo_vae.datasets.terramesh_datamodule import normalize_image, unnormalize_image

from terratorch.registry import FULL_MODEL_REGISTRY

OmegaConf.register_new_resolver('eval', eval)


def build_terramind_model(
    modality: str, base_ckpt_dir: str = '/mnt/SSD2/nils/eo-vae/checkpoints/terramind'
):
    """
    Dynamically builds the correct TerraMind model based on modality.
    """
    # 1. Infer Model Name & Checkpoint Filename
    if modality in ['S2L2A', 'S2RGB']:
        model_name = 'terramind_v1_tokenizer_s2l2a'
        ckpt_filename = 'TerraMind_Tokenizer_S2L2A.pt'
    elif modality == 'S1RTC':
        model_name = 'terramind_v1_tokenizer_s1rtc'
        ckpt_filename = 'TerraMind_Tokenizer_S1RTC.pt'
    else:
        raise ValueError(f'TerraMind not implemented for modality: {modality}')

    # 2. Build Path
    ckpt_path = os.path.join(base_ckpt_dir, ckpt_filename)

    print(f'Loading TerraMind: {model_name} | CKPT: {ckpt_path}')

    # 3. Build Model
    return FULL_MODEL_REGISTRY.build(model_name, pretrained=True, ckpt_path=ckpt_path)


def get_display_image(tensor, modality):
    """Convert raw (unnormalized) tensor to (H, W, 3) numpy image for plotting."""
    tensor = tensor.cpu().detach()
    if modality == 'S2L2A':
        # RGB indices for S2L2A: Red(3), Green(2), Blue(1). Scale DN(3000)->1.0
        img = tensor[[3, 2, 1], :, :]
        img = torch.clamp(img / 3000.0, 0, 1)
    elif modality == 'S1RTC':
        # False color: R=VV, G=VH, B=Ratio
        vv, vh = tensor[0], tensor[1]
        ratio = vh - vv
        img = torch.stack([vv, vh, ratio])
        # Robust min-max for visualization
        for c in range(3):
            mn, mx = img[c].quantile(0.02), img[c].quantile(0.98)
            img[c] = (img[c] - mn) / (mx - mn + 1e-8)
        img = torch.clamp(img, 0, 1)
    else:
        # Default fallback
        img = tensor[:3]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img.permute(1, 2, 0).numpy()


def load_model_and_config(entry, device, modality):
    """Parses 'Name=config_path:ckpt_path'."""
    name, path_str = entry.split('=')
    cfg_path, ckpt_path = path_str.split(':') if ':' in path_str else (path_str, None)

    cfg = OmegaConf.load(cfg_path)
    if 'terramind' in name.lower():
        model = build_terramind_model(modality)
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

    # Extract norm method, default to zscore if missing
    norm_method = cfg.datamodule.get('norm_method', 'zscore')
    return name, model, norm_method


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base_config',
        type=str,
        required=True,
        help='Config defining the Ground Truth data',
    )
    parser.add_argument(
        '--models', nargs='+', required=True, help="List of 'Name=config.yaml:ckpt.pt'"
    )
    parser.add_argument('--modality', type=str, default='S2L2A')
    parser.add_argument('--batch_idx', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # 1. Setup Base Data (Ground Truth)
    base_cfg = OmegaConf.load(args.base_config)
    # Force validation settings
    base_cfg.datamodule.val_collate_mode = args.modality
    base_norm_method = base_cfg.datamodule.get('norm_method', 'zscore')

    datamodule = instantiate(base_cfg.datamodule, eval_batch_size=8)
    datamodule.setup('fit')

    # Get specific batch
    loader = datamodule.val_dataloader()
    for i, batch in enumerate(loader):
        if i == args.batch_idx:
            base_images = batch['image'].to(device)
            wvs = batch['wvs'].to(device) if 'wvs' in batch else None
            break

    # 2. Inference Loop
    results = {}  # Stores RAW (unnormalized) reconstructions

    with torch.no_grad():
        # Get Ground Truth in Raw space
        gt_raw = unnormalize_image(base_images, args.modality, method=base_norm_method)
        results['Ground Truth'] = gt_raw

        for entry in args.models:
            name, model, model_norm_method = load_model_and_config(
                entry, device, modality=args.modality
            )

            # A. Bridge: Base Norm -> Raw -> Model Norm
            # If base and model use same method, skip unnorm/norm steps for speed
            if base_norm_method == model_norm_method:
                model_input = base_images
            else:
                model_input = normalize_image(
                    gt_raw, args.modality, method=model_norm_method
                )

            # B. Forward Pass
            # Detect model API (TerraMind vs EO-VAE)
            if 'TerraMind' in name:
                recon = model(model_input, timesteps=20)
            elif 'Refiner' in name:
                recon = model(model_input, wvs)
            else:
                try:
                    # Try Standard VAE encode/decode
                    z = model.encode(model_input, wvs).mode()
                    # Some VAEs require explicit normalization of latents
                    if hasattr(model, 'normalize_latent'):
                        z_shuffled = rearrange(
                            z,
                            '... c (i pi) (j pj) -> ... (c pi pj) i j',
                            pi=model.ps[0],
                            pj=model.ps[1],
                        )
                        z = model.normalize_latent(z_shuffled)
                    recon = model.decode(z, wvs)
                except:
                    # Fallback
                    recon = model(model_input, wvs)[0]

            # C. Un-normalize back to Raw Space for visual comparison
            results[name] = unnormalize_image(
                recon, args.modality, method=model_norm_method
            )

    # 3. Plotting
    B = base_images.shape[0]
    cols = len(results)
    fig, axes = plt.subplots(B, cols, figsize=(4 * cols, 4 * B))
    plt.subplots_adjust(wspace=0.05, hspace=0.1)

    # Ensure axes is 2D array
    if B == 1:
        axes = axes[None, :]
    if cols == 1:
        axes = axes[:, None]

    # Order keys so GT is last
    keys = [k for k in results.keys() if k != 'Ground Truth'] + ['Ground Truth']

    for b in range(B):
        for c, name in enumerate(keys):
            ax = axes[b, c]

            # Convert raw tensor to RGB numpy
            img_disp = get_display_image(results[name][b], args.modality)

            ax.imshow(img_disp)
            if b == 0:
                ax.set_title(name, fontsize=14, fontweight='bold')
            ax.axis('off')

    out_path = f'visual_comp_{args.modality}.png'
    plt.savefig(out_path, bbox_inches='tight')
    print(f'Visualization saved to {out_path}')


if __name__ == '__main__':
    main()
