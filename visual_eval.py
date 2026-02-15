import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from terratorch.registry import FULL_MODEL_REGISTRY
from tqdm import tqdm  # Added for progress bar

from eo_vae.datasets.terramesh_datamodule import NormalizerFactory

OmegaConf.register_new_resolver('eval', eval, replace=True)


def get_norm_scheme(cfg) -> str:
    """Extract normalization scheme from config, defaulting to 'legacy'."""
    return cfg.datamodule.get('norm_scheme', 'legacy')


def unnormalize(img, modality, scheme, device):
    """Unnormalize image from normalized space back to raw physical units."""
    normalizer = NormalizerFactory.create(modality, scheme).to(device)
    return img * normalizer.std + normalizer.mean


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


def get_display_image(tensor, modality):
    """Convert raw (unnormalized) tensor to (H, W, 3) numpy image for plotting."""
    tensor = tensor.cpu().detach()

    def percentile_norm(x, p_min=2, p_max=98):
        mn, mx = np.percentile(x, p_min), np.percentile(x, p_max)
        return (x - mn) / (mx - mn + 1e-16)

    if modality == 'S2L2A':
        img = tensor[[3, 2, 1], :, :]  # RGB bands
        return torch.clamp(img / 4000.0, 0, 1).permute(1, 2, 0).numpy()
    elif modality == 'S1RTC':
        t_np = tensor.numpy()
        vv = percentile_norm(t_np[0]) + 1e-16
        vh = percentile_norm(t_np[1]) + 1e-16
        return np.clip(np.stack([vv, vh, vv / vh], axis=-1), 0, 1)
    else:
        img = tensor[:3]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img.permute(1, 2, 0).numpy()


def load_model_and_config(entry, device, modality):
    """Load model and return (name, model, norm_scheme, cfg)."""
    name, path_str = entry.split('=')
    cfg_path, ckpt_path = path_str.split(':') if ':' in path_str else (path_str, None)

    print('Loading model:', name)
    cfg = OmegaConf.load(cfg_path)
    if 'terramind' in name.lower():
        model = build_terramind_model(modality)
        norm_scheme = 'legacy'
    else:
        model = instantiate(cfg.model)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(
                f'  WARNING: Missing keys: {missing[:5]}...'
                if len(missing) > 5
                else f'  WARNING: Missing keys: {missing}'
            )
        if unexpected:
            print(
                f'  WARNING: Unexpected keys: {unexpected[:5]}...'
                if len(unexpected) > 5
                else f'  WARNING: Unexpected keys: {unexpected}'
            )
        norm_scheme = get_norm_scheme(cfg)

        # Debug: Check BatchNorm stats if present
        if hasattr(model, 'bn'):
            print(
                f'  BN running_mean range: [{model.bn.running_mean.min():.3f}, {model.bn.running_mean.max():.3f}]'
            )
            print(
                f'  BN running_var range: [{model.bn.running_var.min():.3f}, {model.bn.running_var.max():.3f}]'
            )

    if name == 'EO-VAE-custom':
        print("State dict keys containing 'bn':")
        print([k for k in state_dict.keys() if 'bn' in k.lower()])

    model.to(device)
    model.eval()
    return name, model, norm_scheme, cfg


def create_dataloader(cfg, modality, eval_batch_size=2):
    """Create a dataloader with the config's normalization scheme."""
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.datamodule.val_collate_mode = modality
    datamodule = instantiate(cfg.datamodule, eval_batch_size=eval_batch_size)
    datamodule.setup('fit')
    return datamodule.test_dataloader()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_config', type=str, required=True)
    parser.add_argument(
        '--models', nargs='+', required=True, help="List of 'Name=config.yaml:ckpt.pt'"
    )
    parser.add_argument('--modality', type=str, default='S2L2A')
    parser.add_argument('--gpu', type=int, default=0)

    # New arguments for batch processing
    parser.add_argument(
        '--num_batches', type=int, default=100, help='Number of batches to visualize'
    )
    parser.add_argument(
        '--out_dir', type=str, default='visual_results', help='Directory to save images'
    )

    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load all models first
    print('Loading models:')
    models = [load_model_and_config(e, device, args.modality) for e in args.models]
    for name, _, scheme, _ in models:
        print(f'  {name}: norm_scheme={scheme}')

    # 2. Group models by norm_scheme
    scheme_to_models = {}
    for name, model, scheme, cfg in models:
        if scheme not in scheme_to_models:
            scheme_to_models[scheme] = {'models': [], 'cfg': cfg}
        scheme_to_models[scheme]['models'].append((name, model))

    # 3. Initialize Iterators for each scheme
    # We do this once to keep dataloaders persistent across the loop
    scheme_iterators = {}
    print('Initializing dataloaders...')
    for scheme, group in scheme_to_models.items():
        loader = create_dataloader(group['cfg'], args.modality)
        scheme_iterators[scheme] = iter(loader)

    # 4. Loop through batches
    print(f'Starting visualization of {args.num_batches} batches...')

    for batch_idx in tqdm(range(args.num_batches)):
        results = {}  # name -> raw tensor

        # Run inference for each scheme group
        for scheme, group in scheme_to_models.items():
            try:
                batch = next(scheme_iterators[scheme])
            except StopIteration:
                print(
                    f"Dataloader for scheme '{scheme}' exhausted at batch {batch_idx}."
                )
                break

            images = batch['image'].to(device)
            wvs = batch['wvs'].to(device) if 'wvs' in batch else None

            with torch.no_grad():
                # Store GT for this scheme (only if not already present from another compatible scheme)
                gt_key = f'GT ({scheme})'
                results[gt_key] = unnormalize(images, args.modality, scheme, device)

                for name, model in group['models']:
                    if 'TerraMind' in name:
                        recon = model(images, timesteps=20)
                    elif 'Refiner' in name:
                        recon = model(images, wvs)
                    else:
                        if hasattr(model, 'reconstruct'):
                            recon = model.reconstruct(images, wvs)
                        else:
                            recon, _ = model(images, wvs)

                    # Unnormalize with the same scheme used for input
                    results[name] = unnormalize(recon, args.modality, scheme, device)

        # 5. Plot Logic
        gt_keys = [k for k in results if k.startswith('GT')]

        # Consolidate GT: Use the first one found as the display Truth
        if gt_keys:
            results['Ground Truth'] = results.pop(gt_keys[0])
            for k in gt_keys[1:]:
                results.pop(k, None)

        # Ensure 'Ground Truth' is the last key for consistency
        keys = [k for k in results if k != 'Ground Truth'] + ['Ground Truth']

        # Setup Figure
        B = results['Ground Truth'].shape[0]
        cols = len(results)

        # Calculate figure size dynamically
        fig, axes = plt.subplots(B, cols, figsize=(4 * cols, 4 * B))
        plt.subplots_adjust(wspace=0.05, hspace=0.1)

        if B == 1:
            axes = axes[None, :]
        if cols == 1:
            axes = axes[:, None]

        for b in range(B):
            for c, name in enumerate(keys):
                ax = axes[b, c]
                ax.imshow(get_display_image(results[name][b], args.modality))
                if b == 0:
                    ax.set_title(name, fontsize=16)
                ax.axis('off')

        # Save and Close
        out_path = os.path.join(args.out_dir, f'batch_{batch_idx:03d}.png')
        plt.savefig(out_path, bbox_inches='tight')
        plt.close(fig)  # Crucial: Close figure to prevent memory leak

    print(f'\nDone. Saved {args.num_batches} images to {args.out_dir}')


if __name__ == '__main__':
    main()
