import argparse

import matplotlib.pyplot as plt
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Import your normalization logic
from eo_vae.datasets.terramesh_datamodule import normalize_image, unnormalize_image

OmegaConf.register_new_resolver('eval', eval)


def collect_pixel_data(cfg_path, modality, num_pixels=50000):
    """Loads data for a specific modality and converts it to Raw, Z-Score, and Robust spaces."""
    # 1. Setup Loader
    cfg = OmegaConf.load(cfg_path)
    cfg.datamodule.val_collate_mode = modality

    # Determine the native space of the loader (e.g., 'zscore' or 'robust')
    loader_norm = cfg.datamodule.get('norm_method', 'zscore')
    print(f'[{modality}] Loading data... (Native Loader Space: {loader_norm})')

    datamodule = instantiate(cfg.datamodule, eval_batch_size=8, num_workers=4)
    datamodule.setup('fit')
    loader = datamodule.val_dataloader()

    # 2. Collect Pixels
    # We collect a batch of tensors (B, C, H, W)
    collected_tensors = []
    pixel_count = 0

    for batch in loader:
        img = batch['image']  # (B, C, H, W)
        collected_tensors.append(img)

        pixel_count += img.shape[0] * img.shape[2] * img.shape[3]
        if pixel_count >= num_pixels:
            break

    # Combine into one large tensor: (N, C, 1, 1) to satisfy normalize_image dims
    full_batch = torch.cat(collected_tensors, dim=0)

    # Flatten spatial dims to sample random pixels
    # (B, C, H, W) -> (B*H*W, C, 1, 1)
    b, c, h, w = full_batch.shape
    flat = full_batch.permute(0, 2, 3, 1).reshape(-1, c, 1, 1)

    # Sample subset to save memory/time
    if flat.shape[0] > num_pixels:
        indices = torch.randperm(flat.shape[0])[:num_pixels]
        samples = flat[indices]  # (Num_Pixels, C, 1, 1)
    else:
        samples = flat

    # 3. Convert to All Spaces
    # Step A: Convert Loader Data -> Raw (Physical)
    raw = unnormalize_image(samples, modality, method=loader_norm)

    # Step B: Raw -> Z-Score
    zscore = normalize_image(raw, modality, method='zscore')

    # Step C: Raw -> Robust
    robust = normalize_image(raw, modality, method='robust')

    # Return as numpy arrays of shape (N, C)
    return {
        'Original (Raw)': raw.squeeze().numpy(),
        'Z-Score': zscore.squeeze().numpy(),
        'Robust Scaled': robust.squeeze().numpy(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, required=True, help='Path to base datamodule config'
    )
    parser.add_argument('--output', type=str, default='band_histograms.png')
    parser.add_argument(
        '--pixels', type=int, default=50000, help='Number of pixels to sample'
    )
    args = parser.parse_args()

    modalities = ['S2L2A', 'S1RTC']
    spaces = ['Original (Raw)', 'Z-Score', 'Robust Scaled']

    # Create 2x3 Grid
    fig, axes = plt.subplots(
        nrows=2, ncols=3, figsize=(18, 10), constrained_layout=True
    )

    for row_idx, mod in enumerate(modalities):
        print(f'Processing {mod}...')
        try:
            # Get Data
            data_dict = collect_pixel_data(args.config, mod, num_pixels=args.pixels)

            for col_idx, space in enumerate(spaces):
                ax = axes[row_idx, col_idx]
                data = data_dict[space]  # Shape (N, Bands)

                # Plot Histogram for each band
                num_bands = data.shape[1]

                # Iterate bands
                for b in range(num_bands):
                    # For S2L2A (12 bands), we use thin lines or low alpha
                    # For S1RTC (2 bands), we can be bolder
                    label = f'Band {b}'

                    # Plot
                    ax.hist(
                        data[:, b],
                        bins=100,
                        density=True,
                        histtype='step',
                        linewidth=1.5,
                        alpha=0.8,
                        label=label,
                    )

                # Formatting
                ax.set_title(f'{mod} - {space}')
                ax.grid(True, alpha=0.2, linestyle='--')

                # X-Axis Labels & Limits optimization
                if space == 'Z-Score':
                    ax.set_xlabel('Sigma ($\sigma$)')
                    ax.set_xlim(-5, 5)  # Focus on the bell curve
                elif space == 'Robust Scaled':
                    ax.set_xlabel('Normalized Value (0-1)')
                    ax.set_xlim(-0.1, 1.1)  # Focus on 0-1 range
                else:
                    ax.set_xlabel('Physical Value')
                    # Log scale often helps Raw S2 data visualization
                    # ax.set_yscale('log')

                if col_idx == 0:
                    ax.set_ylabel('Frequency Density')

                # Add Legend (maybe only for S1 or if bands are few)
                if num_bands <= 4:
                    ax.legend(loc='upper right', fontsize='small')

        except Exception as e:
            print(f'Error processing {mod}: {e}')
            axes[row_idx, 0].text(0.5, 0.5, f'Could not load {mod}\n{e}', ha='center')

    # Save
    plt.savefig(args.output, dpi=300)
    print(f'Plot saved to: {args.output}')


if __name__ == '__main__':
    main()
