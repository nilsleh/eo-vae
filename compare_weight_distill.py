import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import DecoderOutput
from hydra.utils import instantiate
from omegaconf import OmegaConf

from eo_vae.datasets.terramesh_datamodule import NORM_STATS


def load_config(config_path: str):
    """Load YAML configuration from a given path."""
    return OmegaConf.load(config_path)


def create_objects(cfg):
    """Instantiate datamodule and model from the config."""
    datamodule = instantiate(cfg.datamodule, batch_size=8)
    datamodule.setup('fit')
    model = instantiate(cfg.model)
    model.eval()
    return datamodule, model


def unnormalize_for_input(image: torch.Tensor, norm_mode: str) -> torch.Tensor:
    """Unnormalize input image based on norm_mode: for '01', convert z-score to [0,1]; for 'zscore', leave as is."""
    if norm_mode == '01':
        modality = 'S2RGB'
        mean = torch.tensor(NORM_STATS[modality]['mean'], device=image.device).view(
            -1, 1, 1
        )
        std = torch.tensor(NORM_STATS[modality]['std'], device=image.device).view(
            -1, 1, 1
        )
        out = image * (std + 1e-8) + mean
        out = out / 255.0
        out = torch.clamp(out, 0.0, 1.0)
        return out
    elif norm_mode == 'zscore':
        return image
    else:
        raise ValueError("norm_mode must be 'zscore' or '01'")


def unnormalize_for_plotting(recon: torch.Tensor, norm_mode: str) -> torch.Tensor:
    """Unnormalize reconstruction for plotting: for 'zscore', convert to [0,1]; for '01', just clamp."""
    if norm_mode == 'zscore':
        modality = 'S2RGB'
        mean = torch.tensor(NORM_STATS[modality]['mean'], device=recon.device).view(
            -1, 1, 1
        )
        std = torch.tensor(NORM_STATS[modality]['std'], device=recon.device).view(
            -1, 1, 1
        )
        out = recon * (std + 1e-8) + mean
        out = out / 255.0
        out = torch.clamp(out, 0.0, 1.0)
        return out
    elif norm_mode == '01':
        return torch.clamp(recon, 0.0, 1.0)
    else:
        raise ValueError("norm_mode must be 'zscore' or '01'")


def forward_pass(model, input_tensor: torch.Tensor, wvs: torch.Tensor | None = None):
    """Pass the input through the model to obtain the reconstruction."""
    with torch.no_grad():
        if wvs is not None:
            output = model(input_tensor, wvs=wvs)
        else:
            output = model(input_tensor)

    if isinstance(output, (list, tuple)):
        output = output[0]
    elif isinstance(output, DecoderOutput):
        output = output['sample']

    return output


def compute_mse(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Compute Mean Squared Error between original and reconstructed image."""
    return F.mse_loss(original, reconstructed).item()


def tensor_to_image(img: torch.Tensor) -> np.ndarray:
    """Convert a tensor of shape (1, C, H, W) to a NumPy image (H, W, C)."""
    img = img.squeeze(0).cpu().numpy()  # now C, H, W
    img = np.transpose(img, (1, 2, 0))  # now H, W, C
    return img


def plot_batch_comparison(
    originals: list[torch.Tensor],
    trained_recons: list[torch.Tensor],
    original_recons: list[torch.Tensor],
    mses_trained: list[float],
    mses_original: list[float],
    num_samples: int,
):
    """Plot a grid of original, trained reconstruction, and original reconstruction for multiple samples."""
    fig, axes = plt.subplots(num_samples, 3, figsize=(18, 6 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)  # Ensure 2D array

    for i in range(num_samples):
        orig_img = tensor_to_image(originals[i])
        trained_img = tensor_to_image(trained_recons[i])
        orig_recon_img = tensor_to_image(original_recons[i])

        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(f'Sample {i + 1}: Original')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(trained_img)
        axes[i, 1].set_title(f'Trained Recon\nMSE: {mses_trained[i]:.4f}')
        axes[i, 1].axis('off')
        axes[i, 2].imshow(orig_recon_img)
        axes[i, 2].set_title(f'Original Flux Recon\nMSE: {mses_original[i]:.4f}')
        axes[i, 2].axis('off')

    fig.suptitle('Reconstruction Comparison (Batch)', fontsize=16)
    fig.savefig('recon_comparison_batch.png')


def main():
    parser = argparse.ArgumentParser(
        description='Compare reconstructions using trained and original FLUX2 autoencoders for a batch.'
    )
    parser.add_argument(
        '--config', type=str, required=True, help='Path to the YAML config file.'
    )
    parser.add_argument(
        '--ckpt',
        type=str,
        required=True,
        help='Path to the trained model checkpoint file.',
    )
    parser.add_argument(
        '--norm-mode',
        type=str,
        choices=['zscore', '01'],
        default='zscore',
        help='Normalization mode: zscore or 01.',
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=4,
        help='Number of samples to plot from the batch.',
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    datamodule, trained_model = create_objects(cfg)

    # Load the trained checkpoint
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    trained_model.load_state_dict(
        checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    )

    # Load the original FLUX2 autoencoder from Hugging Face
    flux2_vae = AutoencoderKL.from_pretrained(
        'black-forest-labs/FLUX.2-dev', subfolder='vae'
    )
    flux2_vae.eval()

    batch = next(iter(datamodule.train_dataloader()))
    input_batch = batch['image']
    num_samples = min(args.num_samples, input_batch.shape[0])

    # Select the first num_samples
    input_selected = input_batch[:num_samples]
    input_for_model = unnormalize_for_input(input_selected, args.norm_mode)
    input_unnorm_selected = unnormalize_for_plotting(
        input_selected, 'zscore'
    )  # Always unnormalize originals for plotting

    wvs_batch = torch.tensor([0.665, 0.560, 0.490])

    # Reconstruct with trained model (batch processing)
    trained_recon_selected = forward_pass(trained_model, input_for_model, wvs_batch)
    trained_recon_for_plot = unnormalize_for_plotting(
        trained_recon_selected, args.norm_mode
    )

    # Reconstruct with original model (batch processing)
    original_recon_selected = forward_pass(flux2_vae, input_for_model)
    original_recon_for_plot = unnormalize_for_plotting(
        original_recon_selected, args.norm_mode
    )

    # Create lists for plotting
    originals = [input_unnorm_selected[i : i + 1] for i in range(num_samples)]
    trained_recons = [trained_recon_for_plot[i : i + 1] for i in range(num_samples)]
    original_recons = [original_recon_for_plot[i : i + 1] for i in range(num_samples)]
    mses_trained = [
        compute_mse(input_unnorm_selected[i : i + 1], trained_recon_for_plot[i : i + 1])
        for i in range(num_samples)
    ]
    mses_original = [
        compute_mse(
            input_unnorm_selected[i : i + 1], original_recon_for_plot[i : i + 1]
        )
        for i in range(num_samples)
    ]

    # Plot comparison for the batch
    plot_batch_comparison(
        originals,
        trained_recons,
        original_recons,
        mses_trained,
        mses_original,
        num_samples,
    )


if __name__ == '__main__':
    main()
