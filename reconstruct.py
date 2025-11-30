import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf


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


def forward_pass(model, input_tensor: torch.Tensor):
    """Pass the input through the model to obtain the reconstruction.

    If the model returns a tuple/list, take the first element.
    """
    with torch.no_grad():
        output = model(input_tensor)
    if isinstance(output, (list, tuple)):
        output = output[0]
    return output


def compute_mse(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Compute Mean Squared Error between original and reconstructed image."""
    return F.mse_loss(original, reconstructed).item()


def tensor_to_image(img: torch.Tensor) -> np.ndarray:
    """Convert a tensor of shape (1, C, H, W) to a NumPy image (H, W, C)."""
    img = img.squeeze(0).cpu().numpy()  # now C, H, W
    img = np.transpose(img, (1, 2, 0))  # now H, W, C
    return img


def plot_reconstruction(
    original: torch.Tensor, reconstructed: torch.Tensor, mse_error: float
):
    """Plot original and reconstructed images side by side with MSE error in title."""
    orig_img = tensor_to_image(original)
    recon_img = tensor_to_image(reconstructed)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(orig_img)
    axes[0].set_title('Original')
    axes[0].axis('off')
    axes[1].imshow(recon_img)
    axes[1].set_title('Reconstructed')
    axes[1].axis('off')
    fig.suptitle(f'Reconstruction MSE: {mse_error:.4f}', fontsize=16)
    fig.savefig('reconstruction.png')


def main():
    parser = argparse.ArgumentParser(
        description='Reconstruct images using a trained model.'
    )
    parser.add_argument(
        '--config', type=str, required=True, help='Path to the YAML config file.'
    )
    parser.add_argument(
        '--ckpt', type=str, required=True, help='Path to the model checkpoint file.'
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    datamodule, model = create_objects(cfg)

    # Load the checkpoint
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(
        checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    )

    batch = next(iter(datamodule.train_dataloader()))

    input = batch['image'][2:3, ...]

    import pdb

    pdb.set_trace()

    # input = F.interpolate(input, (256, 256))

    wvs = torch.tensor([0.665, 0.560, 0.490])

    with torch.no_grad():
        recon = model(input, wvs)[0]

    mse_error = compute_mse(input, recon)
    plot_reconstruction(input, recon, mse_error)


if __name__ == '__main__':
    main()
