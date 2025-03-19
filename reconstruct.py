import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader


def load_config(config_path: str):
    """Load YAML configuration from a given path."""
    return OmegaConf.load(config_path)


def create_objects(cfg):
    """Instantiate datamodule and model from the config."""
    cfg.datamodule.root = cfg.datamodule.root.replace(
        '/mnt/data/', '/mnt/rg_climate_benchmark/data/'
    )
    datamodule = instantiate(cfg.datamodule, batch_size=1)
    datamodule.setup('fit')
    datamodule.setup('test')
    model = instantiate(cfg.model)
    model.eval()
    return datamodule, model


def forward_pass(model, input_tensor: torch.Tensor):
    """
    Pass the input through the model to obtain the reconstruction.

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

    if img.shape[-1] == 12:
        img = img[..., :3]
    return img


def plot_reconstruction(
    original: torch.Tensor, reconstructed: torch.Tensor, mse_error: float
):
    """Plot original and reconstructed images side by side with MSE error in title."""
    orig_img = tensor_to_image(original)
    recon_img = tensor_to_image(reconstructed)

    # normalize the images
    orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min())
    recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min())

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
    config_path = os.path.join('configs', 'seasonet.yaml')
    cfg = load_config(config_path)
    datamodule, model = create_objects(cfg)

    # ckpt = '/mnt/rg_climate_benchmark/results/nils/exps/eo-vae/seasonet/eo-vae-seasonet_02-12-2025_17-17-46-859655/epoch=392-step=7860.ckpt'
    # model.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])

    dl = datamodule.train_dataloader()

    new_dl = DataLoader(dl.dataset, batch_size=1, shuffle=True)

    batch = next(iter(new_dl))

    input = batch['image']
    input = torch.stack([input[:,3,...],input[:,2,...],input[:,1,...]],dim=1)


    input = F.interpolate(input, (256, 256))

    wvs = torch.tensor([0.665, 0.560, 0.490])


    wvs = batch['wvs'][0, [3, 2, 1]]

    with torch.no_grad():
        recon = model(input, wvs)[0]

    mse_error = compute_mse(input, recon)
    plot_reconstruction(input, recon, mse_error)


if __name__ == '__main__':
    main()
