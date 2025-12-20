import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf
from terratorch.registry import FULL_MODEL_REGISTRY

from eo_vae.datasets.terramesh_datamodule import NORM_STATS


def unnormalize(tensor, modality):
    """Reverse the normalization for visualization."""
    if modality not in NORM_STATS:
        return tensor
    device = tensor.device
    mean = torch.tensor(NORM_STATS[modality]['mean'], device=device).view(1, -1, 1, 1)
    std = torch.tensor(NORM_STATS[modality]['std'], device=device).view(1, -1, 1, 1)
    return tensor * (std + 1e-8) + mean


def get_display_image(tensor, modality):
    """Convert (C, H, W) tensor to (H, W, 3) numpy image for plotting."""
    tensor = tensor.cpu().detach()
    if modality == 'S2L2A':
        # RGB indices for S2L2A: Red(3), Green(2), Blue(1)
        img = tensor[[3, 2, 1], :, :]
        img = torch.clamp(img / 3000.0, 0, 1)  # Simple scaling for visualization
    elif modality == 'S1RTC':
        # False color: R=VV, G=VH, B=Ratio
        vv = tensor[0]
        vh = tensor[1]
        ratio = vh - vv
        img = torch.stack([vv, vh, ratio])
        # Min-max scale per channel for visibility
        for c in range(3):
            mn, mx = img[c].min(), img[c].max()
            img[c] = (img[c] - mn) / (mx - mn + 1e-8)
    else:
        # Default to first 3 channels or repeat if single channel
        if tensor.shape[0] >= 3:
            img = tensor[:3]
        else:
            img = tensor.repeat(3, 1, 1)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    return img.permute(1, 2, 0).numpy()


def get_latent_image(latent):
    """Convert (C, H, W) latent tensor to (H, W, 3) numpy image.
    Uses PCA to project to 3 channels if C > 3, otherwise pads/selects.
    """
    latent = latent.cpu().detach()
    C, H, W = latent.shape

    if C < 3:
        # Pad with zeros if less than 3 channels
        img = torch.cat([latent, torch.zeros(3 - C, H, W)], dim=0)
    elif C == 3:
        img = latent
    else:
        # Simple approach: Take first 3 channels.
        # Alternatively, use PCA for better representation:
        flat = latent.permute(1, 2, 0).reshape(-1, C)  # (HW, C)
        try:
            # Fast PCA via SVD on small subset if image is large, or full
            U, S, V = torch.pca_lowrank(flat, q=3, center=True, niter=2)
            projected = torch.matmul(flat, V[:, :3])
            img = projected.reshape(H, W, 3).permute(2, 0, 1)
        except:
            # Fallback to first 3 channels if PCA fails
            img = latent[:3]

    # Normalize to 0-1 for display
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img.permute(1, 2, 0).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument(
        '--modality', type=str, default='S2L2A', choices=['S2L2A', 'S1RTC']
    )
    parser.add_argument('--batch_idx', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    args = parser.parse_args()

    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    print(f'Using device: {device}')

    # 1. Load EO-VAE
    cfg = OmegaConf.load(args.config)
    # Force validation mode to deterministic modality
    cfg.datamodule.val_collate_mode = args.modality
    datamodule = instantiate(cfg.datamodule, eval_batch_size=8)
    datamodule.setup('fit')

    eo_vae = instantiate(cfg.model)
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    eo_vae.load_state_dict(state_dict)
    eo_vae.to(device)
    eo_vae.eval()

    # 2. Load Terramind
    # Infer model name from modality
    tm_model_name = (
        'terramind_v1_tokenizer_s2l2a'
        if args.modality == 'S2L2A'
        else 'terramind_v1_tokenizer_s1rtc'
    )

    print(f'Loading Terramind model: {tm_model_name}')
    tm_model = FULL_MODEL_REGISTRY.build(
        tm_model_name,
        pretrained=True,
        ckpt_path=os.path.join(
            '/mnt/SSD2/nils/eo-vae/checkpoints/terramind',
            f'TerraMind_Tokenizer_{args.modality}.pt',
        ),
    )
    tm_model.to(device)
    tm_model.eval()

    # 3. Get Data
    loader = datamodule.val_dataloader()
    for i, batch in enumerate(loader):
        if i == args.batch_idx:
            images = batch['image'].to(device)
            wvs = batch['wvs'].to(device)
            break

    # 4. Inference
    from einops import rearrange

    with torch.no_grad():
        # eo-vae
        latents_eo = eo_vae.encode(images, wvs).mode()
        z_shuffled = rearrange(
            latents_eo,
            '... c (i pi) (j pj) -> ... (c pi pj) i j',
            pi=eo_vae.ps[0],
            pj=eo_vae.ps[1],
        )
        z_normalized = eo_vae.normalize_latent(z_shuffled)
        eo_recon = eo_vae.decode(z_normalized, wvs)

        # terramind
        tm_recon = tm_model(images, timesteps=20)

    # 5. Visualization
    B = images.shape[0]
    # Increased columns to 7: Input, EO-Rec, TM-Rec, EO-Latent, EO-Err, TM-Err
    fig, axes = plt.subplots(B, 6, figsize=(28, 4 * B))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)

    for b in range(B):
        orig = images[b]
        rec_eo = eo_recon[b]
        rec_tm = tm_recon[b]
        lat_eo = latents_eo[b]

        # Compute MSE
        mse_eo = F.mse_loss(rec_eo, orig).item()
        mse_tm = F.mse_loss(rec_tm, orig).item()

        # MSE Maps (Average over channels)
        mse_map_eo = torch.mean((rec_eo - orig) ** 2, dim=0).cpu().numpy()
        mse_map_tm = torch.mean((rec_tm - orig) ** 2, dim=0).cpu().numpy()

        # Unnormalize for display
        orig_disp = unnormalize(orig.unsqueeze(0), args.modality).squeeze(0)
        rec_eo_disp = unnormalize(rec_eo.unsqueeze(0), args.modality).squeeze(0)
        rec_tm_disp = unnormalize(rec_tm.unsqueeze(0), args.modality).squeeze(0)

        # Plot
        ax_row = axes[b] if B > 1 else axes

        # 1. Input
        ax_row[0].imshow(get_display_image(orig_disp, args.modality))
        ax_row[0].set_title('Input')
        ax_row[0].axis('off')

        # 2. EO-VAE Recon
        ax_row[1].imshow(get_display_image(rec_eo_disp, args.modality))
        ax_row[1].set_title(f'EO-VAE\nMSE: {mse_eo:.4f}')
        ax_row[1].axis('off')

        # 3. Terramind Recon
        ax_row[2].imshow(get_display_image(rec_tm_disp, args.modality))
        ax_row[2].set_title(f'Terramind\nMSE: {mse_tm:.4f}')
        ax_row[2].axis('off')

        # 4. EO-VAE Latent
        ax_row[3].imshow(get_latent_image(lat_eo))
        ax_row[3].set_title(f'EO-VAE Latent\n{tuple(lat_eo.shape)}')
        ax_row[3].axis('off')

        # 6. EO-VAE Error
        im5 = ax_row[4].imshow(mse_map_eo, cmap='hot', vmin=0, vmax=0.1)
        ax_row[4].set_title('EO-VAE Error')
        ax_row[4].axis('off')
        plt.colorbar(im5, ax=ax_row[4], fraction=0.046, pad=0.04)

        # 7. Terramind Error
        im6 = ax_row[5].imshow(mse_map_tm, cmap='hot', vmin=0, vmax=0.1)
        ax_row[5].set_title('Terramind Error')
        ax_row[5].axis('off')
        plt.colorbar(im6, ax=ax_row[5], fraction=0.046, pad=0.04)

    plt.suptitle(f'Reconstruction & Latent Comparison - {args.modality}', fontsize=16)

    # Save to config directory
    output_dir = os.path.dirname(args.config)
    save_path = os.path.join(output_dir, f'comparison_{args.modality}.png')
    fig.savefig(save_path)
    print('Saving comparison figure to:', save_path)


if __name__ == '__main__':
    main()
