import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from hydra.utils import instantiate
from einops import rearrange

# Import datasets and normalization utils
from eo_vae.datasets.sen2naip import Sen2NaipCrossSensorDataModule
from eo_vae.models.autoencoder_flux import FluxAutoencoderKL


def load_eo_vae(config_path, ckpt_path, device):
    """Loads Standard EO-VAE from config."""
    print(f'Loading EO-VAE from config: {config_path}')
    conf = OmegaConf.load(config_path)
    model = instantiate(conf.model)
    if ckpt_path:
        print(f'Loading EO-VAE checkpoint from {ckpt_path}')
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = (
            checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        )
        model.load_state_dict(state_dict, strict=False)

    model.to(device).eval()
    return model


from diffusers import AutoencoderKL


def load_flux():
    flux2_vae = AutoencoderKL.from_pretrained(
        'black-forest-labs/FLUX.2-dev', subfolder='vae'
    )
    flux2_vae.eval()

    return flux2_vae


@torch.no_grad()
def encode_batch(model, img, wvs=None):
    """Encodes batch to latents."""

    # --- Case A: EO-VAE (FluxAutoencoderKL) ---
    if isinstance(model, FluxAutoencoderKL):
        posterior = model.encode(img, wvs)
        z = posterior.mode()
        return z

    # --- Case B:: Flux VAE ---
    if hasattr(model, 'encode'):
        res = model.encode(img, return_dict=False)[0].mode()
        return res

    return model(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sen2naip_root',
        type=str,
        default='/mnt/SSD2/nils/datasets/sen2naip/cross-sensor/cross-sensor',
    )
    parser.add_argument(
        '--config', type=str, required=True, help='Path to EO-VAE config'
    )
    parser.add_argument(
        '--ckpt', type=str, default=None, help='Path to EO-VAE checkpoint'
    )
    parser.add_argument(
        '--output_root', type=str, required=True, help='Root folder e.g. /data/latents'
    )

    parser.add_argument(
        '--modality_lr', type=str, default='S2L2A', help='Modality stats for LR'
    )
    parser.add_argument(
        '--modality_hr',
        type=str,
        default='S2RGB',
        help='Modality stats for HR (EO-VAE)',
    )
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # 1. Setup Dataset
    print('Setting up Sen2Naip DataModule...')
    dm = Sen2NaipCrossSensorDataModule(
        root=args.sen2naip_root, batch_size=args.batch_size, num_workers=4
    )
    dm.setup('fit')

    # 2. Load Models
    # A. EO-VAE
    eo_vae_model = load_eo_vae(args.config, args.ckpt, device)

    # Flux VAE
    flux_vae = load_flux()
    flux_vae.to(device).eval()

    # helper for EO-VAE wavelengths
    wvs_hr = torch.tensor([0.665, 0.56, 0.49, 0.842], device=device)
    wvs_lr = torch.tensor([0.665, 0.56, 0.49, 0.842], device=device)

    splits = ['train', 'val', 'test']

    for split in splits:
        if split == 'train':
            loader = dm.train_dataloader()
        elif split == 'val':
            loader = dm.val_dataloader()
        else:
            loader = dm.test_dataloader()

        # Directories
        out_eo_vae = os.path.join(args.output_root, 'eo_vae', split)
        os.makedirs(out_eo_vae, exist_ok=True)

        out_flux_vae = None
        if flux_vae:
            out_flux_vae = os.path.join(args.output_root, 'flux_vae', split)
            os.makedirs(out_flux_vae, exist_ok=True)

        print(f'Processing {split} split...')

        for batch in tqdm(loader):
            # 1. Get Data (Sen2Naip Z-Scored)
            lr_sen = batch['image_lr'].to(device)
            hr_sen = batch['image_hr'].to(device)
            aois = batch['aoi']

            # Encoding
            z_lr_vae = encode_batch(eo_vae_model, lr_sen, wvs_lr)
            z_hr_vae = encode_batch(eo_vae_model, hr_sen, wvs_hr)

            # ---------------- FLUX VAE PATH ----------------
            z_lr_flux, z_hr_flux = None, None
            if flux_vae:
                # Preprocessing (Flux VAE is RGB only for this tokenizer)
                # Slice HR to 3 channels (R,G,B)
                hr_flux = hr_sen[:, :3, :, :]

                # Slice LR to 3 channels (R,G,B)
                lr_flux = lr_sen[:, :3, :, :]

                z_lr_flux = encode_batch(flux_vae, lr_flux)
                z_hr_flux = encode_batch(flux_vae, hr_flux)

            # ---------------- SAVE ----------------
            for i, aoi_id in enumerate(aois):
                # Save EO-VAE
                np.savez_compressed(
                    os.path.join(out_eo_vae, f'{aoi_id}.npz'),
                    lr_latent=z_lr_vae[i].cpu().numpy(),
                    hr_latent=z_hr_vae[i].cpu().numpy(),
                    lr_image=lr_sen[i].cpu().numpy(),
                    hr_image=hr_sen[i].cpu().numpy(),
                )

                # Save Flux VAE
                if flux_vae:
                    np.savez_compressed(
                        os.path.join(out_flux_vae, f'{aoi_id}.npz'),
                        lr_latent=z_lr_flux[i].cpu().numpy(),
                        hr_latent=z_hr_flux[i].cpu().numpy(),
                        lr_image=lr_sen[i].cpu().numpy(),
                        hr_image=hr_sen[i].cpu().numpy(),
                    )

    print('Encoding complete.')


if __name__ == '__main__':
    main()
