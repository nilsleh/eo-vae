import argparse

import pandas as pd
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf
from terratorch.registry import FULL_MODEL_REGISTRY
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument(
        '--modality', type=str, default='S2L2A', choices=['S2L2A', 'S1RTC']
    )
    parser.add_argument(
        '--tm_model_name', type=str, default='terramind_v1_tokenizer_s2l2a'
    )
    args = parser.parse_args()

    # 1. Setup
    cfg = OmegaConf.load(args.config)
    cfg.datamodule.val_collate_mode = args.modality
    datamodule = instantiate(cfg.datamodule)
    datamodule.setup('fit')
    loader = datamodule.val_dataloader()

    # 2. Load Models
    eo_vae = instantiate(cfg.model)
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    eo_vae.load_state_dict(
        checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    )
    eo_vae.eval().cuda()

    # 2. Load Terramind
    # Infer model name from modality
    tm_model_name = (
        'terramind_v1_tokenizer_s2l2a'
        if args.modality == 'S2L2A'
        else 'terramind_v1_tokenizer_s1rtc'
    )

    print(f'Loading Terramind model: {tm_model_name}')
    tm_model = FULL_MODEL_REGISTRY.build(tm_model_name, pretrained=True)
    tm_model.eval().cuda()

    # 3. Evaluation Loop
    metrics = {'eo_mse': 0.0, 'eo_mae': 0.0, 'tm_mse': 0.0, 'tm_mae': 0.0, 'count': 0}

    print(f'Starting evaluation on {len(loader)} batches...')
    with torch.no_grad():
        for batch in tqdm(loader):
            images = batch['image'].cuda()
            wvs = batch['wvs'].cuda()
            B = images.shape[0]

            # EO-VAE Inference
            out_eo = eo_vae(images, wvs)
            recon_eo = out_eo[0] if isinstance(out_eo, (tuple, list)) else out_eo

            # Terramind Inference
            out_tm = tm_model(images)
            recon_tm = out_tm[0] if isinstance(out_tm, (tuple, list)) else out_tm

            # Accumulate Metrics
            metrics['eo_mse'] += F.mse_loss(recon_eo, images, reduction='sum').item()
            metrics['eo_mae'] += F.l1_loss(recon_eo, images, reduction='sum').item()

            metrics['tm_mse'] += F.mse_loss(recon_tm, images, reduction='sum').item()
            metrics['tm_mae'] += F.l1_loss(recon_tm, images, reduction='sum').item()

            metrics['count'] += B * images.numel() // B  # Total elements

    # 4. Final Calculation
    total_elements = metrics['count']
    results = {
        'Model': ['EO-VAE', 'Terramind'],
        'MSE': [metrics['eo_mse'] / total_elements, metrics['tm_mse'] / total_elements],
        'MAE': [metrics['eo_mae'] / total_elements, metrics['tm_mae'] / total_elements],
    }

    df = pd.DataFrame(results)
    print('\nFinal Evaluation Results:')
    print(df.to_markdown(index=False))


if __name__ == '__main__':
    main()
