# EO-VAE

Earth Observation Variational AutoEncoder (EO-VAE) repository. This project focuses on training Variational AutoEncoders for remote sensing data and leveraging the latent space for downstream tasks like super-resolution.

## Table of Contents

1. [Installation](#installation)
2. [Stage 1: Weight Distillation](#stage-1-weight-distillation)
3. [Stage 2: VAE Finetuning](#stage-2-vae-finetuning)
4. [Stage 3: Super-Resolution Training](#stage-3-super-resolution-training)
5. [Evaluation](#evaluation)
6. [Project Structure](#project-structure)

---

## Installation

### Install Dependencies

Install the package and its dependencies in editable mode:

```bash
cd /path/to/eo-vae
pip install -e .
```

### Using a pretrained checkpoint (Inference)

Use the built-in config-driven loader to avoid manual `Encoder`/`Decoder` construction.

```python
import torch
from eo_vae.models.new_autoencoder import EOFluxVAE

model = EOFluxVAE.from_pretrained(
    repo_id="nilsleh/eo-vae", 
    ckpt_filename="eo-vae.ckpt",
    config_filename="model_config.yaml",
    device="cpu",
)

# Run reconstruction / latent extraction
x = torch.randn(1, 3, 256, 256)
wvs = torch.tensor([0.665, 0.56, 0.49], dtype=torch.float32)  # example for S2RGB

with torch.no_grad():
    recon = model.reconstruct(x, wvs)                # [B, 3, 256, 256]
    z = model.encode_spatial_normalized(x, wvs)      # [B, 32, 32, 32] for 256x256 input
```

These are the wavelengths used across modalities:


```python
WAVELENGTHS = {
    'S2RGB': [0.665, 0.56, 0.49],
    'S1RTC': [5.4, 5.6],
    'S2L2A': [
        0.443, 0.490, 0.560, 0.665, 0.705, 0.740,
        0.783, 0.842, 0.865, 1.610, 2.190, 0.945,
    ],
    'S2L1C': [
        0.443, 0.490, 0.560, 0.665, 0.705, 0.740,
        0.783, 0.842, 0.865, 0.945, 1.375, 1.610, 2.190,
    ],
}
```

### Data Requirements

The pipeline expects the following data structure:

```
/path/to/data/
├── terramesh/               # For weight distillation & VAE finetuning
│   ├── S2L2A/
│   ├── S1RTC/
│   └── ...
└── sen2naip/                # For super-resolution
    ├── train/
    ├── val/
    └── test/
```

Update the data paths in the relevant config files before running.

---

## Stage 1: Weight Distillation

**Purpose**: Distill weights from a pretrained Flux VAE into the RGB channels of the dynamic convolution layers, preparing them for multi-modality finetuning.

### Prerequisites

- Pre-trained Flux2 AutoEncoder checkpoint (e.g., `ae.safetensors`)
- TerraMesh dataset (S2L2A modality)

### Configuration

The distillation uses the config at `configs/weight_distill.yaml`. Review and update:

```yaml
experiment:
  exp_dir: "results/exps/eo-vae/all"  # Output directory for checkpoints

datamodule:
  data_path: "/mnt/SSD2/nils/datasets/terramesh"  # Update to your data path
  batch_size: 16
  num_workers: 4
```

### Run

```bash
python weight_distill_train.py \
  --config configs/weight_distill.yaml \
  --teacher-ckpt /path/to/flux_ae.safetensors
```

**Optional Arguments**:
- `--max-steps 5000` - Maximum training steps (default from config)
- `--lr 1e-4` - Learning rate
- `--val-every 500` - Validation frequency
- `--log-every 50` - Logging frequency

### Output

A checkpoint saving distilled weights will be created in:
```
results/exps/eo-vae/all/eo-vae-distill-2-layers_MM-DD-YYYY_HH-MM-SS-ffffff/
```

This checkpoint path will be used as `--ckpt` in Stage 2.

---

## Stage 2: VAE Finetuning

**Purpose**: Finetune the weight-distilled VAE on multi-modal TerraMesh data (S2L2A, S1RTC, etc.).

### Prerequisites

- Distilled checkpoint from Stage 1
- TerraMesh dataset with multiple modalities

### Configuration

The finetuning uses the config at `configs/eo-vae.yaml`. Key settings:

```yaml
experiment:
  exp_dir: "results/exps/eo-vae/full"

model:
  _target_: eo_vae.models.new_autoencoder.EOFluxVAE
  base_lr: 1e-4
  final_lr: 2e-5
  
  loss_fn:
    _target_: eo_vae.models.modules.consistency_loss.EOConsistencyLoss
    pixel_weight: 1.0
    msssim_weight: 1.0  # MS-SSIM loss starts after 2000 steps

datamodule:
  data_path: "path to terramesh data"  # Update to your path
  modalities: ["S2L2A"]  # Can add more: ["S2L2A", "S1RTC"]
  batch_size: 16
  train_collate_mode: "random"  # Randomly select modality per batch
  val_collate_mode: "S2L2A"     # Validation on S2L2A

trainer:
  max_epochs: 100  # Adjust based on your needs
```

### Run

```bash
python train.py \
  --config configs/eo-vae.yaml \
  --ckpt /path/to/distilled_checkpoint.pt
```

This will:
1. Load the distilled VAE weights
2. Train on the specified modalities with consistency and MS-SSIM losses
3. Save checkpoints periodically
4. Log progress to Weights & Biases (if configured)

### Output


This checkpoint path will be used for latent encoding and super-resolution.

---

## Stage 3: Super-Resolution Training

**Purpose**: Train a diffusion-based super-resolution model in the latent space of the frozen VAE.

### Prerequisites

- Finetuned VAE checkpoint from Stage 2
- Sen2NAIP dataset

### Step 3.1: Encode Latents

First, encode the entire dataset to latent space using the frozen VAE to speed up training:

```bash
python encode_latents.py \
  --sen2naip_root /path/to/sen2naip \
  --config configs/eo-vae.yaml \
  --ckpt /path/to/vae_checkpoint.pt \
  --output_root /path/to/latent_output \
  --use_spatial_norm
```

**Arguments**:
- `--sen2naip_root` - Path to Sen2NAIP dataset
- `--config` - VAE config file
- `--ckpt` - Trained VAE checkpoint
- `--output_root` - Directory to save encoded latents and statistics
- `--use_spatial_norm` - Apply spatial normalization to latents

**Output**:
- Latents saved as `.npy` files for train/val/test splits
- Statistics file: `latent_stats.json` (containing mean and std for normalization)

### Step 3.2: Configure Super-Resolution Training

Update the super-resolution config at `configs_superres/eo_vae_latent.yaml`:

```yaml
experiment:
  exp_dir: "results/exps/eo-vae/sr"

trainer:
  max_epochs: 750
  accelerator: "gpu"
  devices: [0]  # GPU device ID

lightning_module:
  base_lr: 1e-4
  final_lr: 1e-5
  warmup_epochs: 10
  
datamodule:
  latent_root: /path/to/latent_output  # From Step 3.1
  batch_size: 4
  num_workers: 4
```

### Step 3.3: Train Super-Resolution

```bash
python train_super_res.py --config configs_superres/eo_vae_latent.yaml
```

**Optional Debug Mode** (CPU, no logging):
```bash
python train_super_res.py --config configs_superres/eo_vae_latent.yaml --debug
```


## Evaluation

### VAE Reconstruction Evaluation

#### Visual Evaluation

Visualize VAE reconstructions:

```bash
python visual_eval.py \
  --config configs/eo-vae.yaml \
  --ckpt /path/to/vae_checkpoint.pt \
  --modality S2L2A \
  --output_dir results/vae_visuals
```

This generates visualizations comparing input and reconstructed images.

#### Metric Evaluation

Evaluate VAE on reconstruction metrics (MSE, SSIM):

```bash
python eval_metric_super_res.py \
  --config_vae configs/eo-vae.yaml \
  --ckpt_vae /path/to/vae_checkpoint.pt
```

Outputs metrics to console and `results/sr_metrics.json`.

### Super-Resolution Evaluation

#### Metric-Based Evaluation

Evaluate super-resolution performance on test set:

```bash
python eval_metric_super_res.py \
  --config configs_superres/eo_vae_latent.yaml \
  --ckpt /path/to/sr_checkpoint.pt \
  --split test
```

**Metrics computed**:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- MSE (Mean Squared Error)
- SAM (Spectral Angle Mapper) - for multi-spectral data

Output saved to `results/sr_metrics.json`.

#### Visual Evaluation

Generate visual comparisons:

```bash
python eval_viz_tokenizer.py \
  --config configs_superres/eo_vae_latent.yaml \
  --ckpt /path/to/sr_checkpoint.pt \
  --output_dir results/sr_visuals \
  --num_samples 50
```

Generates comparison plots:
- Low-resolution input
- Super-resolved output
- High-resolution ground truth
- Difference maps

### Tokenizer/Model Comparison

Compare against TerraMind tokenizer:

```bash
python evaluate_metrics_tokenizer.py \
  --config configs/eo-vae.yaml \
  --ckpt /path/to/vae_checkpoint.pt \
  --modality S2L2A \
  --tm_model_name terramind_v1_tokenizer_s2l2a
```

Outputs comparison table of reconstruction metrics.

---

## Project Structure

```
.
├── configs/                          # VAE training configs
│   ├── eo-vae.yaml                  # Main VAE finetuning config
│   ├── weight_distill.yaml          # Weight distillation config
│   └── ...
│
├── configs_superres/                 # Super-resolution configs
│   ├── eo_vae_latent.yaml           # Main SR config
│   └── ...
│
├── eo_vae/
│   ├── models/
│   │   ├── autoencoder.py              # VAE encoder/decoder
│   │   ├── autoencoder_flux.py         # Flux-based VAE
│   │   ├── new_autoencoder.py          # Multi-modal VAE (EOFluxVAE)
│   │   ├── super_res.py                # Diffusion SR model
│   │   └── modules/
│   │       ├── consistency_loss.py     # Multi-modal consistency loss
│   │       ├── dynamic_conv.py         # Dynamic convolution layers
│   │       ├── loss_functions.py       # Training losses
│   │       └── ...
│   │
│   ├── datasets/
│   │   ├── terramesh_datamodule.py     # TerraMesh data loader
│   │   ├── sen2naip.py                 # Sen2NAIP data loader
│   │   └── ...
│   │
│   └── utils/                           # Training utilities
│       ├── callbacks.py                 # Lightning callbacks
│       ├── image_logger.py              # VAE image logging
│       └── super_res_image_logger.py    # SR image logging
│
├── train.py                           # VAE finetuning script
├── train_super_res.py                 # SR training script
├── weight_distill_train.py            # Weight distillation script
├── encode_latents.py                  # Latent encoding for SR
│
├── visual_eval.py                     # VAE visual evaluation
├── eval_metric_super_res.py          # SR metric evaluation
├── evaluate_metrics_tokenizer.py     # Model comparison
└── eval_viz_tokenizer.py             # Visual comparisons
```

## License

Apache-2.0 License

## Citation

If you use this code, please cite:

```bibtex
@article{eo-vae,
  title={EO-VAE: Towards A Multi-sensor Tokenizer for Earth Observation Data},
  author={Lehmann, Nils and Wang, Yi and Xiong, Zhitong and Zhu, Xiaoxiang},
  journal={arXiv preprint arXiv:2602.12177},
  year={2026}
}
```
