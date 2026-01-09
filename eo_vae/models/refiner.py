import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim.lr_scheduler import LambdaLR
import math


class RefinerTrainer(LightningModule):
    """
    Stage 2 Trainer: Trains a Flow Matching Refiner on top of a frozen Base VAE.

    Architecture:
        - Base VAE (Frozen): Generates 'Blurry' condition (z) from Ground Truth (x).
        - Flow Model: Predicts the vector field to transport z -> x.

    This class is modality-agnostic. It assumes the 'flow_model' passed to it
    is the correct architecture (DeCo) for the data loader's current modality.
    """

    def __init__(
        self,
        base_vae: LightningModule,
        denoiser: torch.nn.Module | None = None,
        sampler: torch.nn.Module | None = None,
        lr: float = 1e-4,
        final_lr: float = 1e-6,
        warmup_steps: int = 1000,
        num_training_steps: int = 200000,
        num_inference_steps: int = 10,  # For validation
        use_clamping: bool = True,
    ):
        super().__init__()
        # We ignore base_vae and flow_model in hparams to avoid pickling huge models in checkpoints
        self.lr = lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.num_training_steps = num_training_steps

        # 1. Base VAE (Stage 1) - Frozen
        self.base_vae = base_vae
        self.base_vae.eval()
        self.base_vae.freeze()  # explicit freeze

        # 2. Flow Model (Stage 2) - The Trainable Component
        # Expected to be the FlowRefinementDenoiser wrapper we defined
        self.denoiser = denoiser
        self.sampler = sampler

        self.num_inference_steps = num_inference_steps
        self.use_clamping = use_clamping

    def forward(self, x, wvs, **kwargs):
        """Forward pass mainly for inference/debugging."""
        with torch.no_grad():
            # Generate condition from Base VAE
            z_cond = self.get_condition({'image': x, 'wvs': wvs})

        x_recon = self.refine(z_cond, steps=self.num_inference_steps)
        return x_recon

    # def on_train_start(self):
    #     """Ensure VAE remains frozen at start of training."""
    #     self.base_vae.eval()
    #     self.base_vae.freeze()

    def get_condition(self, batch):
        """Extracts the 'Blurry' condition (z) from the Base VAE."""
        x = batch['image']
        wvs = batch.get('wvs', None)

        with torch.no_grad():
            # Run Base VAE Encoder+Decoder
            # AutoencoderFlux returns (reconstruction, posterior, ...)
            z, _ = self.base_vae(x, wvs)
            z = z.detach()
            if self.use_clamping:
                # Essential for stability: match the training bounds
                z = torch.clamp(z, -1, 1)
        return z

    def training_step(self, batch, batch_idx):
        # 1. Ground Truth (Sharp)
        x_target = batch['image']

        # 2. Generate Condition (Blurry)
        z_cond = self.get_condition(batch)

        # 3. Sample Time t ~ U[0, 1]
        # We rely on the FlowRefinementDenoiser to handle the broadcasting logic
        t = torch.rand(x_target.shape[0], device=self.device)

        # 4. Calculate Flow Matching Loss
        # The wrapper handles the interpolation (x_t) and calls the backbone
        # We pass 'condition' as a kwarg which gets forwarded to DeCoPixelDecoder
        loss = self.denoiser.loss(x=x_target, t=t, cond=z_cond)

        self.log('train/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_target = batch['image']
        z_cond = self.get_condition(batch)

        x_refined = self.refine(z_cond, steps=25)

        base_mse = F.mse_loss(z_cond, x_target)
        refined_mse = F.mse_loss(x_refined, x_target)

        self.log_dict(
            {
                'val/mse_base': base_mse,
                'val/mse_refined': refined_mse,
                'val/refinement_gain': base_mse - refined_mse,
            },
            prog_bar=True,
        )

    def refine(self, z_cond, steps=25):
        sampler = self.sampler(denoiser=self.denoiser, steps=steps)
        x_start = torch.randn_like(z_cond)
        return sampler(x_start, cond=z_cond)

    def configure_optimizers(self):
        """Uses the same scheduler logic as your AutoencoderFlux."""
        optimizer = torch.optim.AdamW(
            self.denoiser.parameters(), lr=self.lr, weight_decay=1e-3
        )

        # Use the utility provided in your utils or define here
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.num_training_steps,
            base_lr=self.lr,
            final_lr=self.final_lr,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            },
        }


# --- Copy of your Scheduler Utility for standalone completeness ---
def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, base_lr, final_lr
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (final_lr + (base_lr - final_lr) * cosine_decay) / base_lr

    return LambdaLR(optimizer, lr_lambda)
