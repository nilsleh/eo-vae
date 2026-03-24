"""EO-UNITE: Unified tokenization and generation for multi-modal Earth Observation.

Adapts the UNITE framework (Duggal et al., 2025) to arbitrary-channel EO data:
- Multi-spectral patch embedding (Panopticon-style or dynamic hypernetwork)
- 2D ALiBi positional biases (from CROMA) replacing RoPE
- Wavelength + geolocation + time conditioning for generation
- Wavelength-conditioned decoder output (DynamicUnpatchify)
- Joint training: L1 reconstruction + LPIPS (S2RGB) + flow matching loss

Batch structure (unchanged from EO-VAE):
    batch['image']:    [B, C, H, W] — all samples same modality
    batch['wvs']:      [C] — wavelengths in µm
    batch['modality']: str — e.g. 'S2L2A', 'S1RTC', 'S2RGB'
    batch['lat']:      [B] — latitude (optional)
    batch['lon']:      [B] — longitude (optional)
    batch['time']:     [B] — day-of-year (optional)
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch import Tensor

from eo_vae.models.modules.alibi_2d import get_alibi
from eo_vae.models.modules.dynamic_unpatchify import build_unpatchify
from eo_vae.models.modules.eo_unite_patch_embed import build_patch_embed
from eo_vae.models.modules.modality_conditioner import ModalityConditioner
from eo_vae.models.unite_core.decoder import Decoder, DecoderConfig
from eo_vae.models.unite_core.denoiser import Sampler, Transport
from eo_vae.models.unite_core.encoder import Encoder
from eo_vae.models.unite_core.encoder_utils import get_1d_sincos_pos_embed_from_grid

ENCODER_CONFIGS = {
    'Base':  (12, 768, 12),   # depth, hidden_size, num_heads
    'Large': (24, 1152, 16),
}

DECODER_CONFIGS = {
    'Base': (8, 512, 16, 2048),  # layers, hidden_size, num_heads, mlp_dim
}


class EOUnite(L.LightningModule):
    """UNITE-based tokenizer + generator for multi-modal EO data.

    Args:
        encoder_model_size: 'Base' or 'Large'.
        decoder_model_size: 'Base'.
        num_latent_tokens: Number of learnable latent tokens.
        patch_size: Spatial patch size (must divide image size evenly).
        image_size: Expected training image size.
        diffusion_input_dim: Latent token channel dimension.
        patch_embed_type: 'panopticon' or 'dynamic'.
        patch_embed_cfg: Config dict for the patch embedding module.
        use_adain: Whether to use AdaIN-style wavelength conditioning in encoder.
        cond_dim: Modality conditioning vector dimension.
        loss_fn: EOUniteLoss instance.
        ckpt_path: Path to UNITE pretrained checkpoint (.pt) or full checkpoint (.ckpt).
        base_lr: Peak learning rate.
        final_lr: Final LR after cosine decay.
        warmup_epochs: Linear warmup epochs.
        decay_end_epoch: Cosine decay ends at this epoch.
        modulation_recon_timestep_max: Max timestep for reconstruction path (in [0,1]).
        noising_t_start: Fraction at which latent noising starts during reconstruction.
    """

    def __init__(
        self,
        encoder_model_size: str = 'Base',
        decoder_model_size: str = 'Base',
        num_latent_tokens: int = 256,
        patch_size: int = 16,
        image_size: int = 256,
        diffusion_input_dim: int = 32,
        patch_embed_type: str = 'panopticon',
        patch_embed_cfg: dict | None = None,
        unpatchify_cfg: dict | None = None,
        use_adain: bool = True,
        cond_dim: int = 256,
        loss_fn: nn.Module | None = None,
        ckpt_path: str | None = None,
        base_lr: float = 1e-4,
        final_lr: float = 1e-5,
        warmup_epochs: int = 1,
        decay_end_epoch: int = 100,
        modulation_recon_timestep_max: float = 0.3,
        noising_t_start: float = 0.7,
        lognorm_mu: float = 0.0,
        lognorm_sigma: float = 1.0,
        freeze_body: bool = False,
        unfreeze_epoch: int | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['loss_fn'])
        self.loss_fn = loss_fn

        encoder_depth, encoder_hidden, encoder_heads = ENCODER_CONFIGS[encoder_model_size]
        dec_layers, dec_hidden, dec_heads, dec_mlp = DECODER_CONFIGS[decoder_model_size]

        self.patch_size = patch_size
        self.image_size = image_size
        self.num_latent_tokens = num_latent_tokens
        self.diffusion_input_dim = diffusion_input_dim
        self.patch_embed_type = patch_embed_type
        self.modulation_recon_timestep_max = modulation_recon_timestep_max
        self.noising_t_start = noising_t_start

        num_patches = (image_size // patch_size) ** 2

        # --- Patch embedding ---
        self.patch_embed = build_patch_embed(patch_embed_type, patch_embed_cfg or {})

        # --- Modality conditioner (replaces class-label embedding) ---
        self.modality_conditioner = ModalityConditioner(cond_dim=cond_dim)

        # --- Conditioner projection to encoder hidden size ---
        self.cond_proj = nn.Linear(cond_dim, encoder_hidden)

        # --- Encoder (ViT with ALiBi) ---
        self.encoder = Encoder(
            input_size=image_size // patch_size,
            patch_size=1,
            in_channels=diffusion_input_dim,
            hidden_size=encoder_hidden,
            depth=encoder_depth,
            num_heads=encoder_heads,
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            num_classes=1000,  # unused when precomputed_cond is provided
            learn_sigma=False,
            use_swiglu=True,
            use_rope=False,
            use_alibi=True,
            use_rmsnorm=True,
            wo_shift=False,
            in_context_start=None,
            max_tokens=num_latent_tokens + num_patches,
            block_norm=True,
        )

        # LayerNorm applied to encoder output (latent normalization)
        self.encoder_ln = nn.LayerNorm(diffusion_input_dim, elementwise_affine=True)

        # Learnable latent tokens (sincos init)
        latent_init = get_1d_sincos_pos_embed_from_grid(
            encoder_hidden, np.arange(num_latent_tokens, dtype=np.float64)
        )
        self.latent_tokens = nn.Parameter(
            torch.from_numpy(latent_init).unsqueeze(0).float(), requires_grad=True
        )

        # --- Decoder ---
        decoder_config = DecoderConfig(
            hidden_size=encoder_hidden,
            patch_size=patch_size,
            decoder_hidden_size=dec_hidden,
            decoder_num_hidden_layers=dec_layers,
            decoder_num_attention_heads=dec_heads,
            decoder_intermediate_size=dec_mlp,
            num_channels=dec_hidden,  # placeholder; DynamicUnpatchify handles output channels
        )
        self.decoder = Decoder(decoder_config, num_patches=num_patches)

        # Upsample latent tokens to encoder_hidden for decoder input
        self.up_sample_decoder = nn.Linear(diffusion_input_dim, encoder_hidden, bias=True)

        # --- Dynamic decoder output ---
        _unp_cfg = unpatchify_cfg or {}
        if patch_embed_type == 'panopticon':
            _unp_cfg.setdefault('dec_dim', dec_hidden)
            _unp_cfg.setdefault('attn_dim', dec_hidden)
            _unp_cfg.setdefault('patch_size', patch_size)
        else:
            _unp_cfg.setdefault('dec_dim', dec_hidden)
            _unp_cfg.setdefault('patch_size', patch_size)
        self.unpatchify = build_unpatchify(patch_embed_type, _unp_cfg)

        # --- Flow matching transport ---
        self.transport = Transport(
            train_eps=5e-2,
            sample_eps=5e-2,
            use_cosine_loss=True,
            use_lognorm=True,
            lognorm_mu=lognorm_mu,
            lognorm_sigma=lognorm_sigma,
        )
        self.flow_sampler = Sampler(self.transport)

        # --- 2D ALiBi bias (precomputed, registered as buffer) ---
        # Shape: (1, H, num_latent_tokens + num_patches, num_latent_tokens + num_patches)
        # Latent token pairs have 0 bias (no spatial meaning).
        # Patch-to-patch pairs carry the Euclidean distance bias.
        patch_alibi = get_alibi(encoder_heads, num_patches)  # (1, H, P, P)
        N_full = num_latent_tokens + num_patches
        full_alibi = torch.zeros(1, encoder_heads, N_full, N_full)
        full_alibi[:, :, num_latent_tokens:, num_latent_tokens:] = patch_alibi
        self.register_buffer('alibi', full_alibi)

        # Load checkpoint if provided
        if ckpt_path is not None:
            self._load_checkpoint(ckpt_path)

        self._body_unfrozen = False
        if freeze_body:
            self._freeze_body()

    # ------------------------------------------------------------------
    # Core encode / decode
    # ------------------------------------------------------------------

    def encode(self, x: Tensor, wvs: Tensor, cond: Tensor | None = None) -> Tensor:
        """Encode images to latent tokens.

        Args:
            x: Images [B, C, H, W].
            wvs: Wavelengths [C].
            cond: Pre-computed conditioning [B, encoder_hidden]. If None, uses zeros.

        Returns:
            Latent tokens [B, num_latent_tokens, diffusion_input_dim].
        """
        B = x.shape[0]
        img_embed = self.patch_embed(x, wvs)  # [B, num_patches, encoder_hidden]

        # Noise tokens for latent positions
        z_noise = torch.randn(
            B, self.num_latent_tokens, self.diffusion_input_dim,
            device=x.device, dtype=x.dtype,
        )

        pos_embed = self.latent_tokens[:, :self.num_latent_tokens].expand(B, -1, -1)

        # Small timestep for reconstruction path (near-zero noise)
        flow_t = torch.rand(B, device=x.device) * self.modulation_recon_timestep_max

        z = self.encoder(
            z_noise,
            t=flow_t,
            pos_embed=pos_embed,
            img_patch_embed=img_embed,
            precomputed_cond=cond,
            alibi_bias=self.alibi,
        )
        # Encoder returns latent tokens + patch tokens; take only latent tokens
        return z[:, :self.num_latent_tokens]

    def decode(self, z: Tensor, wvs: Tensor) -> Tensor:
        """Decode latent tokens to image reconstruction.

        Args:
            z: Latent tokens [B, num_latent_tokens, diffusion_input_dim].
                Should be LayerNorm-normalized before calling.
            wvs: Target wavelengths [C].

        Returns:
            Reconstructed images [B, C, H, W].
        """
        # Upsample to decoder input dim
        z_up = self.up_sample_decoder(z)  # [B, L, encoder_hidden]

        # Decoder produces hidden states [B, num_patches, dec_hidden]
        patch_hidden = self.decoder(z_up, drop_cls_token=False, return_hidden=True)

        # Dynamic unpatchify: [B, num_patches, dec_hidden] → [B, C, H, W]
        return self.unpatchify(patch_hidden, wvs)

    def _noising(self, x: Tensor, sampling_prob: float = 0.5) -> Tensor:
        """Optionally corrupt latents during reconstruction training (from UNITE)."""
        B = x.shape[0]
        mask = (torch.rand(B, device=x.device) < sampling_prob).view(B, *([1] * (x.dim() - 1)))
        t, x0, x1 = self.transport.sample(x, sp_timesteps=[self.noising_t_start, 1.0])
        _, x_t, _ = self.transport.path_sampler.plan(t, x0, x1)
        return torch.where(mask, x_t, x)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        images = batch['image']          # [B, C, H, W]
        wvs = batch['wvs']               # [C]
        modality = batch.get('modality')

        # Build modality conditioning
        geo = self._build_geo(batch, images.device, images.dtype)
        time = batch.get('time')
        if time is not None:
            time = time.float()

        cond_raw = self.modality_conditioner(wvs, geo=geo, time=time)  # [B, cond_dim]
        cond = self.cond_proj(cond_raw)  # [B, encoder_hidden]

        # Tokenizer forward
        z = self.encode(images, wvs, cond=cond)
        z_normed = self.encoder_ln(z)

        # Optional latent noising (representation robustness, from UNITE)
        z_for_decode = self._noising(z_normed) if self.training else z_normed
        recon = self.decode(z_for_decode, wvs)

        # Loss
        total_loss, logs = self.loss_fn(
            recon=recon,
            target=images,
            wvs=wvs,
            modality=modality,
            z=z_normed,
            transport=self.transport,
            encoder=self.encoder,
            encoder_ln=self.encoder_ln,
            latent_tokens=self.latent_tokens,
            cond=cond,
            global_step=self.global_step,
            split='train',
            alibi_bias=self.alibi,
        )
        self.log_dict(logs, prog_bar=True, on_step=True, on_epoch=False)
        return total_loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        images = batch['image']
        wvs = batch['wvs']
        modality = batch.get('modality')

        geo = self._build_geo(batch, images.device, images.dtype)
        time = batch.get('time')
        if time is not None:
            time = time.float()

        cond_raw = self.modality_conditioner(wvs, geo=geo, time=time)
        cond = self.cond_proj(cond_raw)

        z = self.encode(images, wvs, cond=cond)
        z_normed = self.encoder_ln(z)
        recon = self.decode(z_normed, wvs)

        _, logs = self.loss_fn(
            recon=recon,
            target=images,
            wvs=wvs,
            modality=modality,
            z=z_normed,
            transport=self.transport,
            encoder=self.encoder,
            encoder_ln=self.encoder_ln,
            latent_tokens=self.latent_tokens,
            cond=cond,
            global_step=self.global_step,
            split='val',
            alibi_bias=self.alibi,
        )
        self.log_dict(logs, prog_bar=True, on_step=False, on_epoch=True)

    # ------------------------------------------------------------------
    # Generation (inference only)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        wvs: Tensor,
        geo: Tensor | None = None,
        time: Tensor | None = None,
        num_samples: int = 4,
        cfg_scale: float = 4.0,
        num_steps: int = 50,
    ) -> Tensor:
        """Generate EO images from modality conditioning via ODE sampling.

        Args:
            wvs: Target wavelengths [C].
            geo: Geolocation [B, 2] or None.
            time: Day-of-year [B] or None.
            num_samples: Number of samples to generate.
            cfg_scale: Classifier-free guidance scale.
            num_steps: ODE sampling steps.

        Returns:
            Generated images [B, C, H, W].
        """
        device = self.latent_tokens.device
        cond_raw = self.modality_conditioner(wvs, geo=geo, time=time)
        cond = self.cond_proj(cond_raw)

        noise = torch.randn(num_samples, self.num_latent_tokens, self.diffusion_input_dim, device=device)
        pos_embed = self.latent_tokens[:, :self.num_latent_tokens].expand(num_samples, -1, -1)

        sample_fn = self.flow_sampler.sample_ode(num_steps=num_steps)

        def _model_fn(x_t, t, **_):
            t_b = t.view(-1, 1, 1)
            denom = (1.0 - t_b).clamp_min(self.transport.sample_eps)
            x_pred = self.encoder(
                x_t, t=t, pos_embed=pos_embed,
                precomputed_cond=cond, alibi_bias=self.alibi,
            )
            x_pred = self.encoder_ln(x_pred)
            return (x_pred - x_t) / denom

        samples = sample_fn(noise, _model_fn)[-1]
        return self.decode(samples, wvs)

    # ------------------------------------------------------------------
    # Optimizer / LR schedule
    # ------------------------------------------------------------------

    def on_train_epoch_start(self) -> None:
        if (
            self.hparams.unfreeze_epoch is not None
            and not self._body_unfrozen
            and self.current_epoch >= self.hparams.unfreeze_epoch
        ):
            self._unfreeze_body()
            self._body_unfrozen = True
            body_params = (
                list(self.encoder.parameters())
                + list(self.encoder_ln.parameters())
                + list(self.decoder.parameters())
                + list(self.up_sample_decoder.parameters())
                + [self.latent_tokens]
            )
            current_lr = self.optimizers().param_groups[0]['lr']
            self.optimizers().add_param_group({'params': body_params, 'lr': current_lr})
            print(f'[EOUnite] Added body params to optimizer at epoch {self.current_epoch}, lr={current_lr:.2e}.')

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.hparams.base_lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, self._lr_lambda)
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}

    def _lr_lambda(self, epoch: int) -> float:
        warmup = self.hparams.warmup_epochs
        decay_end = self.hparams.decay_end_epoch
        base = self.hparams.base_lr
        final = self.hparams.final_lr

        if epoch < warmup:
            return (epoch + 1) / max(1, warmup)

        progress = (epoch - warmup) / max(1, decay_end - warmup)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return final / base + (1 - final / base) * cosine

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def _load_checkpoint(self, ckpt_path: str) -> None:
        import os
        if not os.path.exists(ckpt_path):
            print(f'[EOUnite] Checkpoint not found: {ckpt_path}')
            return

        if ckpt_path.endswith('.ckpt'):
            # Full Lightning checkpoint — load entire state dict
            state = torch.load(ckpt_path, map_location='cpu')
            sd = state.get('state_dict', state)
            missing, unexpected = self.load_state_dict(sd, strict=False)
            print(f'[EOUnite] Loaded .ckpt — missing: {len(missing)}, unexpected: {len(unexpected)}')

        elif ckpt_path.endswith('.pt'):
            # UNITE pretrained checkpoint: load encoder/decoder transformer weights
            # Skip patch_embed, unpatchify, and modality_conditioner (new modules)
            state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            sd = state.get('model', state)

            # Remap UNITE key names → EO-Unite key names
            RENAME = {
                'encoder_layer_norm.weight': 'encoder_ln.weight',
                'encoder_layer_norm.bias': 'encoder_ln.bias',
            }
            sd = {RENAME.get(k, k): v for k, v in sd.items()}

            # Filter to only keys/shapes that exist in this model
            own_sd = self.state_dict()
            filtered = {}
            for k, v in sd.items():
                if k in own_sd and own_sd[k].shape == v.shape:
                    filtered[k] = v

            missing, unexpected = self.load_state_dict(filtered, strict=False)
            print(f'[EOUnite] Loaded UNITE .pt — matched: {len(filtered)}, '
                  f'missing: {len(missing)}, unexpected: {len(unexpected)}')

    def _load_distilled_io_ckpt(self, ckpt_path: str) -> None:
        """Load distilled IO layer weights from a weight_distill_unite checkpoint."""
        import os
        if not os.path.exists(ckpt_path):
            print(f'[EOUnite] Distilled IO checkpoint not found: {ckpt_path}')
            return
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        self.patch_embed.load_state_dict(ckpt['patch_embed_state_dict'])
        self.unpatchify.load_state_dict(ckpt['unpatchify_state_dict'])
        self.modality_conditioner.load_state_dict(ckpt['modality_conditioner_state_dict'])
        self.cond_proj.load_state_dict(ckpt['cond_proj_state_dict'])
        print(f'[EOUnite] Loaded distilled IO layers from {ckpt_path}')

    # ------------------------------------------------------------------
    # Freeze / unfreeze body
    # ------------------------------------------------------------------

    def _freeze_body(self) -> None:
        """Freeze pretrained UNITE body; keep EO-specific I/O layers trainable."""
        for mod in [self.encoder, self.encoder_ln, self.decoder, self.up_sample_decoder]:
            for p in mod.parameters():
                p.requires_grad = False
        self.latent_tokens.requires_grad = False
        print('[EOUnite] Body frozen. Training I/O layers only.')

    def _unfreeze_body(self) -> None:
        """Restore requires_grad for all body parameters."""
        for p in self.parameters():
            p.requires_grad = True
        print('[EOUnite] Body unfrozen. Full end-to-end training.')

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_geo(self, batch: dict, device, dtype) -> Tensor | None:
        lat = batch.get('lat')
        lon = batch.get('lon')
        if lat is not None and lon is not None:
            return torch.stack([lat.float(), lon.float()], dim=-1).to(device)
        return None
