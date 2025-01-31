# Based on https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/autoencoder.py
# which is released under MIT-License
# - Adopted to support newer torch/lightning versions
# - added docstrings and type annotations
# - adopted for anysensor inputs


import torch
import os
from lightning import LightningModule
import torch.nn.functional as F
from typing import Any
from torchvision.datasets.utils import download_url

from .modules.distributions import DiagonalGaussianDistribution


class AutoencoderKL(LightningModule):
    # that checkpoint works together with this config
    # https://github.com/CompVis/latent-diffusion/blob/main/configs/autoencoder/autoencoder_kl_32x32x4.yaml
    hf_url = 'https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt'

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        loss_fn: torch.nn.Module,
        embed_dim: int,
        ckpt_path: str | None = None,
        ignore_keys: list[str] = [],
        image_key: str = 'image',
        colorize_nlabels: int | None = None,
        monitor: str | None = None,
    ) -> None:
        """Initialize the KL-regularized Autoencoder.

        Args:
            # ddconfig: Configuration dictionary for encoder/decoder architecture
            lossconfig: Configuration dictionary for loss function
            embed_dim: Dimension of the latent embedding space
            ckpt_path: Path to checkpoint file for loading pretrained weights
            ignore_keys: List of keys to ignore when loading checkpoint
            image_key: Key used to access images in the input batch
            colorize_nlabels: Number of labels for colorization feature
            monitor: Metric to monitor during training

        Returns:
            None
        """
        super().__init__()
        self.image_key = image_key
        self.encoder = encoder
        self.decoder = decoder
        self.loss = loss_fn

        self.z_channels = self.encoder.z_channels
        self.quant_conv = torch.nn.Conv2d(2 * self.z_channels, 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, self.z_channels, 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer('colorize', torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()) -> None:
        """
        Load model weights from a checkpoint file.

        Args:
            path: Path to the checkpoint file containing model weights
            ignore_keys: List of keys to ignore when loading the state dict

        Returns:
            None
        """
        if path is None:
            path = 'vae-ft-mse-840000-ema-pruned.ckpt'
            # get the Huggingface checkpoint
            if not os.path.exists(path):
                download_url(self.hf_url, '.', path)

        sd = torch.load(path, map_location='cpu')['state_dict']
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print('Deleting key {} from state_dict.'.format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f'Restored from {path}')

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        """
        Encode input tensor to latent representation.

        Args:
            x: Input tensor to encode [B, C, H, W]

        Returns:
            Posterior distribution in latent space
        """
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to image space.

        Args:
            z: Latent representation to decode [B, D, H, W]

        Returns:
            Decoded tensor in image space [B, C, H, W]
        """
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(
        self, input: torch.Tensor, sample_posterior: bool = True
    ) -> tuple[torch.Tensor, DiagonalGaussianDistribution]:
        """
        Forward pass through the autoencoder.

        Args:
            input: Input tensor [B, C, H, W]
            sample_posterior: Whether to sample from posterior or take mode

        Returns:
            Tuple of (decoded tensor, posterior distribution)
        """
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch: dict[str, torch.Tensor], k: str) -> torch.Tensor:
        """
        Extract and process input tensor from batch.

        Args:
            batch: Dictionary containing batch data
            k: Key to extract from batch

        Returns:
            Processed input tensor [B, C, H, W]
        """
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, optimizer_idx: int
    ) -> torch.Tensor:
        """
        Training step for autoencoder.

        Args:
            batch: Input batch dictionary
            batch_idx: Index of current batch
            optimizer_idx: Index of optimizer (0: AE, 1: Discriminator)

        Returns:
            Loss value for current step
        """
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(
                inputs,
                reconstructions,
                posterior,
                optimizer_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split='train',
            )
            self.log(
                'aeloss',
                aeloss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            self.log_dict(
                log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False
            )
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(
                inputs,
                reconstructions,
                posterior,
                optimizer_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split='train',
            )

            self.log(
                'discloss',
                discloss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            self.log_dict(
                log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False
            )
            return discloss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """
        Validation step for autoencoder.

        Args:
            batch: Input batch dictionary
            batch_idx: Index of current batch

        Returns:
            Dictionary of logged metrics
        """
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(
            inputs,
            reconstructions,
            posterior,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split='val',
        )

        discloss, log_dict_disc = self.loss(
            inputs,
            reconstructions,
            posterior,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split='val',
        )

        self.log('val/rec_loss', log_dict_ae['val/rec_loss'])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list]:
        """
        Configure optimizers for autoencoder and discriminator.

        Returns:
            Tuple of (list of optimizers, list of schedulers)
        """
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quant_conv.parameters())
            + list(self.post_quant_conv.parameters()),
            lr=lr,
            betas=(0.5, 0.9),
        )
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9)
        )
        # TODO learning rate schedulers?
        return [opt_ae, opt_disc], []

    def get_last_layer(self) -> torch.Tensor:
        """
        Get weights of the last decoder layer.

        Returns:
            Weight tensor of final convolution layer
        """
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(
        self, batch: dict[str, torch.Tensor], only_inputs: bool = False, **kwargs
    ) -> dict[str, torch.Tensor]:
        """
        Generate images for logging during training.

        Args:
            batch: Input batch dictionary
            only_inputs: If True, only log input images
            **kwargs: Additional arguments

        Returns:
            Dictionary containing input, reconstructed and sampled images
        """
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log['samples'] = self.decode(torch.randn_like(posterior.sample()))
            log['reconstructions'] = xrec
        log['inputs'] = x
        return log

    def to_rgb(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert segmentation masks to RGB images.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            RGB tensor [B, 3, H, W] with values in [-1, 1]
        """
        assert self.image_key == 'segmentation'
        if not hasattr(self, 'colorize'):
            self.register_buffer('colorize', torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
        return x
