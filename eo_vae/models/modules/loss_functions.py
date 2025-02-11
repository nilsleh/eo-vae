# MIT License

# Copyright (c) 2022 Machine Vision and Learning Group, LMU Munich

# Based on: https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/losses/contperceptual.py

# Adapted to work with DOFA model and input wavelengths

import torch
import torch.nn as nn
from typing import Any, Tuple
from torch import Tensor

from .loss_utils import LPIPS, hinge_d_loss, vanilla_d_loss, adopt_weight


class LPIPSWithDiscriminator(nn.Module):
    """Combined LPIPS and GAN loss with adaptive weighting."""

    def __init__(
        self,
        perceptual_loss: nn.Module,
        discriminator: nn.Module,
        disc_start: int,
        logvar_init: float = 0.0,
        kl_weight: float = 1.0,
        pixelloss_weight: float = 1.0,
        disc_num_layers: int = 3,
        disc_in_channels: int = 3,
        disc_factor: float = 1.0,
        disc_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        use_actnorm: bool = False,
        disc_conditional: bool = False,
        disc_loss: str = 'hinge',
    ) -> None:
        """Initialize combined loss function.

        Args:
            perceptual_loss: Perceptual loss module, should be the LPIPS module
                with a desired pretrained net
            discriminator: Discriminator module
            disc_start: Step to start discriminator training
            logvar_init: Initial log variance
            kl_weight: Weight for KL divergence term
            pixelloss_weight: Weight for pixel reconstruction loss
            disc_num_layers: Number of discriminator layers
            disc_in_channels: Number of discriminator input channels
            disc_factor: Factor for discriminator loss
            disc_weight: Weight for discriminator contribution
            perceptual_weight: Weight for perceptual loss
            use_actnorm: Whether to use activation normalization
            disc_conditional: Whether discriminator is conditional
            disc_loss: Type of discriminator loss ("hinge" or "vanilla")
        """

        super().__init__()
        assert disc_loss in ['hinge', 'vanilla']
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = perceptual_loss
        self.perceptual_loss.eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = discriminator
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == 'hinge' else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(
        self, nll_loss: Tensor, g_loss: Tensor, last_layer: nn.Module | None = None
    ) -> Tensor:
        """Calculate adaptive weight for discriminator loss.

        Args:
            nll_loss: Negative log likelihood loss
            g_loss: Generator loss
            last_layer: Last layer for adaptive weight calculation

        Returns:
            Adaptive weight value
        """
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(
                nll_loss, self.last_layer[0], retain_graph=True
            )[0]
            g_grads = torch.autograd.grad(
                g_loss, self.last_layer[0], retain_graph=True
            )[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(
        self,
        inputs: Tensor,
        reconstructions: Tensor,
        posteriors: Any,
        optimizer_idx: int,
        global_step: int,
        wvs: Tensor,
        last_layer: torch.nn.Parameter | None = None,
        cond: Tensor | None = None,
        split: str = 'train',
        weights: Tensor | None = None,
    ) -> tuple[Tensor, dict]:
        """Calculate combined loss.

        Args:
            inputs: Input images
            reconstructions: Reconstructed images
            posteriors: Posterior distributions
            optimizer_idx: Index of optimizer (0=generator, 1=discriminator)
            global_step: Current training step
            wvs: Wavelengths of input images for perceptual loss
            last_layer: Last layer for adaptive weight calculation
            cond: Conditional input if using conditional GAN
            split: Dataset split name
            weights: Optional weights for loss terms

        Returns:
            Tuple of:
                - Combined loss value
                - Dictionary of logging information
        """
        rec_loss = torch.abs(
            inputs.contiguous() - reconstructions.contiguous()
        )  # L1 Loss
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(
                input=inputs.contiguous(), target=reconstructions.contiguous(), wvs=wvs
            )
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights * nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous(), wvs)
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.contiguous(), cond), dim=1), wvs
                )

            # This needs to be the Generator?
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(
                        nll_loss, g_loss, last_layer=last_layer
                    )
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            loss = (
                weighted_nll_loss
                + self.kl_weight * kl_loss
                + d_weight * disc_factor * g_loss
            )

            log = {
                '{}/total_loss'.format(split): loss.clone().detach().mean(),
                '{}/logvar'.format(split): self.logvar.detach(),
                '{}/kl_loss'.format(split): kl_loss.detach().mean(),
                '{}/nll_loss'.format(split): nll_loss.detach().mean(),
                '{}/rec_loss'.format(split): rec_loss.detach().mean(),
                '{}/d_weight'.format(split): d_weight.detach(),
                '{}/disc_factor'.format(split): torch.tensor(disc_factor),
                '{}/g_loss'.format(split): g_loss.detach().mean(),
            }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach(), wvs)
                logits_fake = self.discriminator(
                    reconstructions.contiguous().detach(), wvs
                )
            else:
                logits_real = self.discriminator(
                    torch.cat((inputs.contiguous().detach(), cond), dim=1), wvs
                )
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.contiguous().detach(), cond), dim=1), wvs
                )

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                '{}/disc_loss'.format(split): d_loss.clone().detach().mean(),
                '{}/logits_real'.format(split): logits_real.detach().mean(),
                '{}/logits_fake'.format(split): logits_fake.detach().mean(),
            }
            return d_loss, log
