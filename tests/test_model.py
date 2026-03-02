import os

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from eo_vae.models import Decoder, Encoder, EOFluxVAE
from eo_vae.models.modules import (
    DiagonalGaussianDistribution,
    EOConsistencyLoss,
    EOVAVAELoss,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_encoder(use_adain=False):
    kwargs = {'num_layers': 2, 'wv_planes': 64}
    if use_adain:
        kwargs['use_adain'] = True
    return Encoder(
        resolution=64,
        in_channels=3,
        ch=32,
        ch_mult=[1, 2, 2],
        num_res_blocks=1,
        z_channels=4,
        use_dynamic_ops=True,
        dynamic_conv_kwargs=kwargs,
    )


def _make_decoder(use_adain=False):
    kwargs = {'num_layers': 2, 'wv_planes': 64}
    if use_adain:
        kwargs['use_adain'] = True
    return Decoder(
        ch=32,
        out_ch=3,
        ch_mult=[1, 2, 2],
        num_res_blocks=1,
        resolution=64,
        z_channels=4,
        use_dynamic_ops=True,
        dynamic_conv_kwargs=kwargs,
    )


def _make_vae(use_adain=False, freeze_body=False):
    return EOFluxVAE(
        encoder=_make_encoder(use_adain=use_adain),
        decoder=_make_decoder(use_adain=use_adain),
        loss_fn=EOConsistencyLoss(pixel_weight=1.0),
        freeze_body=freeze_body,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVAE:
    @pytest.fixture
    def model(self):
        config_path = os.path.join('configs', 'test_config.yaml')
        config = OmegaConf.load(config_path)
        return instantiate(config.model)

    def test_forward(self, model):
        """Forward pass matches input shape."""
        x = torch.randn(2, 3, 64, 64)
        wv = torch.tensor([0.49, 0.56, 0.665])

        recon, posterior = model(x, wv)
        assert isinstance(recon, torch.Tensor)
        assert recon.shape == x.shape

    def test_forward_adain(self):
        """Smoke test: use_adain forward pass produces correct output shape."""
        model = _make_vae(use_adain=True)
        model.eval()

        x = torch.randn(2, 3, 64, 64)
        wv = torch.tensor([0.49, 0.56, 0.665])

        with torch.no_grad():
            recon, posterior = model(x, wv)

        assert recon.shape == x.shape

    def test_kl_loss(self):
        """EOVAVAELoss adds KL term and logs it correctly."""
        loss_fn = EOVAVAELoss(pixel_weight=1.0, kl_weight=1e-4)

        B, C, H, W = 2, 3, 64, 64
        inputs = torch.randn(B, C, H, W)
        recon = torch.randn(B, C, H, W)
        wvs = torch.tensor([0.49, 0.56, 0.665])

        # Build a DiagonalGaussianDistribution from random moments
        moments = torch.randn(B, C * 2, H, W)
        posterior = DiagonalGaussianDistribution(moments)

        total_loss, logs = loss_fn(
            inputs=inputs,
            wvs=wvs,
            reconstructions=recon,
            global_step=0,
            split='train',
            posterior=posterior,
        )

        assert torch.isfinite(total_loss)
        assert 'train/loss_kl' in logs
        assert torch.isfinite(logs['train/loss_kl'])

    def test_freeze_adain(self):
        """With freeze_body=True + use_adain, conditioner and emb_proj are trainable, body is not."""
        model = _make_vae(use_adain=True, freeze_body=True)

        # conditioner must be trainable
        assert list(model.encoder.conditioner.parameters())[0].requires_grad
        assert list(model.decoder.conditioner.parameters())[0].requires_grad

        # emb_proj inside ResnetBlocks must be trainable
        assert model.encoder.down[0].block[0].emb_proj.weight.requires_grad
        assert model.decoder.mid.block_1.emb_proj.weight.requires_grad

        # Body norms/convs must be frozen
        assert not model.encoder.down[0].block[0].norm1.weight.requires_grad
        assert not model.decoder.mid.block_1.norm1.weight.requires_grad
