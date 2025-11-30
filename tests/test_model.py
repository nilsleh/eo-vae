import os

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


class TestVAE:
    @pytest.fixture
    def model(self):
        config_path = os.path.join('configs', 'test_config.yaml')

        config = OmegaConf.load(config_path)

        model = instantiate(config.model)

        return model

    def test_forward(self, model):
        """Test forward pass of VAE model."""
        x = torch.randn(1, 3, 224, 224)
        wv = torch.randn(3)

        recon, posterior = model(x, wv)
        assert isinstance(recon, torch.Tensor)
        assert recon.shape == x.shape

    # def test_training_step(self, model):
    #     """Test training step of VAE model."""
    #     # DOES not work with manual optimization
    #     batch = {'image': torch.randn(1, 3, 224, 224), 'wvs': torch.randn(3)}

    #     gen_loss, disc_loss = model.training_step(batch, 0)
    #     assert isinstance(gen_loss, torch.Tensor)
    #     assert isinstance(disc_loss, torch.Tensor)
