import pytest
from omegaconf import OmegaConf
import os
from hydra.utils import instantiate
import torch



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
        wv = torch.randn(1, 3)

        recon, posterior = model(x, wv)
        assert isinstance(recon, torch.Tensor)
        assert recon.shape == x.shape

# import pickle
# out = pickle.load(open("/mnt/rg_climate_benchmark/data/cc_benchmark/classification_v1.0/m-so2sat/task_specs.pkl", "rb"))

# import pdb
# pdb.set_trace()

# print(0)