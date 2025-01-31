import torch
import os
from omegaconf import OmegaConf
from hydra.utils import instantiate

from src.models import AutoencoderKL


config_path = os.path.join(os.getcwd(), 'configs', 'test_config.yaml')

config = OmegaConf.load(config_path)

model = instantiate(config.model)

x = torch.randn(1, 3, 32, 32)

out = model(x)

import pdb

pdb.set_trace()

print(0)
