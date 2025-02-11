import torch
import os
from omegaconf import OmegaConf
from hydra.utils import instantiate

from src.models import AutoencoderKL
from src.datasets import get_flair_dataloader
from PIL import Image
import numpy as np
import pdb
from torchvision.transforms import ToPILImage


to_pil = ToPILImage()
config_path = os.path.join(os.getcwd(), 'configs', 'test_config.yaml')
config = OmegaConf.load(config_path)
model = instantiate(config.model).cuda()


dataloader = get_flair_dataloader(batch_size=1)

for idx, sample in enumerate(dataloader):
    x = sample['image']
    pdb.set_trace()
    x = x.permute(0, 3, 1, 2)
    pil_image = to_pil(x.squeeze())
    pil_image.save(f'img_{idx}.png')

    x = x.cuda()
    wvs = torch.FloatTensor([0.665, 0.56, 0.49]).to(x.device)
    x_2 = torch.ones([1, 3, 512, 512]).cuda()
    x_2[:, 0, ...] = x[:, 0, ...]
    x_2[:, 1, ...] = x[:, 0, ...]
    x_2[:, 2, ...] = x[:, 0, ...]
    out, _ = model(x_2, wvs)
    out = out[:, :3, ...]
    pil_image = to_pil(out.squeeze())
    pil_image.save(f'decode_img_{idx}.png')
