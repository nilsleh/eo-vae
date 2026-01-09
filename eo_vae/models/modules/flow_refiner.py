import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor


from .ssdd_dyn_conv import SSDD_InputConv, SSDD_OutputConv, ModulatedSSDD_OutputConv
from .embeddings import TimestepEmbedding, Timesteps


class FlowRefiner(nn.Module):
    def __init__(self, wv_planes=128, embed_dim=96, kernel_size=3):
        super().__init__()

        time_embed_dim = embed_dim * 2
        self.time_proj = Timesteps(
            embed_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedding = TimestepEmbedding(
            embed_dim, time_embed_dim, act_fn='silu'
        )

        self.input_conv = SSDD_InputConv(
            wv_planes, embed_dim, time_embed_dim, kernel_size, use_latents=False
        )
        self.output_conv = ModulatedSSDD_OutputConv(
            wv_planes=wv_planes,
            embed_dim=embed_dim,
            time_embed_dim=time_embed_dim,
            kernel_size=kernel_size,
        )

    def get_time_embed(
        self, sample: torch.Tensor, timestep: torch.Tensor | float
    ) -> torch.Tensor:
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                [timesteps], dtype=torch.float64, device=sample.device
            )
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)
        return t_emb

    def forward(self, x, t, wvs):
        t_emb = self.get_time_embed(sample=x, timestep=t)
        t_emb = self.time_embedding(t_emb)

        h = self.input_conv(x, wvs, t_emb)
        out = self.output_conv(h, wvs, t_emb)
        return out
