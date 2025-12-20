import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .dynamic_conv import get_1d_sincos_pos_embed_from_grid_torch


class SpectralFiLMBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        # Use a smaller number of groups for GroupNorm to save memory
        self.norm1 = nn.GroupNorm(4, dim)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=1)

        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, dim * 2))

        self.norm2 = nn.GroupNorm(4, dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        h = self.norm1(x)
        mod = self.modulation(cond).chunk(2, dim=1)
        gamma, beta = (
            mod[0].unsqueeze(-1).unsqueeze(-1),
            mod[1].unsqueeze(-1).unsqueeze(-1),
        )
        h = h * (1 + gamma) + beta
        h = self.act(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        return x + h


class SpectralFlowRefiner(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        cond_dim: int = 256,
        n_blocks: int = 4,
        factor: int = 4,
    ):
        super().__init__()
        self.factor = factor

        # 1. Pixel Unshuffle reduces H,W by 'factor' and increases channels by factor^2
        # For factor=4: 1 channel -> 16 channels
        self.unshuffle = nn.PixelUnshuffle(factor)

        # 2. Condition Integrator
        self.cond_mlp = nn.Sequential(
            nn.Linear(128 + 1, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )

        # 3. Refinement Backbone
        # Input channels = factor^2 (e.g., 16 for factor 4)
        self.input_proj = nn.Conv2d(factor**2, hidden_dim, 1)

        self.blocks = nn.ModuleList(
            [SpectralFiLMBlock(hidden_dim, cond_dim) for _ in range(n_blocks)]
        )

        # Output channels = factor^2 to be shuffled back to 1
        self.output_proj = nn.Conv2d(hidden_dim, factor**2, 1)
        self.shuffle = nn.PixelShuffle(factor)

    def forward(self, x_t: Tensor, t: Tensor, wvs: Tensor, **kwargs) -> Tensor:
        B, N, H, W = x_t.shape

        # Flatten and Downscale spatially (Lossless)
        # (B, N, H, W) -> (B*N, 1, H, W) -> (B*N, 16, H/4, W/4)
        x_t_flat = rearrange(x_t, 'b n h w -> (b n) 1 h w')
        x_t_down = self.unshuffle(x_t_flat)

        # 1. Conditioning
        wv_emb = get_1d_sincos_pos_embed_from_grid_torch(128, wvs * 1000)
        wv_emb = repeat(wv_emb, 'n d -> b n d', b=B)
        t_emb = t.view(B, 1, 1).expand(-1, N, -1)

        full_cond = torch.cat([wv_emb, t_emb], dim=-1)
        full_cond = self.cond_mlp(rearrange(full_cond, 'b n d -> (b n) d'))

        # 2. Refinement pass (at 1/16th the spatial memory cost)
        h = self.input_proj(x_t_down)
        for block in self.blocks:
            if self.training:
                h = checkpoint(block, h, full_cond, use_reentrant=False)
            else:
                h = block(h, full_cond)

        # 3. Upscale back to original resolution
        out_down = self.output_proj(h)
        out = self.shuffle(out_down)

        return rearrange(out, '(b n) 1 h w -> b n h w', b=B)
