import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DiffusionDynamicInput(nn.Module):
    def __init__(self, out_channels, embed_dim, kernel_size=3, use_bias=True):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # HyperNetwork: Generates a unique (D_out, 1, K, K) kernel for EACH band
        # Conditioned on joint [Time + Wavelength]
        self.weight_gen = nn.Sequential(
            nn.Linear(embed_dim, out_channels * 4),
            nn.SiLU(),
            nn.Linear(out_channels * 4, out_channels * (kernel_size**2)),
        )

        if use_bias:
            self.bias_gen = nn.Linear(embed_dim, out_channels)
        else:
            self.register_parameter('bias_gen', None)

    def forward(self, x, t_emb, wv_embs):
        """x: (B, N, H, W) - Input bands (e.g. 3 for RGB, 13 for S2)
        t_emb: (B, D_emb) - Global time embedding
        wv_embs: (B, N, D_emb) - Individual wavelength embeddings
        """
        B, N, H, W = x.shape

        # 1. Joint Conditioning: Fuse time and wavelength
        # (B, N, D_emb)
        ctx = wv_embs + t_emb.unsqueeze(1)

        # 2. Weight Generation
        # (B, N, D_out * K*K) -> (B, N, D_out, 1, K, K)
        weights = self.weight_gen(ctx)
        weights = rearrange(
            weights,
            'b n (d k1 k2) -> (b n) d 1 k1 k2',
            d=self.out_channels,
            k1=self.kernel_size,
            k2=self.kernel_size,
        )

        # 3. Dynamic Convolution
        # We process all bands in the batch dimension temporarily for speed
        x_flat = rearrange(x, 'b n h w -> 1 (b n) h w')

        # Grouped convolution: each band gets its own generated kernel
        # We use groups=B*N so each 'channel' in x_flat gets one weight from weights
        # But we want D_out channels as result per band.
        # Easier way: process via loop or sophisticated einsum/conv2d

        # Efficient approach: sum of depthwise convs
        x_in = rearrange(x, 'b n h w -> (b n) 1 h w')
        out = F.conv2d(x_in, weights, padding=self.padding, groups=1)
        # out: (BN, D_out, H, W)

        # 4. Global Fusion (Sum across bands) + Bias
        out = rearrange(out, '(b n) d h w -> b n d h w', b=B)
        out = out.sum(dim=1)  # (B, D_out, H, W)

        if self.bias_gen is not None:
            # Aggregate bias from all bands (or use a global time bias)
            bias = self.bias_gen(ctx).sum(dim=1)  # (B, D_out)
            out = out + bias.view(B, -1, 1, 1)

        return out


class DiffusionDynamicOutput(nn.Module):
    def __init__(self, in_channels, embed_dim, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # HyperNetwork: Generates (1, D_in, K, K) kernel for EACH output band
        self.weight_gen = nn.Sequential(
            nn.Linear(embed_dim, in_channels * 4),
            nn.SiLU(),
            nn.Linear(in_channels * 4, in_channels * (kernel_size**2)),
        )

        # Zero-init the last layer so the initial velocity is 0
        nn.init.zeros_(self.weight_gen[-1].weight)
        nn.init.zeros_(self.weight_gen[-1].bias)

    def forward(self, latent, t_emb, wv_embs):
        """latent: (B, D_in, H, W) - From UViT backbone
        t_emb: (B, D_emb)
        wv_embs: (B, N, D_emb)
        """
        B, D_in, H, W = latent.shape
        _, N, _ = wv_embs.shape

        ctx = wv_embs + t_emb.unsqueeze(1)  # (B, N, D_emb)

        # Generate kernels: (B*N, 1, D_in, K, K)
        weights = self.weight_gen(ctx)
        weights = rearrange(
            weights,
            'b n (d k1 k2) -> (b n) 1 d k1 k2',
            d=self.in_channels,
            k1=self.kernel_size,
            k2=self.kernel_size,
        )

        # Repeat latent for each of the N output bands
        # (B, N, D_in, H, W) -> (B*N, D_in, H, W)
        latent_expanded = repeat(latent, 'b d h w -> (b n) d h w', n=N)

        # Project back to 1 channel per band
        out = F.conv2d(latent_expanded, weights, padding=self.padding, groups=1)

        # Reshape to final multispectral cube
        # (B*N, 1, H, W) -> (B, N, H, W)
        return rearrange(out, '(b n) 1 h w -> b n h w', b=B)
