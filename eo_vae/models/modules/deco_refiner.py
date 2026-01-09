import torch
import torch.nn as nn


class AdaGN(nn.Module):
    """Adaptive Group Norm: Modulates features based on Time."""

    def __init__(self, dim, time_dim, groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, dim)
        self.proj = nn.Linear(time_dim, dim * 2)  # Predicts Scale & Shift

    def forward(self, x, t_emb):
        scale, shift = self.proj(t_emb).chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return self.norm(x) * (1 + scale) + shift


class ResBlockAdaGN(nn.Module):
    def __init__(self, dim, time_dim):
        super().__init__()
        self.norm1 = AdaGN(dim, time_dim)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.norm2 = AdaGN(dim, time_dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.conv1(self.act(self.norm1(x, t_emb)))
        h = self.conv2(self.act(self.norm2(h, t_emb)))
        return x + h


class DeCoPixelDecoder(nn.Module):
    """
    Lightweight Refiner Backbone.
    Input: x_t (Noisy)
    Kwargs: 'condition' (Blurry VAE Recon)
    """

    def __init__(self, channels, hidden_dim=128, num_blocks=6):
        super().__init__()

        # Input accepts: [Noisy_Image (C) | Condition (C)] -> 2*C Channels
        self.input_conv = nn.Conv2d(channels * 2, hidden_dim, kernel_size=3, padding=1)

        # Time Embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

        # Deep Backbone
        self.blocks = nn.ModuleList(
            [ResBlockAdaGN(hidden_dim, time_dim=hidden_dim) for _ in range(num_blocks)]
        )

        self.final_norm = nn.GroupNorm(8, hidden_dim)
        self.output_conv = nn.Conv2d(hidden_dim, channels, kernel_size=3, padding=1)

        # Zero-init output for stability
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, x, t, cond):
        # 2. Concatenate [Noisy | Blurry]
        # x: [B, C, H, W], condition: [B, C, H, W]
        x_cat = torch.cat([x, cond], dim=1)

        # 3. Initial Features
        h = self.input_conv(x_cat)

        # 4. Time Embedding
        t_emb = self.time_mlp(t.view(-1, 1))

        # 5. Residual Blocks
        for block in self.blocks:
            h = block(h, t_emb)

        h = self.final_norm(h)
        return self.output_conv(h)
