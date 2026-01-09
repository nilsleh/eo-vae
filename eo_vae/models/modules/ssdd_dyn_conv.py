import torch
import torch.nn as nn
import torch.nn.functional as F
from .dynamic_conv import FCResLayer, get_1d_sincos_pos_embed_from_grid_torch


class SSDD_WeightGenerator(nn.Module):
    """
    Stabilized Weight Generator for SSDD.
    Increased depth (3 layers) allows for more complex spectral/temporal mapping.
    """

    def __init__(
        self,
        input_dim,
        kernel_size,
        embed_dim,
        mode='encoder',
        latent_channels=3,
        time_embed_dim=None,
        num_heads=4,
        num_layers=3,
        use_latents=False,
        spatial_grid=4,
    ):
        super().__init__()
        self.mode = mode
        self.input_dim = input_dim
        self.use_latents = use_latents
        self.kernel_dim = kernel_size * kernel_size
        self.embed_dim = embed_dim
        self.latent_channels = latent_channels
        self.spatial_grid = spatial_grid

        # Increased depth for better conditioning logic
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                activation='gelu',
                batch_first=False,
                norm_first=True,  # norm_first is more stable for deep stacks
            ),
            num_layers=num_layers,
        )

        self.time_proj = nn.Linear(
            time_embed_dim if time_embed_dim else input_dim, input_dim
        )

        # Learning shared context tokens
        self.wt_num = 64
        self.weight_tokens = nn.Parameter(torch.randn(self.wt_num, input_dim) * 0.02)

        if mode == 'encoder':
            # Only initialize latent tokens if we intend to use latents
            if use_latents:
                self.latent_chan_tokens = nn.Parameter(
                    torch.randn(latent_channels, input_dim) * 0.02
                )
            self.fc_weight = nn.Linear(input_dim, self.kernel_dim * embed_dim)
            self.fc_bias = nn.Linear(input_dim, embed_dim)
        else:
            # Decoder produces N kernels: [embed_dim -> 1]
            self.fc_weight = nn.Linear(input_dim, self.kernel_dim * embed_dim)
            self.fc_bias = nn.Linear(input_dim, 1)

        if use_latents:
            self.spatial_tokenizer = nn.Sequential(
                nn.Conv2d(latent_channels, 32, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.AdaptiveAvgPool2d(spatial_grid),
            )
            self.latent_to_dim = nn.Linear(64, input_dim)
            self.latent_pos_embed = nn.Parameter(
                torch.randn(1, spatial_grid**2, input_dim) * 0.02
            )

    def forward(self, wv_emb, t_emb, latents=None):
        # B is our target batch size (e.g., 16)
        B = t_emb.shape[0]

        if wv_emb.dim() == 1:
            # Case: [N] (just raw wavelengths) -> needs embedding first!
            # If you are passing raw wvs here, ensure they are embedded
            # to [N, D] before calling the generator.
            # FIX: Use input_dim (from weight_tokens) instead of embed_dim
            input_dim = self.weight_tokens.shape[-1]
            wv_emb = wv_emb.unsqueeze(-1).expand(-1, input_dim)

        if wv_emb.dim() == 2:
            # Case: [N, D] (Your current case: shared across batch)
            # -> [N, 1, D] -> [N, B, D]
            w_toks = wv_emb.unsqueeze(1).expand(-1, B, -1)
            N = wv_emb.shape[0]
        elif wv_emb.dim() == 3:
            # Case: [B, N, D] (Batched wavelengths)
            # -> [N, B, D]
            w_toks = wv_emb.permute(1, 0, 2)
            N = w_toks.shape[1]

        t_tok = self.time_proj(t_emb).unsqueeze(0)
        gen_toks = self.weight_tokens.unsqueeze(1).repeat(1, B, 1)

        tokens = [gen_toks, w_toks, t_tok]

        # Only append latent channel tokens if we are in encoder mode, configured to use latents, AND latents are provided
        use_latent_tokens = (
            (self.mode == 'encoder') and self.use_latents and (latents is not None)
        )

        if use_latent_tokens:
            tokens.append(self.latent_chan_tokens.unsqueeze(1).repeat(1, B, 1))

        if self.use_latents and latents is not None:
            s_feat = self.spatial_tokenizer(latents).flatten(2).permute(2, 0, 1)
            tokens.append(
                self.latent_to_dim(s_feat) + self.latent_pos_embed.permute(1, 0, 2)
            )

        x = self.transformer_encoder(torch.cat(tokens, dim=0))

        # Map features to weights
        # Dynamically determine the slice end based on whether latent tokens were used
        num_latent_out = self.latent_channels if use_latent_tokens else 0

        target_features = x[self.wt_num : self.wt_num + N + num_latent_out]
        weights = self.fc_weight(target_features).permute(1, 0, 2)

        if self.mode == 'encoder':
            bias = self.fc_bias(x[0])  # Global hidden bias
        else:
            bias = self.fc_bias(target_features).squeeze(-1).permute(1, 0)

        return weights, bias


def weight_standardization(weight, eps=1e-5):
    """Prevents signal explosion by centering kernel weights."""
    mean = weight.mean(dim=[1, 2, 3], keepdim=True)
    var = weight.var(dim=[1, 2, 3], keepdim=True)
    return (weight - mean) / (torch.sqrt(var + eps))


class SSDD_OutputConv(nn.Module):
    def __init__(self, wv_planes=128, embed_dim=96, time_embed_dim=None, kernel_size=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.generator = SSDD_WeightGenerator(
            wv_planes,
            kernel_size,
            embed_dim,
            mode='decoder',
            time_embed_dim=time_embed_dim,
            use_latents=False,
        )
        self.fclayer = FCResLayer(wv_planes)
        self.scaler = 0.1

    def forward(self, x, wvs, t_emb):
        B, C_hid, H, W = x.shape
        C_out = wvs.shape[0]
        waves = get_1d_sincos_pos_embed_from_grid_torch(128, wvs * 1000).to(x.device)
        waves = self.fclayer(waves)

        weights, bias = self.generator(waves, t_emb)

        # Reshape and Standardize
        weights = weights.reshape(B * C_out, C_hid, self.kernel_size, self.kernel_size)
        weights = weight_standardization(weights)  # CRITICAL for artifact reduction

        bias = bias.reshape(B * C_out) * self.scaler

        feat = x.view(1, B * C_hid, H, W)
        # Apply kernels with scaling
        out = F.conv2d(
            feat,
            weights * self.scaler,
            bias=bias,
            groups=B,
            padding=self.kernel_size // 2,
        )

        return out.view(B, C_out, H, W)


class SSDD_InputConv(nn.Module):
    def __init__(
        self,
        wv_planes=128,
        embed_dim=96,
        time_embed_dim=None,
        latent_channels=64,
        use_latents=True,
        kernel_size=3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.generator = SSDD_WeightGenerator(
            wv_planes,
            kernel_size,
            embed_dim,
            mode='encoder',
            time_embed_dim=time_embed_dim,
            use_latents=use_latents,
            latent_channels=latent_channels,
        )
        self.fclayer = FCResLayer(wv_planes)
        self.scaler = 0.1

    def forward(self, x_spec, wvs, t_emb, latents=None):
        B, C_spec, H, W = x_spec.shape
        if latents is None:
            x_combined = x_spec
        else:
            x_combined = torch.cat([x_spec, latents], dim=1)
        C_total = x_combined.shape[1]

        waves = get_1d_sincos_pos_embed_from_grid_torch(128, wvs * 1000).to(
            x_spec.device
        )
        waves = self.fclayer(waves)

        weights, bias = self.generator(waves, t_emb, latents=latents)

        # Reshape from flat output to 5D tensor before permuting
        weights = weights.reshape(
            B, C_total, self.embed_dim, self.kernel_size, self.kernel_size
        )

        # Standardize and map to [B*Out, In, K, K]
        weights = weights.permute(0, 2, 1, 3, 4).reshape(
            B * self.embed_dim, C_total, self.kernel_size, self.kernel_size
        )
        weights = weight_standardization(weights)

        bias = bias.reshape(B * self.embed_dim) * self.scaler

        feat = x_combined.reshape(1, B * C_total, H, W)
        out = F.conv2d(
            feat,
            weights * self.scaler,
            bias=bias,
            groups=B,
            padding=self.kernel_size // 2,
        )

        return out.view(B, self.embed_dim, H, W)


class ModulatedSSDD_OutputConv(nn.Module):
    def __init__(
        self,
        wv_planes=128,
        embed_dim=96,
        out_channels=16,
        time_embed_dim=None,
        kernel_size=3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.wv_planes = wv_planes

        # Generator now predicts a scaling vector of size 'embed_dim' instead of full kernels
        self.generator = SSDD_WeightGenerator(
            wv_planes,
            kernel_size=1,  # We only need a 1D scaling vector per channel
            embed_dim=embed_dim,
            mode='decoder',
            time_embed_dim=time_embed_dim,
            use_latents=False,
        )
        self.fclayer = FCResLayer(wv_planes)

        # Fixed base weights that represent high-frequency "templates"
        # Changed to [1, embed_dim, k, k] to act as a shared basis for any number of output channels
        self.weight = nn.Parameter(torch.randn(1, embed_dim, kernel_size, kernel_size))

    def forward(self, x, wvs, t_emb):
        B, C_hid, H, W = x.shape
        C_out = wvs.shape[0]  # Modality-specific channels (e.g., 10 for S2, 2 for S1)

        waves = get_1d_sincos_pos_embed_from_grid_torch(self.wv_planes, wvs * 1000).to(
            x.device
        )
        waves = self.fclayer(waves)

        # Get modulation coefficients from the hypernetwork
        # weights shape: [B, C_out, embed_dim]
        # Pass 'waves' (embedded), not 'wvs' (raw)
        mod_coeffs, bias = self.generator(waves, t_emb)
        mod_coeffs = mod_coeffs.view(B, C_out, C_hid, 1, 1)

        # 1. Modulate: Scale the base weights by the predicted coefficients
        # self.weight: [1, C_hid, K, K] -> unsqueeze(0) -> [1, 1, C_hid, K, K]
        # mod_coeffs: [B, C_out, C_hid, 1, 1]
        # Result: [B, C_out, C_hid, K, K]
        weights = self.weight.unsqueeze(0) * mod_coeffs

        # 2. Demodulate (Standardization): Prevents signal explosion
        # This is a more robust version of your weight_standardization
        dcoefs = torch.rsqrt(weights.pow(2).sum([2, 3, 4]) + 1e-8)
        weights = weights * dcoefs.view(B, C_out, 1, 1, 1)

        # Prepare for Grouped Convolution
        # Use reshape instead of view to handle non-contiguous memory from broadcasting
        weights = weights.reshape(B * C_out, C_hid, self.kernel_size, self.kernel_size)
        # use reshape for bias as well
        bias = bias.reshape(B * C_out)

        feat = x.view(1, B * C_hid, H, W)
        out = F.conv2d(
            feat, weights, bias=bias, groups=B, padding=self.kernel_size // 2
        )

        return out.view(B, C_out, H, W)
