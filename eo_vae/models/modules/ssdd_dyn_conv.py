import torch
import torch.nn as nn
import torch.nn.functional as F

from .dynamic_conv import FCResLayer, get_1d_sincos_pos_embed_from_grid_torch


class SSDD_WeightGenerator(nn.Module):
    """Refined Weight Generator for SSDD.
    Generates kernels for both variable spectral bands and fixed latent channels.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        embed_dim,
        mode='encoder',
        latent_channels=64,
        time_embed_dim=None,
        num_heads=4,
        num_layers=1,
        use_latents=False,
        spatial_grid=4,
    ):
        super().__init__()
        self.mode = mode
        self.use_latents = use_latents
        self.wt_num = 128
        self.spatial_grid = spatial_grid
        self.latent_channels = latent_channels

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, activation='gelu', batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        t_dim = time_embed_dim if time_embed_dim is not None else input_dim
        self.time_proj = nn.Linear(t_dim, input_dim)

        # LEARNED TOKENS for the latent channels (since they have no wavelength)
        if mode == 'encoder':
            self.latent_chan_tokens = nn.Parameter(
                torch.randn(latent_channels, input_dim) * 0.02
            )

        if use_latents:
            self.latent_proj = nn.Sequential(
                nn.Conv2d(latent_channels, 32, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(spatial_grid),
            )
            self.latent_to_dim = nn.Linear(64, input_dim)
            self.latent_pos_embed = nn.Parameter(
                torch.randn(1, spatial_grid * spatial_grid, input_dim) * 0.02
            )

        self.fc_weight = nn.Linear(input_dim, output_dim)

        if mode == 'encoder':
            self.fc_bias = nn.Linear(input_dim, embed_dim)
        else:
            self.fc_bias = nn.Linear(input_dim, 1)

        self.weight_tokens = nn.Parameter(torch.empty([self.wt_num, input_dim]))
        self.bias_token = nn.Parameter(torch.empty([1, input_dim]))
        torch.nn.init.normal_(self.weight_tokens, std=0.02)
        torch.nn.init.normal_(self.bias_token, std=0.02)

    def forward(self, wv_emb, t_emb, latents=None):
        B = t_emb.shape[0]
        N = wv_emb.shape[0]

        t_token = self.time_proj(t_emb).unsqueeze(0)
        w_tokens = wv_emb.unsqueeze(1).repeat(1, B, 1)
        gen_tokens = self.weight_tokens.unsqueeze(1).repeat(1, B, 1)
        b_token = self.bias_token.unsqueeze(1).repeat(1, B, 1)

        tokens = [gen_tokens, w_tokens, b_token, t_token]

        # Add the specific tokens for the latent/RGB input channels
        if self.mode == 'encoder':
            l_chan_toks = self.latent_chan_tokens.unsqueeze(1).repeat(1, B, 1)
            tokens.append(l_chan_toks)

        if self.use_latents and latents is not None:
            l_feat = self.latent_proj(latents)
            l_tokens = l_feat.flatten(2).permute(2, 0, 1)
            l_tokens = self.latent_to_dim(l_tokens)
            l_tokens = l_tokens + self.latent_pos_embed.permute(1, 0, 2)
            tokens.append(l_tokens)

        x = torch.cat(tokens, dim=0)
        out = self.transformer_encoder(x)

        # 1. Extract Kernels
        # For Encoder, we need kernels for N spectral bands + 3 latent channels
        if self.mode == 'encoder':
            # Identify the output features for spectral + latent channels
            spec_out = out[self.wt_num : self.wt_num + N]
            # Offset for latent channel tokens (they were appended after b_token and t_token)
            lat_start = self.wt_num + N + 2
            lat_out = out[lat_start : lat_start + self.latent_channels]

            kernel_features = torch.cat(
                [spec_out + w_tokens, lat_out + self.latent_chan_tokens.unsqueeze(1)],
                dim=0,
            )
            weights = self.fc_weight(kernel_features)  # [N+3, B, Output_Dim]
            bias = self.fc_bias(out[self.wt_num + N])  # Global bias for embed_dim
        else:
            wv_out = out[self.wt_num : self.wt_num + N]
            weights = self.fc_weight(wv_out + w_tokens)
            bias_features = wv_out + b_token
            bias = self.fc_bias(bias_features).squeeze(-1).permute(1, 0)

        return weights.permute(1, 0, 2), bias


class SSDD_InputConv(nn.Module):
    def __init__(
        self,
        wv_planes=128,
        embed_dim=128,
        time_embed_dim=None,
        kernel_size=3,
        latent_channels=64,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        # Generator maps variable spectral + 3 latents -> embed_dim
        self.generator = SSDD_WeightGenerator(
            wv_planes,
            kernel_size * kernel_size * embed_dim,
            embed_dim,
            mode='encoder',
            time_embed_dim=time_embed_dim,
            latent_channels=latent_channels,
            use_latents=True,
        )
        self.fclayer = FCResLayer(wv_planes)
        self.scaler = 0.1

    def forward(self, x_spec, wvs, t_emb, latents):
        B, C_spec, H, W = x_spec.shape
        # Rigorous Fusion: Process both as input data
        x_combined = torch.cat([x_spec, latents], dim=1)
        C_total = x_combined.shape[1]

        waves = get_1d_sincos_pos_embed_from_grid_torch(128, wvs * 1000).to(
            x_spec.device
        )
        waves = self.fclayer(waves)

        # Generator now returns C_total kernels
        weights, bias = self.generator(waves, t_emb, latents=latents)

        # Reshape for Grouped Conv: [B * embed_dim, C_total, K, K]
        weights = weights.reshape(
            B, C_total, self.embed_dim, self.kernel_size, self.kernel_size
        )
        weights = weights.permute(0, 2, 1, 3, 4).reshape(
            B * self.embed_dim, C_total, self.kernel_size, self.kernel_size
        )
        bias = bias.reshape(B * self.embed_dim) * self.scaler

        feat = x_combined.view(1, B * C_total, H, W)
        out = F.conv2d(
            feat,
            weights * self.scaler,
            bias=bias,
            groups=B,
            padding=self.kernel_size // 2,
        )

        return out.view(B, self.embed_dim, H, W)


class SSDD_OutputConv(nn.Module):
    def __init__(
        self, wv_planes=128, embed_dim=128, time_embed_dim=None, kernel_size=3
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        # Generator maps embed_dim -> variable spectral output
        self.generator = SSDD_WeightGenerator(
            wv_planes,
            kernel_size * kernel_size * embed_dim,
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

        # Reshape for Grouped Conv: [B * C_out, embed_dim, K, K]
        weights = weights.reshape(B * C_out, C_hid, self.kernel_size, self.kernel_size)
        bias = bias.reshape(B * C_out) * self.scaler

        feat = x.view(1, B * C_hid, H, W)
        out = F.conv2d(
            feat,
            weights * self.scaler,
            bias=bias,
            groups=B,
            padding=self.kernel_size // 2,
        )

        return out.view(B, C_out, H, W)
