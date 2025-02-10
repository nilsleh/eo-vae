import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange
from torch import Tensor

import pdb

random_seed = 1234
torch.manual_seed(random_seed)


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim: int, pos: Tensor) -> Tensor:
    """Generate sinusoidal positional embeddings.

    Args:
        embed_dim: Output dimension for each position (must be even)
        pos: Positions to be encoded [M,]

    Returns:
        Positional embeddings [M, D]
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb

class TransformerWeightGenerator(nn.Module):
    """Transformer-based dynamic weight generator.
    
    Generates weights and biases for dynamic convolutions using a transformer architecture.
    """

    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        embed_dim: int, 
        num_heads: int = 4, 
        num_layers: int = 1
    ) -> None:
        """Initialize the weight generator.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output weight matrix
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
        """
        super(TransformerWeightGenerator, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            activation="gelu",
            norm_first=False,
            batch_first=False,
            dropout=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Linear layer to map transformer output to desired weight shape
        self.fc_weight = nn.Linear(input_dim, output_dim)
        self.fc_bias = nn.Linear(input_dim, embed_dim)
        self.wt_num = 128
        self.weight_tokens = nn.Parameter(torch.empty([self.wt_num, input_dim]))
        self.bias_token = nn.Parameter(torch.empty([1, input_dim]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is
        # too big (2.)
        torch.nn.init.normal_(self.weight_tokens, std=0.02)
        torch.nn.init.normal_(self.bias_token, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate dynamic weights and biases.

        Args:
            x: Input features [seq_len, batch, input_dim]

        Returns:
            Tuple of:
                - Generated weights [seq_len, output_dim]
                - Generated bias [embed_dim]
        """
        # x should have shape [seq_len, batch, input_dim]
        pos_wave = x
        x = torch.cat([self.weight_tokens, pos_wave], dim=0)
        x = torch.cat([x, self.bias_token], dim=0)
        transformer_output = self.transformer_encoder(x)
        weights = self.fc_weight(transformer_output[self.wt_num : -1] + pos_wave)
        bias = self.fc_bias(
            transformer_output[-1]
        )  # Using the last output to generate bias
        return weights, bias


class TransformerWeightGenerator_decoder(TransformerWeightGenerator):
    """Decoder version of transformer-based weight generator with modified bias generation.
    
    Inherits from TransformerWeightGenerator but modifies the bias generation to output
    a single channel.
    """
    
    def __init__(self, input_dim: int, output_dim: int, embed_dim: int, 
                 num_heads: int = 4, num_layers: int = 1) -> None:
        """Initialize the decoder weight generator.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output weight matrix
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
        """
        super(TransformerWeightGenerator_decoder, self).__init__(input_dim, output_dim, embed_dim, num_heads=num_heads, num_layers=num_layers)
        self.fc_bias = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate weights and single-channel bias.

        Args:
            x: Input features [seq_len, batch, input_dim]

        Returns:
            Tuple of:
                - Generated weights [seq_len, output_dim]
                - Generated bias [1]
        """
        # x should have shape [seq_len, batch, input_dim]
        pos_wave = x
        x = torch.cat([self.weight_tokens, pos_wave], dim=0)
        x = torch.cat([x, self.bias_token], dim=0)
        transformer_output = self.transformer_encoder(x)
        weights = self.fc_weight(transformer_output[self.wt_num : -1] + pos_wave)
        bias = self.fc_bias(
            transformer_output[self.wt_num : -1] + self.bias_token.repeat((pos_wave.shape[0],1))
        )  # Using the last output to generate bias
        return weights, bias


class Basic1d(nn.Module):
    """Basic 1D convolutional block with optional layer norm and ReLU activation."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True) -> None:
        """Initialize the 1D conv block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            bias: Whether to include bias in linear layer
        """
        super().__init__()
        conv = nn.Linear(in_channels, out_channels, bias)
        self.conv = nn.Sequential(
            conv,
        )
        if not bias:
            self.conv.add_module("ln", nn.LayerNorm(out_channels))
        self.conv.add_module("relu", nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through conv block.

        Args:
            x: Input tensor [B, in_channels]

        Returns:
            Output tensor [B, out_channels]
        """
        out = self.conv(x)
        return out


class FCResLayer(nn.Module):
    """Fully connected residual layer with ReLU activations."""

    def __init__(self, linear_size: int = 128) -> None:
        """Initialize the FC residual layer.

        Args:
            linear_size: Size of linear layers
        """
        super(FCResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual layer.

        Args:
            x: Input tensor [B, linear_size]

        Returns:
            Output tensor [B, linear_size] with residual connection
        """
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y
        return out

class DynamicConv(nn.Module):
    """Dynamic convolution layer with wavelength-dependent kernels."""

    def __init__(self, wv_planes: int, inter_dim: int = 128, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, embed_dim: int = 128) -> None:
        """Initialize dynamic convolution.

        Args:
            wv_planes: Dimension of wavelength features
            inter_dim: Intermediate dimension size
            kernel_size: Size of convolution kernel
            stride: Convolution stride
            padding: Convolution padding
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self._num_kernel = self.kernel_size * self.kernel_size * self.embed_dim
        self.inter_dim = inter_dim
        self.patch_size = (kernel_size, kernel_size)
        self.num_patches = -1
        self.stride = stride
        self.padding = padding

        self.weight_generator = TransformerWeightGenerator(
            wv_planes, self._num_kernel, embed_dim
        )
        self.scaler = 0.01

        self.fclayer = FCResLayer(wv_planes)

        self._init_weights()

    def _get_weights(self, waves):
        dynamic_weights = self.weight_generator(waves)

        return dynamic_weights

    def weight_init(self, m: nn.Module) -> None:
        """Initialize weights of linear layers using Xavier initialization.

        Args:
            m: Module to initialize
        """
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _init_weights(self) -> None:
        """Initialize base weights and dynamic MLP weights."""
        self.weight_generator.apply(self.weight_init)
        self.fclayer.apply(self.weight_init)

    def forward(self, img_feat: torch.Tensor, wvs: torch.Tensor) -> torch.Tensor:
        """Apply dynamic convolution.

        Args:
            img_feat: Input image features [B, C, H, W]
            wvs: Wavelength values [num_wavelengths]

        Returns:
            Convolved features [B, embed_dim, H', W']
        """
        inplanes = wvs.size(0)
        # wv_feats: 9,128 -> 9, 3x3x3
        waves = get_1d_sincos_pos_embed_from_grid_torch(self.wv_planes, wvs * 1000)
        waves = self.fclayer(waves)
        weight, bias = self._get_weights(waves)  # 3x3x3

        # small bug fixed
        dynamic_weight = weight.view(inplanes, self.kernel_size, self.kernel_size, self.embed_dim)
        dynamic_weight = dynamic_weight.permute([3,0,1,2])

        if bias is not None:
            bias = bias.view([self.embed_dim]) * self.scaler

        weights = dynamic_weight * self.scaler

        dynamic_out = F.conv2d(img_feat, weights, bias=bias, stride=(self.stride, self.stride), padding=self.padding)

        return dynamic_out


class DynamicConv_decoder(nn.Module):
    """Decoder version of dynamic convolution with modified bias handling."""

    def __init__(self, wv_planes: int, inter_dim: int = 128, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, embed_dim: int = 128) -> None:
        """Initialize decoder dynamic convolution.

        Args:
            wv_planes: Dimension of wavelength features
            inter_dim: Intermediate dimension size
            kernel_size: Size of convolution kernel
            stride: Convolution stride
            padding: Convolution padding
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self._num_kernel = self.kernel_size * self.kernel_size * self.embed_dim
        self.inter_dim = inter_dim
        self.patch_size = (kernel_size, kernel_size)
        self.num_patches = -1
        self.stride = stride
        self.padding = padding

        self.weight_generator = TransformerWeightGenerator_decoder(
            wv_planes, self._num_kernel, embed_dim
        )
        self.scaler = 0.1

        self.fclayer = FCResLayer(wv_planes)

        self._init_weights()

    def _get_weights(self, waves):
        dynamic_weights = self.weight_generator(waves)

        return dynamic_weights

    def weight_init(self, m: nn.Module) -> None:
        """Initialize weights of linear layers using Xavier initialization.

        Args:
            m: 
        """
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _init_weights(self) -> None:
        """Initialize base weights and dynamic MLP weights."""
        self.weight_generator.apply(self.weight_init)
        self.fclayer.apply(self.weight_init)

    def forward(self, img_feat: torch.Tensor, waves: torch.Tensor) -> torch.Tensor:
        """Apply decoder dynamic convolution.

        Args:
            img_feat: Input image features [B, C, H, W]
            waves: Wavelength values [num_wavelengths]

        Returns:
            Convolved features [B, num_wavelengths, H', W']
        """
        inplanes = waves.size(0)
        #wv_feats: 9,128 -> 9, 3x3x3
        self.scaler = 0.1
        waves = get_1d_sincos_pos_embed_from_grid_torch(self.wv_planes, waves*1000)
        waves = self.fclayer(waves)
        weight,bias = self._get_weights(waves) #3x3x3

        # small bug fixed
        dynamic_weight = weight.view(inplanes, self.kernel_size, self.kernel_size, self.embed_dim)
        dynamic_weight = dynamic_weight.permute([0,3,1,2])

        if bias is not None:
            bias = bias.squeeze() * self.scaler

        weights = dynamic_weight * self.scaler
        bias = bias * self.scaler

        dynamic_out = F.conv2d(img_feat, weights, bias=bias, stride=(self.stride, self.stride), padding=self.padding)

        return dynamic_out

if __name__ == "__main__":
    embed_dim = 768
    #dconv = DynamicConv(wv_planes=128, inter_dim=128, kernel_size=3, embed_dim=embed_dim).cuda()
    inp = torch.randn([1,3,224,224]).cuda()
    conv_in = torch.nn.Conv2d(
        3, 128, kernel_size=3, stride=1, padding=1
    ).cuda()
    print(conv_in(inp).shape)
    inp = torch.randn([1,8,224,224]).cuda()
    dconv_in = DynamicConv(
            wv_planes=128, inter_dim=128, kernel_size=3, stride=1, padding=1, embed_dim=128
    ).cuda()
    wvs = torch.FloatTensor([0.665, 0.56, 0.49, 0.665, 0.56, 0.49, 0.665, 0.56]).to(inp.device)
    print(dconv_in(inp,wvs).shape)

    inp = torch.randn([1,128,224,224]).cuda()
    dconv_decoder = DynamicConv_decoder(
        wv_planes=128, inter_dim=128, kernel_size=3, stride=1, padding=1, embed_dim=128
    ).cuda()
    print(dconv_decoder(inp, wvs).shape)
