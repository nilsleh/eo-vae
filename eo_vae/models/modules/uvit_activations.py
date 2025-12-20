# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import nn

from .uvit_torch_utils import ACTIVATIONS


def get_linear_activation(activation_fn: str, dim: int, *, inner_dim=None, bias=True):
    activation_fn = activation_fn.lower()
    if activation_fn == 'gelu':
        assert inner_dim is not None, 'inner_dim must be provided for GELU activation'
        return GELU(dim, inner_dim, bias=bias)
    elif activation_fn == 'gelu-approximate':
        assert inner_dim is not None, 'inner_dim must be provided for GELU activation'
        return GELU(dim, inner_dim, approximate='tanh', bias=bias)
    elif activation_fn == 'geglu':
        assert inner_dim is not None, 'inner_dim must be provided for GEGLU activation'
        return GEGLU(dim, inner_dim, bias=bias)
    elif activation_fn == 'geglu-approximate':
        assert inner_dim is not None, 'inner_dim must be provided for GEGLU activation'
        return ApproximateGELU(dim, inner_dim, bias=bias)
    elif activation_fn == 'swiglu':
        assert inner_dim is not None, 'inner_dim must be provided for SwiGLU activation'
        return SwiGLU(dim, inner_dim, bias=bias)
    elif activation_fn == 'linear-silu':
        assert inner_dim is not None, (
            'inner_dim must be provided for LinearActivation activation'
        )
        return LinearActivation(dim, inner_dim, bias=bias, activation='silu')
    else:
        raise ValueError(f'Unknown activation function: {activation_fn}.')


class GELU(nn.Module):
    # From https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/activations.py
    r"""GELU activation function with tanh approximation support with `approximate="tanh"`.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        approximate (`str`, *optional*, defaults to `"none"`): If `"tanh"`, use tanh approximation.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(
        self, dim_in: int, dim_out: int, approximate: str = 'none', bias: bool = True
    ):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.approximate = approximate

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        return F.gelu(gate, approximate=self.approximate)  # pylint: disable=E1102

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states


class GEGLU(nn.Module):
    # From https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/activations.py
    r"""A [variant](https://arxiv.org/abs/2002.05202) of the gated linear unit activation function.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        return F.gelu(gate)  # pylint: disable=E1102

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)


class SwiGLU(nn.Module):
    # From https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/activations.py
    r"""A [variant](https://arxiv.org/abs/2002.05202) of the gated linear unit activation function. It's similar to `GEGLU`
    but uses SiLU / Swish instead of GeLU.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()

        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)
        self.activation = nn.SiLU()

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        return hidden_states * self.activation(gate)


class ApproximateGELU(nn.Module):
    # From https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/activations.py
    r"""The approximate form of the Gaussian Error Linear Unit (GELU). For more details, see section 2 of this
    [paper](https://arxiv.org/abs/1606.08415).

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x * torch.sigmoid(1.702 * x)


class LinearActivation(nn.Module):
    # From https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/activations.py
    def __init__(
        self, dim_in: int, dim_out: int, bias: bool = True, activation: str = 'silu'
    ):
        super().__init__()

        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.activation = ACTIVATIONS[activation]()

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        return self.activation(hidden_states)
