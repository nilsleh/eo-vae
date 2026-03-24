"""2D ALiBi (Attention with Linear Biases) positional encoding.

Ported from CROMA (Fuller et al., NeurIPS 2023):
https://github.com/antofuller/CROMA/blob/main/pretrain_croma.py

Instead of adding positional embeddings to tokens, ALiBi adds a relative
distance-based bias directly to attention logits before softmax. This enables
extrapolation to image sizes larger than those seen during training without
any modification.
"""

import itertools
import math

import torch
from torch import Tensor


def get_alibi(attention_heads: int, num_patches: int) -> Tensor:
    """Compute 2D ALiBi bias matrix for a square grid of patches.

    Args:
        attention_heads: Number of attention heads.
        num_patches: Total number of patches (must be a perfect square).

    Returns:
        Bias tensor of shape (1, attention_heads, num_patches, num_patches).
        Negative Euclidean distances between patch positions, scaled by
        head-specific slopes. Added to attention logits before softmax.
    """
    grid_size = int(math.sqrt(num_patches))
    assert grid_size * grid_size == num_patches, 'num_patches must be a perfect square'
    points = list(itertools.product(range(grid_size), range(grid_size)))

    slopes = torch.Tensor(_get_slopes(attention_heads)).unsqueeze(1)  # (H, 1)

    idxs = []
    for p1 in points:
        for p2 in points:
            dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            idxs.append(dist * slopes * -1)  # (H, 1) each

    all_bias = torch.cat(idxs, dim=1)  # (H, num_patches * num_patches)
    return all_bias.view(1, attention_heads, num_patches, num_patches)


def _get_slopes(n: int) -> list[float]:
    """Compute ALiBi head slopes following the power-of-2 scheme."""

    def _get_slopes_power_of_2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return _get_slopes_power_of_2(n)

    closest_power_of_2 = 2 ** math.floor(math.log2(n))
    return _get_slopes_power_of_2(closest_power_of_2) + _get_slopes(
        2 * closest_power_of_2
    )[0::2][: n - closest_power_of_2]
