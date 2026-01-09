import torch
import torch.nn as nn
from azula.denoise import DiracPosterior, SimpleDenoiser
from torch import Tensor

import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class EOFrequencyLoss(nn.Module):
    def __init__(self, block_size=8, high_freq_weight=4.0):
        super().__init__()
        self.block_size = block_size

        # 1. DCT Matrix
        self.register_buffer('dct_mat', self._get_dct_matrix(block_size))

        # 2. Aggressive Frequency Weights
        # We use a steeper curve to force high-freq focus
        self.register_buffer(
            'freq_weights', self._get_steep_weights(block_size, high_freq_weight)
        )

    def _get_dct_matrix(self, N):
        n = torch.arange(N).float()
        k = torch.arange(N).float()
        n, k = torch.meshgrid(n, k, indexing='ij')
        dct_m = torch.cos(math.pi / N * (n + 0.5) * k)
        dct_m[:, 0] *= 1.0 / math.sqrt(2)
        dct_m *= math.sqrt(2 / N)
        return dct_m

    def _get_steep_weights(self, N, max_weight):
        u = torch.arange(N).float()
        v = torch.arange(N).float()
        dist = torch.sqrt(u**2 + v**2)
        dist_norm = dist / dist.max()
        # Quadratic ramp (steep) instead of linear
        return 1.0 + (max_weight - 1.0) * (dist_norm**2)

    def forward(self, pred, target):
        # 1. Pad & Blockify
        b, c, h, w = pred.shape
        p = self.block_size
        pad_h = (p - h % p) % p
        pad_w = (p - w % p) % p
        if pad_h > 0 or pad_w > 0:
            pred = F.pad(pred, (0, pad_w, 0, pad_h), mode='reflect')
            target = F.pad(target, (0, pad_w, 0, pad_h), mode='reflect')

        pred_blocks = pred.view(b, c, -1, p, w // p if pad_w == 0 else -1, p).permute(
            0, 1, 2, 4, 3, 5
        )
        target_blocks = target.view(
            b, c, -1, p, w // p if pad_w == 0 else -1, p
        ).permute(0, 1, 2, 4, 3, 5)

        # 2. DCT Transform
        diff_spatial = pred_blocks - target_blocks
        dct_cols = torch.matmul(self.dct_mat, diff_spatial)
        diff_freq = torch.matmul(dct_cols, self.dct_mat.t())  # Raw Frequency Difference

        # 3. Log-L1 Loss Logic (The Fix)
        # We don't just weight the difference. We weight the MAGNITUDE.
        # But simply diffing logs is unstable.
        # Better: Apply L1 Loss to the Weighted Raw Differences

        weighted_diff = diff_freq * self.freq_weights

        # L1 Loss is much better for high-frequency restoration (sparsity)
        loss = torch.abs(weighted_diff).mean()

        return loss


class EODenoiser(SimpleDenoiser):
    """
    Extends SimpleDenoiser with an auxiliary Frequency-Aware Loss.
    """

    def __init__(
        self,
        backbone: nn.Module,
        schedule: nn.Module,  # Type hint was Schedule in your code
        freq_weight: float = 0.5,  # Lambda for the aux loss
        high_freq_penalty: float = 3.0,
    ):
        super().__init__(backbone, schedule)
        self.freq_weight = freq_weight
        # Initialize the loss module
        self.freq_loss_fn = EOFrequencyLoss(
            block_size=8, high_freq_weight=high_freq_penalty
        )

    def loss(
        self, x: torch.Tensor, t: torch.Tensor, max_weight: float = 1e4, **kwargs
    ) -> torch.Tensor:
        # 1. Standard Flow Matching / Diffusion Setup
        alpha_t, sigma_t = self.schedule(t)

        while alpha_t.ndim < x.ndim:
            alpha_t, sigma_t = alpha_t[..., None], sigma_t[..., None]

        # Generate Noisy Input
        z = torch.randn_like(x)
        x_t = alpha_t * x + sigma_t * z

        # 2. Get Prediction (Posterior)
        # kwargs (like 'condition') are passed to forward -> backbone
        q = self(x_t, t, **kwargs)
        x_pred = q.mean

        # 3. Compute Standard Spatial Loss (Weighted MSE)
        w_t = (alpha_t / sigma_t) ** 2 + 1
        w_t = torch.clip(w_t, max=max_weight)

        # loss_spatial = (w_t * (x_pred - x).square()).mean()
        # drop weighting for L1 loss
        loss_spatial = torch.nn.functional.l1_loss(x_pred, x)

        # 4. Compute Frequency Loss
        # We generally do NOT apply the diffusion weight w_t to this auxiliary loss,
        # or we scale it separately. DeCo adds it as a raw penalty.
        if self.freq_weight > 0:
            loss_freq = self.freq_loss_fn(x_pred, x)
            return loss_spatial + self.freq_weight * loss_freq

        return loss_spatial


class ResidualEODenoiser(SimpleDenoiser):
    """
    Extends SimpleDenoiser with:
    1. Residual Skip Connection (The Fix for Negative Gain)
    2. Robust Frequency Loss
    """

    def __init__(
        self,
        backbone: nn.Module,
        schedule: nn.Module,
        freq_weight: float = 0.5,
        high_freq_penalty: float = 3.0,
    ):
        super().__init__(backbone, schedule)
        self.freq_weight = freq_weight
        self.freq_loss_fn = EOFrequencyLoss(
            block_size=8, high_freq_weight=high_freq_penalty
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, condition=None, **kwargs):
        """
        Modified Forward Pass:
        Prediction = Network_Output(Residual) + Condition(Base)
        """
        # 1. Resolve 'condition' vs 'cond' collision
        # If the sampler passed 'cond' in kwargs, use it and remove it from kwargs
        if condition is None and 'cond' in kwargs:
            condition = kwargs.pop('cond')

        # 2. Run the Backbone
        # We explicitly pass 'cond=condition' so the backbone receives it correctly.
        # We pass the cleaned '**kwargs' (without 'cond') to avoid duplicates.
        posterior = super().forward(x_t, t, cond=condition, **kwargs)
        residual_prediction = posterior.mean

        # 3. THE FIX: Add the Base Image
        # If the backbone outputs 0 (at init), the result is the Blurry Condition.
        if condition is not None:
            final_prediction = residual_prediction + condition
        else:
            final_prediction = residual_prediction

        # Return as Posterior
        return DiracPosterior(mean=final_prediction)

    def loss(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        max_weight: float = 1e4,
        condition=None,
        **kwargs,
    ) -> torch.Tensor:
        # 1. Standard Noise Injection
        alpha_t, sigma_t = self.schedule(t)
        while alpha_t.ndim < x.ndim:
            alpha_t, sigma_t = alpha_t[..., None], sigma_t[..., None]

        z = torch.randn_like(x)
        x_t = alpha_t * x + sigma_t * z

        # 2. Get Prediction (Calls the NEW forward method above)
        # It will return: Backbone(x_t, cond) + cond
        q = self(x_t, t, condition=condition, **kwargs)
        x_pred = q.mean

        # 3. Robust Loss Calculation
        # We are comparing (Blurry + Residual) vs (Sharp Ground Truth)

        # Unweighted L1 for stability
        loss_spatial = torch.nn.functional.l1_loss(x_pred, x)

        # Frequency Loss
        if self.freq_weight > 0:
            loss_freq = self.freq_loss_fn(x_pred, x)
            return loss_spatial + self.freq_weight * loss_freq

        return loss_spatial
