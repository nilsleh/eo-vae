"""UNITE Encoder adapted for EO-UNITE.

Key changes from the original UNITE encoder:
1. ALiBi support: `use_alibi=True` replaces RoPE with 2D ALiBi positional biases.
   The `alibi_bias` tensor is passed through all blocks and added to attention logits.
2. Precomputed conditioning: `precomputed_cond` in forward() bypasses the
   LabelEmbedder, accepting the ModalityConditioner output directly. This replaces
   class-label conditioning with wavelength+geo+time conditioning.
3. RoPE is disabled by default for EO-UNITE (set use_rope=False).

Original: ShivamDuggal4/UNITE-tokenization-generation/modules/encoder.py
"""

import math

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp
from torch.utils.checkpoint import checkpoint

from eo_vae.models.unite_core.encoder_utils import (
    GaussianFourierEmbedding,
    LabelEmbedder,
    NormAttention,
    RMSNorm,
    SwiGLUFFN,
    VisionRotaryEmbeddingFast,
    get_2d_sincos_pos_embed,
    modulate,
)


class Block(nn.Module):
    """Transformer block with AdaLN modulation.

    Accepts `alibi_bias` for 2D ALiBi positional encoding (EO-UNITE),
    or `feat_rope` for RoPE (original UNITE).
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_qknorm=False,
                 use_swiglu=True, use_rmsnorm=True, wo_shift=False, block_norm=True,
                 **block_kwargs):
        super().__init__()
        self.block_norm = block_norm
        NormCls = RMSNorm if use_rmsnorm else lambda d: nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.norm1 = NormCls(hidden_size)
        self.norm2 = NormCls(hidden_size)
        if self.block_norm:
            self.norm3 = NormCls(hidden_size)

        self.attn = NormAttention(
            hidden_size, num_heads=num_heads, qkv_bias=True,
            qk_norm=use_qknorm, use_rmsnorm=use_rmsnorm, **block_kwargs,
        )

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate='tanh')
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim,
                           act_layer=approx_gelu, drop=0)

        n_modulation = 4 if wo_shift else 6
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, n_modulation * hidden_size, bias=True)
        )
        self.wo_shift = wo_shift

    def forward(self, x, c, feat_rope=None, alibi_bias=None):
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
            shift_msa = shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(c).chunk(6, dim=1)
            )

        attn_in = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.attn(attn_in, rope=feat_rope, alibi_bias=alibi_bias)
        x = x + gate_msa.unsqueeze(1) * attn_out
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        if self.block_norm:
            x = self.norm3(x)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=False):
        super().__init__()
        NormCls = RMSNorm if use_rmsnorm else lambda d: nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.norm_final = NormCls(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class Encoder(nn.Module):
    """UNITE encoder adapted for EO-UNITE.

    When `use_alibi=True`, accepts `alibi_bias` in forward() and passes it
    to each transformer block. RoPE is disabled automatically.

    When `precomputed_cond` is passed to forward(), it replaces the
    LabelEmbedder output, enabling wavelength+geo+time conditioning.
    """

    def __init__(self, input_size=16, patch_size=1, in_channels=768, hidden_size=1152,
                 depth=28, num_heads=16, mlp_ratio=4.0, class_dropout_prob=0.1,
                 num_classes=1000, learn_sigma=False, use_qknorm=False, use_swiglu=True,
                 use_rope=True, use_alibi=False, use_rmsnorm=True, wo_shift=False,
                 use_gembed=True, in_context_start=None, in_context_len=32,
                 max_tokens=512, block_norm=True):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels if not learn_sigma else in_channels * 2
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_rope = use_rope and not use_alibi  # ALiBi disables RoPE
        self.use_alibi = use_alibi
        self.use_rmsnorm = use_rmsnorm
        self.depth = depth
        self.hidden_size = hidden_size
        self.in_context_start = in_context_start
        self.in_context_len = in_context_len
        self.max_tokens = max_tokens

        self.up_sample = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.t_embedder = GaussianFourierEmbedding(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        self.pos_embed = nn.Parameter(torch.zeros(1, 256, hidden_size), requires_grad=False)

        if self.use_rope:
            half_head_dim = hidden_size // num_heads // 2
            self.feat_rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim * 2, pt_seq_len=self.max_tokens,
            )
            if self.in_context_start is not None:
                self.feat_rope_incontext = VisionRotaryEmbeddingFast(
                    dim=half_head_dim * 2, pt_seq_len=self.max_tokens,
                    num_cls_token=self.in_context_len,
                )
                self.in_context_posemb = nn.Parameter(
                    torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True
                )
                torch.nn.init.normal_(self.in_context_posemb, std=0.02)
            else:
                self.in_context_start = torch.inf
        else:
            self.feat_rope = None
            self.in_context_start = torch.inf

        self.blocks = nn.ModuleList([
            Block(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_qknorm=use_qknorm,
                  use_swiglu=use_swiglu, use_rmsnorm=use_rmsnorm, wo_shift=wo_shift,
                  block_norm=block_norm)
            for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels,
                                      use_rmsnorm=use_rmsnorm)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(256 ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def forward(self, x, t=None, y=None, pos_embed=None, checkpoint_blocks=False,
                img_patch_embed=None, force_drop_ids_y_embedder=None,
                precomputed_cond=None, alibi_bias=None):
        """Forward pass.

        Args:
            x: Latent noise tokens [B, num_latent_tokens, in_channels].
            t: Timestep [B].
            y: Class labels [B] (ignored when precomputed_cond is provided).
            pos_embed: Positional embeddings for latent tokens [B, L, hidden_size].
            checkpoint_blocks: Use gradient checkpointing for memory efficiency.
            img_patch_embed: Image patch embeddings [B, num_patches, hidden_size].
                Concatenated after latent tokens for tokenization path.
            force_drop_ids_y_embedder: CFG dropout mask [B].
            precomputed_cond: Pre-computed conditioning vector [B, hidden_size].
                Replaces y_embedder output. Used in EO-UNITE for modality conditioning.
            alibi_bias: 2D ALiBi bias [1, num_heads, N, N]. Added to attention
                logits in each block. Required when use_alibi=True.

        Returns:
            Output tokens [B, num_latent_tokens, out_channels].
        """
        x = self.up_sample(x)
        x = x + pos_embed

        if img_patch_embed is not None:
            x = torch.cat([x, img_patch_embed], dim=1)

        t_emb = self.t_embedder(t)
        c = t_emb

        if precomputed_cond is not None:
            # EO-UNITE: use modality conditioning instead of class labels
            c = c + precomputed_cond
        else:
            y_emb = self.y_embedder(y, self.training, force_drop_ids=force_drop_ids_y_embedder)
            c = c + y_emb

        for block_idx, block in enumerate(self.blocks):
            if self.in_context_len > 0 and block_idx == self.in_context_start:
                # in-context tokens not used in EO-UNITE (in_context_start=None)
                in_context_tokens = c.unsqueeze(1).repeat(1, self.in_context_len, 1)
                in_context_tokens = in_context_tokens + self.in_context_posemb
                x = torch.cat([in_context_tokens, x], dim=1)

            if self.use_rope:
                rope = self.feat_rope if block_idx < self.in_context_start else self.feat_rope_incontext
            else:
                rope = None

            if checkpoint_blocks and self.training:
                def _run_block(x_in, c_in, _block=block, _rope=rope, _alibi=alibi_bias):
                    return _block(x_in, c_in, feat_rope=_rope, alibi_bias=_alibi)
                x = checkpoint(_run_block, x, c, use_reentrant=False)
            else:
                x = block(x, c, feat_rope=rope, alibi_bias=alibi_bias)

        if self.in_context_start != torch.inf:
            x = x[:, self.in_context_len:]

        x = x[:, :256]  # NOTE: assumes num_latent_tokens == 256
        x = self.final_layer(x, c)
        return x

    def forward_with_cfg(self, x, t, y=None, pos_embed=None, cfg_scale=4.0,
                         bn_func=None, cfg_interval=(0.0, 1.0), cfg_norm_order='norm_first',
                         precomputed_cond=None, alibi_bias=None):
        """Forward with classifier-free guidance (doubled batch)."""
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(
            combined, t, y, pos_embed=pos_embed,
            precomputed_cond=precomputed_cond, alibi_bias=alibi_bias,
        )

        eps = model_out[:, :, :self.in_channels]
        rest = model_out[:, :, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

        if cfg_norm_order in ('norm_first', 'both') and bn_func is not None:
            cond_eps = bn_func(cond_eps)
            uncond_eps = bn_func(uncond_eps)

        low, high = cfg_interval
        interval_mask = (t < high) & ((low == 0.0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, cfg_scale, 1.0)
        cfg_scale_half = cfg_scale_interval[:len(cond_eps)]
        half_eps = uncond_eps + cfg_scale_half.unsqueeze(-1).unsqueeze(-1) * (cond_eps - uncond_eps)

        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=2)
