# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

# ADAPTED from above for DOFA

from functools import partial
from torchgeo.models.dofa import DOFABase16_Weights, DOFA

import torch
from typing import Any
import torch.nn as nn
from torch import Tensor


# overwrite torchgeo model to return the extracted features
class DOFAFeatureExtractor(DOFA):
    # https://github.com/xiong-zhitong/DOFA-pytorch/blob/3f43306e0adfcd01d0e3399e349698bb66ce49c3/src/foundation_models/DOFA/models_dwv_seg.py#L79C5-L115C17
    def forward_features(self, x: Tensor, wavelengths: Tensor) -> list[Tensor]:
        """Forward pass of the feature embedding layer.

        Args:
            x: Input mini-batch.
            wavelengths: Wavelengths of each spectral band (μm).

        Returns:
            Output feature list
        """
        # embed patches
        # wavelist = torch.tensor(wavelengths, device=x.device).float()
        self.waves = wavelengths
        # TODO #1 how to convert coordinates to higher dimension
        x, _ = self.patch_embed(x, self.waves)

        hw = self.img_size // self.patch_embed.kernel_size
        hw_shape = (hw, hw)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        out_features: list[Tensor] = []

        # apply Transformer blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            # if i in self.out_indices:
            out = x[:, 1:]
            B, _, C = out.shape
            out = (
                out.reshape(B, hw_shape[0], hw_shape[1], C)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            out_features.append(out)

        return out_features

    def forward(self, x: Tensor, wavelengths: Tensor) -> list[Tensor]:
        """Forward pass of the model.

        Args:
            x: Input mini-batch.
            wavelengths: Wavelengths of each spectral band (μm).

        Returns:
            Output list of tensors
        """
        x = self.forward_features(x, wavelengths)
        return x


# https://github.com/microsoft/torchgeo/blob/e14073846d71fc17bc69617794883b485fb07728/torchgeo/models/dofa.py#L457C1-L492C17
def dofa_base_patch16_224(pretrained: bool = False, *args: Any, **kwargs: Any) -> DOFA:
    """Dynamic One-For-All (DOFA) base patch size 16 model.

    Args:
        pretrained: Whether to load weights from a pre-trained model.
        *args: Additional arguments to pass to :class:`DOFA`.
        **kwargs: Additional keyword arguments to pass to :class:`DOFA`.

    Returns:
        A DOFA base 16 model.
    """
    kwargs |= {'patch_size': 16, 'embed_dim': 768, 'depth': 12, 'num_heads': 12}
    model = DOFAFeatureExtractor(*args, **kwargs)

    if pretrained:
        weights = DOFABase16_Weights.DOFA_MAE
        missing_keys, unexpected_keys = model.load_state_dict(
            weights.get_state_dict(progress=True), strict=False
        )
        # Both fc_norm and head are generated dynamically
        assert set(missing_keys) <= {
            'fc_norm.weight',
            'fc_norm.bias',
            'head.weight',
            'head.bias',
        }
        assert not unexpected_keys

    return model
