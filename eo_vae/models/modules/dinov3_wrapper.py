import math

import torch
import torch.nn as nn


class DINOv3Wrapper(nn.Module):
    """Thin wrapper around a locally-loaded DINOv3 ViT model for feature extraction.

    Args:
        repo_dir: Path to the local DINOv3 repo clone.
        model_name: Hub model name, e.g. 'dinov2_vitl14' or 'dinov2_vitb14'.
        ckpt_path: Path to the local checkpoint file.
    """

    def __init__(self, repo_dir: str, model_name: str, ckpt_path: str):
        super().__init__()
        self.model = torch.hub.load(repo_dir, model_name, source='local', weights=ckpt_path)
        for p in self.model.parameters():
            p.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial patch token features.

        Args:
            x: Input tensor [B, 3, H, W], z-score normalized.

        Returns:
            Spatial feature map [B, D, hw, hw] where hw = input_size / patch_size.
        """
        out = self.model.forward_features(x)
        patch_tokens = out['x_norm_patchtokens']  # [B, N, D]
        B, N, D = patch_tokens.shape
        hw = int(math.sqrt(N))
        return patch_tokens.permute(0, 2, 1).view(B, D, hw, hw)
