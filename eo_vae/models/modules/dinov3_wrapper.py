import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv3Wrapper(nn.Module):
    """Thin wrapper around a locally-loaded DINOv2 ViT model for feature extraction.

    Resizes input to `input_size` x `input_size` internally before extraction.

    Args:
        repo_dir: Path to the local dinov2 repo clone.
        model_name: Hub model name, e.g. 'dinov2_vitl14' or 'dinov2_vitb14'.
        ckpt_path: Path to the local checkpoint file.
        input_size: Image size fed to the ViT (default 224).
    """

    def __init__(
        self, repo_dir: str, model_name: str, ckpt_path: str, input_size: int = 224
    ):
        super().__init__()
        self.input_size = input_size
        self.model = torch.hub.load(repo_dir, model_name, source='local', weights=ckpt_path)
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial patch-token features from RGB images.

        Args:
            x: Input tensor [B, 3, H, W], any spatial size.

        Returns:
            Spatial feature map [B, D, h_p, w_p] where h_p = input_size / patch_size.
        """
        x = F.interpolate(
            x.float(), size=(self.input_size, self.input_size), mode='bilinear', align_corners=False
        )
        out = self.model.forward_features(x)
        patch_tokens = out['x_norm_patchtokens']  # [B, N, D]
        B, N, D = patch_tokens.shape
        hw = int(math.sqrt(N))
        return patch_tokens.permute(0, 2, 1).view(B, D, hw, hw).contiguous()
