import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F


class SpectralAngleMapperLoss(nn.Module):
    """Computes the spectral angle between reconstructed and target pixel vectors.
    This helps keep the 'color' or spectral curve correct, ignoring brightness differences.
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x_rec, x_true):
        # Permute to [B, H, W, C] for channel-wise angle computation
        # Dot product along channels
        dot = torch.sum(x_rec * x_true, dim=1)

        # Norms for each pixel
        norm_rec = torch.norm(x_rec, dim=1)
        norm_true = torch.norm(x_true, dim=1)

        # Cosine similarity
        cos_sim = dot / (norm_rec * norm_true + self.eps)

        # Clamp to avoid acos issues
        cos_sim = torch.clamp(cos_sim, -1 + self.eps, 1 - self.eps)

        # Spectral Angle in radians
        sam = torch.acos(cos_sim)

        return sam.mean()


class SpatialGradientLoss(nn.Module):
    """Penalizes differences in gradients to preserve edges and reduce blur."""

    def __init__(self):
        super().__init__()
        # Simple Sobel kernels for gradients
        kernel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        kernel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def forward(self, x_rec, x_true):
        b, c, h, w = x_rec.shape

        # Flatten channels for group conv
        x_rec_flat = x_rec.view(-1, 1, h, w)
        x_true_flat = x_true.view(-1, 1, h, w)

        grad_x_rec = F.conv2d(x_rec_flat, self.kernel_x, padding=1)
        grad_y_rec = F.conv2d(x_rec_flat, self.kernel_y, padding=1)

        grad_x_true = F.conv2d(x_true_flat, self.kernel_x, padding=1)
        grad_y_true = F.conv2d(x_true_flat, self.kernel_y, padding=1)

        loss = F.l1_loss(grad_x_rec, grad_x_true) + F.l1_loss(grad_y_rec, grad_y_true)
        return loss


class FocalFrequencyLoss(nn.Module):
    """FFT loss that focuses on frequencies the model struggles with."""

    def __init__(self, loss_weight=1.0, alpha=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha

    def forward(self, x_rec, x_true):
        fft_rec = torch.fft.fft2(x_rec, norm='ortho')
        fft_true = torch.fft.fft2(x_true, norm='ortho')

        rec_stack = torch.stack([fft_rec.real, fft_rec.imag], dim=-1)
        true_stack = torch.stack([fft_true.real, fft_true.imag], dim=-1)

        diff = rec_stack - true_stack

        # Weight based on error magnitude
        tmp = (diff**2).sum(dim=-1, keepdim=True)
        weight = tmp ** (self.alpha / 2)

        loss = (weight * tmp).mean()
        return self.loss_weight * loss


class EOConsistencyLoss(nn.Module):
    def __init__(
        self,
        pixel_weight: float = 1.0,  # Basic L1 loss
        spectral_weight: float = 0.5,  # SAM for spectral accuracy
        spatial_weight: float = 0.5,  # Gradient for edges
        freq_weight: float = 0.1,  # FFT for textures
        feature_weight: float = 0.0,  # Optional DOFA features
        dofa_net: nn.Module = None,
    ):
        super().__init__()
        self.weights = {
            'pixel': pixel_weight,
            'spectral': spectral_weight,
            'spatial': spatial_weight,
            'freq': freq_weight,
            'feature': feature_weight,
        }

        self.sam_loss = SpectralAngleMapperLoss()
        self.grad_loss = SpatialGradientLoss()
        self.fft_loss = FocalFrequencyLoss()

        self.dofa_net = dofa_net
        if self.dofa_net:
            for p in self.dofa_net.parameters():
                p.requires_grad = False

    def forward(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        wvs: torch.Tensor,
        **kwargs,
    ):
        logs = {}
        total_loss = 0.0

        if self.weights['pixel'] > 0:
            l_pix = F.l1_loss(reconstructions, inputs)
            total_loss += self.weights['pixel'] * l_pix
            logs['loss/pixel'] = l_pix.detach()

        if self.weights['spectral'] > 0:
            l_sam = self.sam_loss(reconstructions, inputs)
            total_loss += self.weights['spectral'] * l_sam
            logs['loss/spectral'] = l_sam.detach()

        if self.weights['spatial'] > 0:
            l_spat = self.grad_loss(reconstructions, inputs)
            total_loss += self.weights['spatial'] * l_spat
            logs['loss/spatial'] = l_spat.detach()

        if self.weights['freq'] > 0:
            l_freq = self.fft_loss(reconstructions, inputs)
            total_loss += self.weights['freq'] * l_freq
            logs['loss/freq'] = l_freq.detach()

        if self.weights['feature'] > 0 and self.dofa_net is not None:
            with torch.no_grad():
                f_in = self.dofa_net.extract_features(inputs, wvs)
                f_rec = self.dofa_net.extract_features(reconstructions, wvs)

            l_feat = 0.0
            for fi, fr in zip(f_in, f_rec):
                l_feat += (1.0 - F.cosine_similarity(fi, fr, dim=1)).mean()

            total_loss += self.weights['feature'] * l_feat
            logs['loss/feature'] = l_feat.detach()

        logs['loss/total'] = total_loss.detach()

        # Return loss and logs
        return total_loss, logs
