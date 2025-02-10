"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""

import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple

from taming.util import get_ckpt_path


class LPIPS(nn.Module):
    """Learned Perceptual Image Patch Similarity metric with variable network architecture."""

    def __init__(self, net: nn.Module, scaling_layer: nn.Module | None = None, use_dropout: bool = True) -> None:
        """Initialize LPIPS model.

        Args:
            net: Pretrained network for LPIPS
            scaling_layer: Scaling layer for input images, necessary for some classic networks like VGG
            use_dropout: Whether to use dropout in network
        """
        super().__init__()
        self.scaling_layer = scaling_layer
        self.net = net
        # TODO retrieve chns from net it

        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False


    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate perceptual distance between images.

        Args:
            input: Input images [B, C, H, W]
            target: Target images [B, C, H, W]

        Returns:
            Perceptual distance value
        """
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))

        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]

        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        """Initialize single 1x1 conv layer.
        
        Args:
            chn_in: Number of input channels
            chn_out: Number of output channels
            use_dropout: Whether to use dropout
        """
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)



def normalize_tensor(x: Tensor,eps: float=1e-10):
    """Normalize tensor by its L2 norm.
    
    Args:
        x: Input tensor
        eps: Epsilon value for numerical stability

    Returns:
        Normalized tensor
    """
    norm_factor = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    return x/(norm_factor+eps)


def spatial_average(x: Tensor, keepdim: bool=True):
    """Calculate spatial average of tensor.
    
    Args:
        x: Input tensor
        keepdim: Whether to keep dimensions

    Returns:
        Spatial average tensor
    """
    return x.mean([2,3],keepdim=keepdim)



def adopt_weight(weight: float, global_step: int, threshold: int = 0, value: float = 0.) -> float:
    """Adopt weight value based on global step.

    Args:
        weight: Original weight value
        global_step: Current training step
        threshold: Step threshold for weight adoption
        value: Value to use before threshold

    Returns:
        Adopted weight value
    """
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Calculate hinge loss for discriminator.

    Args:
        logits_real: Discriminator predictions on real data
        logits_fake: Discriminator predictions on fake data

    Returns:
        Hinge loss value
    """
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Calculate vanilla GAN loss for discriminator.

    Args:
        logits_real: Discriminator predictions on real data
        logits_fake: Discriminator predictions on fake data

    Returns:
        Vanilla GAN loss value
    """
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


# https://github.com/CompVis/taming-transformers/blob/master/taming/modules/discriminator/model.py

# GAN discriminator

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)
