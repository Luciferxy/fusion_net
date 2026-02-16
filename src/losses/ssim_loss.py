"""
SSIM-based loss for image fusion.

Provides a differentiable Structural Similarity Index loss using
a Gaussian window, suitable for training on GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _gaussian_kernel(window_size: int, sigma: float) -> torch.Tensor:
    """Create a 1D Gaussian kernel."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    gauss = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    return gauss / gauss.sum()


def _create_window(window_size: int, channels: int) -> torch.Tensor:
    """Create a 2D Gaussian window for SSIM computation."""
    kernel_1d = _gaussian_kernel(window_size, sigma=1.5)
    kernel_2d = kernel_1d.unsqueeze(1) @ kernel_1d.unsqueeze(0)  # outer product
    window = kernel_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    window = window.expand(channels, 1, window_size, window_size).contiguous()
    return window


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True,
    data_range: float = 1.0,
) -> torch.Tensor:
    """
    Compute SSIM between two images.

    Args:
        img1, img2: Input tensors (B, C, H, W) in range [0, data_range]
        window_size: Size of the Gaussian window
        size_average: If True, return mean SSIM over batch
        data_range: Dynamic range of the images

    Returns:
        SSIM value(s)
    """
    channels = img1.size(1)
    window = _create_window(window_size, channels).to(img1.device).to(img1.dtype)

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 ** 2, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channels) - mu1_mu2

    # Clamp for numerical stability
    sigma1_sq = torch.clamp(sigma1_sq, min=0)
    sigma2_sq = torch.clamp(sigma2_sq, min=0)

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(dim=[1, 2, 3])


class SSIMLoss(nn.Module):
    """
    SSIM Loss = 1 - SSIM.

    Lower is better (minimization objective).
    """

    def __init__(self, window_size: int = 11, data_range: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.data_range = data_range

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - ssim(pred, target, self.window_size, data_range=self.data_range)
