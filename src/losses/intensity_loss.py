"""
Intensity and gradient losses for image fusion.

Provides:
- IntensityLoss: Weighted L1 loss between fused and both source images
- GradientLoss:  Sobel-based edge preservation loss
- FusionLoss:    Combined loss (intensity + SSIM + gradient)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssim_loss import SSIMLoss


class IntensityLoss(nn.Module):
    """
    Weighted L1 intensity loss.

    L = λ_ir * ‖fused - ir‖₁ + λ_vis * ‖fused - vis‖₁
    """

    def __init__(self, weight_ir: float = 0.5, weight_vis: float = 0.5):
        super().__init__()
        self.weight_ir = weight_ir
        self.weight_vis = weight_vis
        self.l1 = nn.L1Loss()

    def forward(
        self, fused: torch.Tensor, ir: torch.Tensor, vis: torch.Tensor
    ) -> torch.Tensor:
        loss_ir = self.l1(fused, ir)
        loss_vis = self.l1(fused, vis)
        return self.weight_ir * loss_ir + self.weight_vis * loss_vis


class GradientLoss(nn.Module):
    """
    Gradient preservation loss using Sobel filters.

    Encourages the fused image to preserve edge information by matching
    gradients with the stronger gradient of either source image.
    """

    def __init__(self):
        super().__init__()
        # Sobel kernels
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _gradient_magnitude(self, img: torch.Tensor) -> torch.Tensor:
        """Compute gradient magnitude for each channel."""
        B, C, H, W = img.shape
        # Process each channel separately
        grad_mag = []
        for c in range(C):
            channel = img[:, c : c + 1, :, :]
            gx = F.conv2d(channel, self.sobel_x, padding=1)
            gy = F.conv2d(channel, self.sobel_y, padding=1)
            mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
            grad_mag.append(mag)
        return torch.cat(grad_mag, dim=1)

    def forward(
        self, fused: torch.Tensor, ir: torch.Tensor, vis: torch.Tensor
    ) -> torch.Tensor:
        grad_fused = self._gradient_magnitude(fused)
        grad_ir = self._gradient_magnitude(ir)
        grad_vis = self._gradient_magnitude(vis)

        # Target: maximum gradient from either source (preserve strongest edges)
        grad_target = torch.max(grad_ir, grad_vis)

        return F.l1_loss(grad_fused, grad_target)


class FusionLoss(nn.Module):
    """
    Combined loss for IR-VIS image fusion.

    L_total = L_intensity + α * L_ssim + β * L_gradient

    Args:
        weight_ir:       λ₁ weight for IR intensity loss
        weight_vis:      λ₂ weight for VIS intensity loss
        ssim_weight:     α — relative weight for SSIM loss
        gradient_weight: β — relative weight for gradient loss
        window_size:     SSIM window size
    """

    def __init__(
        self,
        weight_ir: float = 0.5,
        weight_vis: float = 0.5,
        ssim_weight: float = 1.0,
        gradient_weight: float = 0.5,
        window_size: int = 11,
    ):
        super().__init__()
        self.intensity_loss = IntensityLoss(weight_ir, weight_vis)
        self.ssim_loss = SSIMLoss(window_size)
        self.gradient_loss = GradientLoss()
        self.ssim_weight = ssim_weight
        self.gradient_weight = gradient_weight

    def forward(
        self, fused: torch.Tensor, ir: torch.Tensor, vis: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            fused: Model output (B, C, H, W)
            ir:    IR input     (B, C, H, W)
            vis:   VIS input    (B, C, H, W)

        Returns:
            Combined scalar loss
        """
        l_intensity = self.intensity_loss(fused, ir, vis)

        # SSIM loss averaged over both sources
        l_ssim_ir = self.ssim_loss(fused, ir)
        l_ssim_vis = self.ssim_loss(fused, vis)
        l_ssim = 0.5 * (l_ssim_ir + l_ssim_vis)

        l_gradient = self.gradient_loss(fused, ir, vis)

        total = l_intensity + self.ssim_weight * l_ssim + self.gradient_weight * l_gradient
        return total
