"""
Fusion blocks for combining IR and VIS encoder features.

Implements channel attention, spatial attention, and the main fusion module
that merges features from both modalities at each encoder level.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style channel attention.

    Learns per-channel importance weights via global average pooling →
    FC → ReLU → FC → Sigmoid.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.attention(x).unsqueeze(-1).unsqueeze(-1)
        return x * weights


class SpatialAttention(nn.Module):
    """
    Spatial attention module.

    Concatenates channel-wise average and max pooled features, then applies
    a 7×7 convolution followed by Sigmoid to produce a spatial weight map.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        spatial_map = torch.cat([avg_pool, max_pool], dim=1)
        weights = self.conv(spatial_map)
        return x * weights


class FusionBlock(nn.Module):
    """
    Feature fusion module for merging IR and VIS features at a single level.

    Pipeline:
        1. Concatenate IR and VIS features (2C channels)
        2. 1×1 Conv to reduce back to C channels
        3. Channel attention (optional)
        4. Spatial attention (optional)
    """

    def __init__(
        self,
        channels: int,
        use_channel_attention: bool = True,
        use_spatial_attention: bool = True,
        reduction: int = 16,
    ):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.channel_attn = (
            ChannelAttention(channels, reduction) if use_channel_attention else nn.Identity()
        )
        self.spatial_attn = (
            SpatialAttention() if use_spatial_attention else nn.Identity()
        )

    def forward(self, ir_feat: torch.Tensor, vis_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ir_feat:  IR encoder features  (B, C, H, W)
            vis_feat: VIS encoder features (B, C, H, W)

        Returns:
            Fused features (B, C, H, W)
        """
        concat = torch.cat([ir_feat, vis_feat], dim=1)  # (B, 2C, H, W)
        fused = self.reduce(concat)                       # (B, C, H, W)
        fused = self.channel_attn(fused)
        fused = self.spatial_attn(fused)
        return fused
