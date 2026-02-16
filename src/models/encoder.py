"""
Encoder module for Fusion-UNet.

Implements the convolutional building block and the multi-level encoder
used by both the IR and VIS branches.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Double convolution block: (Conv3x3 → BN → ReLU) × 2."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """Single encoder level: ConvBlock followed by MaxPool downsampling."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        """
        Returns:
            features: output of ConvBlock (used as skip connection)
            pooled:   downsampled features (fed to next level)
        """
        features = self.conv(x)
        pooled = self.pool(features)
        return features, pooled


class Encoder(nn.Module):
    """
    Multi-level encoder.

    Produces hierarchical features at 5 levels with channel progression:
        in_channels → 64 → 128 → 256 → 512 → 1024

    The bottleneck (level 5) does NOT apply pooling.
    """

    def __init__(self, in_channels: int = 3, channels: list = None):
        super().__init__()
        if channels is None:
            channels = [64, 128, 256, 512, 1024]

        self.levels = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels[:-1]:
            self.levels.append(EncoderBlock(prev_ch, ch))
            prev_ch = ch

        # Bottleneck — no pooling
        self.bottleneck = ConvBlock(prev_ch, channels[-1])

    def forward(self, x: torch.Tensor):
        """
        Returns:
            features: list of feature maps at each level (including bottleneck)
                      [level1_64, level2_128, level3_256, level4_512, bottleneck_1024]
        """
        features = []
        for level in self.levels:
            feat, x = level(x)
            features.append(feat)

        # Bottleneck
        bottleneck = self.bottleneck(x)
        features.append(bottleneck)

        return features
