"""
Decoder module for Fusion-UNet.

Implements upsampling blocks and the full multi-level decoder with
skip connections from the fused encoder features.
"""

import torch
import torch.nn as nn

from .encoder import ConvBlock


class DecoderBlock(nn.Module):
    """
    Single decoder level: Upsample → Concatenate skip → ConvBlock.

    Uses ConvTranspose2d for learned upsampling.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        # After concatenation with skip: out_channels (up) + out_channels (skip)
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    features from lower level (B, in_channels, H, W)
            skip: skip-connection features  (B, out_channels, 2H, 2W)

        Returns:
            Decoded features (B, out_channels, 2H, 2W)
        """
        x = self.up(x)

        # Handle potential size mismatch from odd spatial dims
        if x.shape != skip.shape:
            x = nn.functional.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=True
            )

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class Decoder(nn.Module):
    """
    Multi-level decoder.

    Takes the bottleneck features and fused skip connections from
    the encoder, progressively upsamples:
        1024 → 512 → 256 → 128 → 64
    """

    def __init__(self, channels: list = None):
        super().__init__()
        if channels is None:
            channels = [1024, 512, 256, 128, 64]

        self.levels = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.levels.append(DecoderBlock(channels[i], channels[i + 1]))

    def forward(self, bottleneck: torch.Tensor, skip_connections: list) -> torch.Tensor:
        """
        Args:
            bottleneck:       Bottleneck features (B, 1024, H, W)
            skip_connections: List of fused skip features from encoder, ordered
                              [level4_512, level3_256, level2_128, level1_64]
                              (deepest to shallowest)

        Returns:
            Decoded features at full resolution (B, 64, H_orig, W_orig)
        """
        x = bottleneck
        for level, skip in zip(self.levels, skip_connections):
            x = level(x, skip)
        return x
