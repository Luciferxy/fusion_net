"""
Fusion-UNet: Dual-encoder U-Net for IR-VIS image fusion.

Architecture overview:
    ┌─────────┐   ┌─────────┐
    │ IR Enc  │   │ VIS Enc │   ← Separate encoders (no weight sharing)
    └────┬────┘   └────┬────┘
         │             │
    ┌────▼─────────────▼────┐
    │     Fusion Blocks     │   ← Attention-based fusion at each level
    └───────────┬───────────┘
                │
    ┌───────────▼───────────┐
    │       Decoder         │   ← Skip connections from fused features
    └───────────┬───────────┘
                │
    ┌───────────▼───────────┐
    │   Output (Sigmoid)    │   ← RGB fused image ∈ [0, 1]
    └───────────────────────┘
"""

import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder
from .fusion_blocks import FusionBlock


class FusionUNet(nn.Module):
    """
    Complete Fusion-UNet model.

    Args:
        in_channels:  Number of input channels per modality (default: 3 for RGB)
        out_channels: Number of output channels (default: 3 for RGB fused)
        enc_channels: Channel progression for encoder levels
        use_channel_attention: Enable channel attention in fusion blocks
        use_spatial_attention:  Enable spatial attention in fusion blocks
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        enc_channels: list = None,
        use_channel_attention: bool = True,
        use_spatial_attention: bool = True,
    ):
        super().__init__()
        if enc_channels is None:
            enc_channels = [64, 128, 256, 512, 1024]

        # --- Dual Encoders (separate weights) ---
        self.ir_encoder = Encoder(in_channels, enc_channels)
        self.vis_encoder = Encoder(in_channels, enc_channels)

        # --- Fusion blocks (one per encoder level, including bottleneck) ---
        self.fusion_blocks = nn.ModuleList([
            FusionBlock(ch, use_channel_attention, use_spatial_attention)
            for ch in enc_channels
        ])

        # --- Decoder ---
        dec_channels = [enc_channels[-1]] + list(reversed(enc_channels[:-1]))
        self.decoder = Decoder(dec_channels)

        # --- Output head ---
        self.output_conv = nn.Sequential(
            nn.Conv2d(enc_channels[0], out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, ir: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ir:  Infrared image  (B, C, H, W)
            vis: Visible image   (B, C, H, W)

        Returns:
            Fused image (B, out_channels, H, W) with values in [0, 1]
        """
        # ---- Encode ----
        ir_features = self.ir_encoder(ir)     # [level1, level2, ..., bottleneck]
        vis_features = self.vis_encoder(vis)

        # ---- Fuse at each level ----
        fused_features = []
        for ir_feat, vis_feat, fusion in zip(ir_features, vis_features, self.fusion_blocks):
            fused_features.append(fusion(ir_feat, vis_feat))

        # ---- Decode ----
        # Bottleneck is the last fused feature
        bottleneck = fused_features[-1]
        # Skip connections: levels 4→1 (deepest to shallowest, excluding bottleneck)
        skips = list(reversed(fused_features[:-1]))

        decoded = self.decoder(bottleneck, skips)

        # ---- Output ----
        output = self.output_conv(decoded)
        return output


def count_parameters(model: nn.Module) -> int:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    return trainable


if __name__ == "__main__":
    # Quick sanity check
    model = FusionUNet(in_channels=3, out_channels=3)
    count_parameters(model)

    ir = torch.randn(2, 3, 256, 256)
    vis = torch.randn(2, 3, 256, 256)
    out = model(ir, vis)
    print(f"Input shape:  {ir.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
