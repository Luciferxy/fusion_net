"""
Inference script for Fusion-UNet.

Fuse a single IR-VIS image pair or a batch of pairs.

Usage:
    # Single pair
    python src/inference.py --ir path/to/ir.png --vis path/to/vis.png --output fused.png

    # Batch mode
    python src/inference.py --ir_dir data/test/ir --vis_dir data/test/vis --output_dir outputs/
"""

import os
import sys
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.fusion_unet import FusionUNet
from src.utils.visualization import tensor_to_numpy


def preprocess(image_path: str, size: int = 256) -> torch.Tensor:
    """Load and preprocess a single image."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Failed to load image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)  # (1, 3, H, W)
    return tensor


def postprocess(tensor: torch.Tensor, original_size: tuple = None) -> np.ndarray:
    """Convert output tensor to uint8 numpy array."""
    img = tensor_to_numpy(tensor)
    img = (img * 255).astype(np.uint8)

    if original_size is not None:
        img = cv2.resize(img, original_size, interpolation=cv2.INTER_LINEAR)

    return img


def fuse_single(
    model: torch.nn.Module,
    ir_path: str,
    vis_path: str,
    output_path: str,
    device: torch.device,
    image_size: int = 256,
    preserve_size: bool = True,
) -> None:
    """Fuse a single IR-VIS pair and save the result."""
    # Get original size if preserving
    original_size = None
    if preserve_size:
        orig_img = cv2.imread(vis_path)
        if orig_img is not None:
            original_size = (orig_img.shape[1], orig_img.shape[0])  # (W, H)

    ir_tensor = preprocess(ir_path, image_size).to(device)
    vis_tensor = preprocess(vis_path, image_size).to(device)

    with torch.no_grad():
        fused_tensor = model(ir_tensor, vis_tensor)

    fused_img = postprocess(fused_tensor, original_size)

    # RGB → BGR for cv2
    fused_bgr = cv2.cvtColor(fused_img, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    cv2.imwrite(output_path, fused_bgr)
    print(f"✅ Fused image saved: {output_path}")


def fuse_batch(
    model: torch.nn.Module,
    ir_dir: str,
    vis_dir: str,
    output_dir: str,
    device: torch.device,
    image_size: int = 256,
) -> None:
    """Fuse all matching IR-VIS pairs in directories."""
    os.makedirs(output_dir, exist_ok=True)

    ir_files = sorted(Path(ir_dir).glob("*"))
    vis_files = sorted(Path(vis_dir).glob("*"))

    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    ir_files = [f for f in ir_files if f.suffix.lower() in valid_ext]
    vis_files = [f for f in vis_files if f.suffix.lower() in valid_ext]

    n = min(len(ir_files), len(vis_files))
    print(f"📂 Found {n} image pairs")

    total_time = 0
    for i in tqdm(range(n), desc="Fusing"):
        t0 = time.time()

        output_path = os.path.join(output_dir, f"fused_{ir_files[i].stem}.png")
        fuse_single(
            model, str(ir_files[i]), str(vis_files[i]),
            output_path, device, image_size, preserve_size=False,
        )

        total_time += time.time() - t0

    avg_time = total_time / max(n, 1)
    print(f"\n⏱️  Average inference time: {avg_time * 1000:.1f} ms/image")
    print(f"✅ All {n} fused images saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Fusion-UNet Inference")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)

    # Single pair mode
    parser.add_argument("--ir", type=str, default=None, help="Path to IR image")
    parser.add_argument("--vis", type=str, default=None, help="Path to VIS image")
    parser.add_argument("--output", type=str, default="fused_output.png")

    # Batch mode
    parser.add_argument("--ir_dir", type=str, default=None)
    parser.add_argument("--vis_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/inference")

    args = parser.parse_args()

    # --- Device ---
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # --- Model ---
    print("🏗️  Loading model...")
    model = FusionUNet(in_channels=3, out_channels=3).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("✅ Model loaded")

    # --- Run ---
    if args.ir and args.vis:
        # Single pair mode
        fuse_single(model, args.ir, args.vis, args.output, device, args.image_size)
    elif args.ir_dir and args.vis_dir:
        # Batch mode
        fuse_batch(model, args.ir_dir, args.vis_dir, args.output_dir, device, args.image_size)
    else:
        print("❌ Error: Provide either --ir/--vis for single pair, or --ir_dir/--vis_dir for batch mode.")
        sys.exit(1)


if __name__ == "__main__":
    main()
