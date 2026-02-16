"""
Testing / evaluation script for Fusion-UNet.

Loads the best model checkpoint, runs inference on the test set,
computes all evaluation metrics, and saves visual comparison grids.

Usage:
    python src/test.py --checkpoint checkpoints/best_model.pth --data_dir data
    python src/test.py --checkpoint checkpoints/best_model.pth --output_dir outputs/test_results
"""

import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.fusion_unet import FusionUNet
from src.data.dataset import get_dataloader
from src.metrics.evaluation import evaluate_fusion, print_metrics
from src.utils.visualization import tensor_to_numpy, save_comparison_grid, save_single_result


def main():
    parser = argparse.ArgumentParser(description="Test Fusion-UNet")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="outputs/test_results")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_images", action="store_true", default=True)
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

    if "metric_value" in checkpoint:
        print(f"   Best {checkpoint.get('metric_name', 'metric')}: {checkpoint['metric_value']:.4f}")

    # --- Data ---
    print("📂 Loading test data...")
    test_loader = get_dataloader(
        root_dir=args.data_dir,
        split="test",
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=2,
        shuffle=False,
    )

    # --- Evaluate ---
    print("\n🔬 Running evaluation...")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "fused"), exist_ok=True)

    all_metrics = []
    ir_images, vis_images, fused_images = [], [], []

    with torch.no_grad():
        for idx, (ir, vis) in enumerate(tqdm(test_loader, desc="Testing")):
            ir = ir.to(device)
            vis = vis.to(device)

            fused = model(ir, vis)

            # Convert to numpy
            for i in range(fused.size(0)):
                ir_np = tensor_to_numpy(ir[i])
                vis_np = tensor_to_numpy(vis[i])
                fused_np = tensor_to_numpy(fused[i])

                # Compute metrics
                metrics = evaluate_fusion(fused_np, ir_np, vis_np)
                all_metrics.append(metrics)

                # Save individual fused images
                if args.save_images:
                    img_idx = idx * args.batch_size + i
                    save_path = os.path.join(
                        args.output_dir, "fused", f"fused_{img_idx:04d}.png"
                    )
                    save_single_result(fused_np, save_path)

                # Collect for grid
                ir_images.append(ir_np)
                vis_images.append(vis_np)
                fused_images.append(fused_np)

    # --- Aggregate Metrics ---
    print("\n📊 Computing aggregate metrics...")
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = float(np.mean(values))
        avg_metrics[f"{key}_std"] = float(np.std(values))

    print_metrics(avg_metrics)

    # --- Save Results ---
    # Metrics JSON
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(avg_metrics, f, indent=2)
    print(f"\n💾 Metrics saved to {metrics_path}")

    # Per-image metrics
    per_image_path = os.path.join(args.output_dir, "per_image_metrics.json")
    with open(per_image_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Comparison grid
    grid_path = os.path.join(args.output_dir, "comparison_grid.png")
    save_comparison_grid(ir_images, vis_images, fused_images, grid_path, max_images=8)

    print(f"\n✅ Testing complete! Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
