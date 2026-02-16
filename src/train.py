"""
Training script for Fusion-UNet IR-VIS image fusion model.

Usage:
    python src/train.py --config config/train_config.yaml
    python src/train.py --config config/train_config.yaml --resume checkpoints/checkpoint_epoch_50.pth
"""

import os
import sys
import time
import argparse
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.fusion_unet import FusionUNet, count_parameters
from src.data.dataset import get_dataloader
from src.losses.intensity_loss import FusionLoss
from src.metrics.ssim import calculate_ssim
from src.metrics.entropy import calculate_entropy
from src.utils.checkpoint import save_checkpoint, load_checkpoint, save_best_model
from src.utils.visualization import tensor_to_numpy, save_comparison_grid


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate(
    model: nn.Module,
    val_loader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
) -> dict:
    """Run validation and compute metrics."""
    model.eval()
    total_loss = 0.0
    total_ssim = 0.0
    total_entropy = 0.0
    n_batches = 0

    with torch.no_grad():
        for ir, vis in val_loader:
            ir = ir.to(device)
            vis = vis.to(device)

            fused = model(ir, vis)
            loss = criterion(fused, ir, vis)
            total_loss += loss.item()

            # Compute metrics on CPU (numpy)
            for i in range(fused.size(0)):
                fused_np = tensor_to_numpy(fused[i])
                ir_np = tensor_to_numpy(ir[i])
                vis_np = tensor_to_numpy(vis[i])

                ssim_ir = calculate_ssim(fused_np, ir_np)
                ssim_vis = calculate_ssim(fused_np, vis_np)
                total_ssim += (ssim_ir + ssim_vis) / 2.0
                total_entropy += calculate_entropy(fused_np)

            n_batches += 1

    n_images = n_batches * val_loader.batch_size
    metrics = {
        "val_loss": total_loss / max(n_batches, 1),
        "ssim": total_ssim / max(n_images, 1),
        "entropy": total_entropy / max(n_images, 1),
    }

    # Log to tensorboard
    for key, value in metrics.items():
        writer.add_scalar(f"val/{key}", value, epoch)

    return metrics


def train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    config: dict,
    scaler=None,
) -> float:
    """Train for a single epoch."""
    model.train()
    total_loss = 0.0
    log_interval = config.get("logging", {}).get("log_interval", 50)

    pbar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Epoch {epoch:03d}",
        leave=True,
    )

    for batch_idx, (ir, vis) in pbar:
        ir = ir.to(device)
        vis = vis.to(device)

        optimizer.zero_grad()

        # Mixed precision
        if scaler is not None:
            with torch.amp.autocast(device_type='cuda'):
                fused = model(ir, vis)
                loss = criterion(fused, ir, vis)
            scaler.scale(loss).backward()
            # Gradient clipping
            grad_clip = config.get("training", {}).get("gradient_clip", 0)
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            fused = model(ir, vis)
            loss = criterion(fused, ir, vis)
            loss.backward()
            grad_clip = config.get("training", {}).get("gradient_clip", 0)
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        global_step = epoch * len(train_loader) + batch_idx

        # Logging
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if batch_idx % log_interval == 0:
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train Fusion-UNet")
    parser.add_argument("--config", type=str, default="config/train_config.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--device", type=str, default=None, help="Device override (cuda/cpu)")
    args = parser.parse_args()

    # --- Load config ---
    config = load_config(args.config)
    data_cfg = config["data"]
    train_cfg = config["training"]
    loss_cfg = config["loss"]
    log_cfg = config["logging"]
    ckpt_cfg = config["checkpoint"]

    # --- Device ---
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")

    # --- Data Loaders ---
    print("\n📂 Loading datasets...")
    train_loader = get_dataloader(
        root_dir=data_cfg["train_dir"].rsplit("/", 1)[0] if "/" in data_cfg["train_dir"] else "data",
        split="train",
        batch_size=data_cfg["batch_size"],
        image_size=data_cfg["image_size"],
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"],
    )
    val_loader = get_dataloader(
        root_dir=data_cfg["val_dir"].rsplit("/", 1)[0] if "/" in data_cfg["val_dir"] else "data",
        split="val",
        batch_size=data_cfg["batch_size"],
        image_size=data_cfg["image_size"],
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"],
        shuffle=False,
    )

    # --- Model ---
    print("\n🏗️  Building model...")
    model = FusionUNet(in_channels=3, out_channels=3).to(device)
    count_parameters(model)

    # --- Loss ---
    criterion = FusionLoss(
        weight_ir=loss_cfg["intensity_weight_ir"],
        weight_vis=loss_cfg["intensity_weight_vis"],
        ssim_weight=loss_cfg["ssim_weight"],
        gradient_weight=loss_cfg["gradient_weight"],
        window_size=loss_cfg["ssim_window_size"],
    ).to(device)

    # --- Optimizer ---
    opt_cfg = train_cfg["optimizer"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg["weight_decay"],
        betas=tuple(opt_cfg["betas"]),
    )

    # --- Scheduler ---
    sch_cfg = train_cfg["scheduler"]
    if sch_cfg["type"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sch_cfg["T_max"],
            eta_min=sch_cfg["eta_min"],
        )
    elif sch_cfg["type"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=sch_cfg["patience"],
            factor=sch_cfg["factor"],
        )
    else:
        scheduler = None

    # --- Mixed Precision ---
    scaler = None
    if train_cfg.get("mixed_precision", False) and device.type == "cuda":
        scaler = torch.amp.GradScaler()
        print("⚡ Mixed precision training enabled")

    # --- Resume ---
    start_epoch = 0
    best_metric = 0.0
    if args.resume:
        info = load_checkpoint(args.resume, model, optimizer, scheduler, device=str(device))
        start_epoch = info["epoch"] + 1
        best_metric = info.get("best_metric", 0.0) or 0.0

    # --- TensorBoard ---
    writer = SummaryWriter(log_cfg["tensorboard_dir"])

    # --- Training Loop ---
    patience_counter = 0
    early_stop_patience = ckpt_cfg["early_stopping_patience"]

    print(f"\n🚀 Starting training for {train_cfg['epochs']} epochs...")
    print(f"   Batch size: {data_cfg['batch_size']}")
    print(f"   LR: {opt_cfg['lr']}")
    print(f"   Early stopping patience: {early_stop_patience}")
    print()

    for epoch in range(start_epoch, train_cfg["epochs"]):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, writer, config, scaler,
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch, writer)

        # Scheduler step
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["ssim"])
            else:
                scheduler.step()

        # Logging
        elapsed = time.time() - t0
        ssim_val = val_metrics["ssim"]
        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val SSIM: {ssim_val:.4f} | "
            f"Entropy: {val_metrics['entropy']:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        writer.add_scalar("train/epoch_loss", train_loss, epoch)

        # Best model
        if ssim_val > best_metric:
            best_metric = ssim_val
            save_best_model(model, ckpt_cfg["save_dir"], "ssim", ssim_val)
            patience_counter = 0
        else:
            patience_counter += 1

        # Periodic checkpoint
        if (epoch + 1) % ckpt_cfg["save_interval"] == 0:
            ckpt_path = os.path.join(
                ckpt_cfg["save_dir"], f"checkpoint_epoch_{epoch:03d}.pth"
            )
            save_checkpoint(
                model, optimizer, epoch, ckpt_path,
                scheduler=scheduler,
                best_metric=best_metric,
                metrics=val_metrics,
            )

        # Save sample images periodically
        if (epoch + 1) % log_cfg.get("image_interval", 5) == 0:
            model.eval()
            with torch.no_grad():
                sample_ir, sample_vis = next(iter(val_loader))
                sample_ir = sample_ir.to(device)
                sample_vis = sample_vis.to(device)
                sample_fused = model(sample_ir, sample_vis)

                writer.add_images("val/ir", sample_ir, epoch)
                writer.add_images("val/vis", sample_vis, epoch)
                writer.add_images("val/fused", sample_fused, epoch)

        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\n⏹️  Early stopping at epoch {epoch} (patience={early_stop_patience})")
            break

    writer.close()
    print(f"\n✅ Training complete! Best SSIM: {best_metric:.4f}")
    print(f"   Best model: {ckpt_cfg['save_dir']}/best_model.pth")


if __name__ == "__main__":
    main()
