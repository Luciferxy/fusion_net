"""
Model checkpointing utilities.

Handles saving and loading of model weights, optimizer state,
epoch number, and best metric values for training resumption.
"""

import os
from typing import Optional, Dict

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    save_path: str,
    scheduler=None,
    best_metric: Optional[float] = None,
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    """
    Save a training checkpoint.

    Args:
        model:       Model to save
        optimizer:   Optimizer state
        epoch:       Current epoch number
        save_path:   Path to save .pth file
        scheduler:   Optional LR scheduler state
        best_metric: Best validation metric so far
        metrics:     Optional dict of current metrics
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
        "metrics": metrics,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, save_path)
    print(f"💾 Checkpoint saved: {save_path} (epoch {epoch})")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device: str = "cpu",
) -> Dict:
    """
    Load a training checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint file
        model:           Model to load weights into
        optimizer:       Optional optimizer to restore state
        scheduler:       Optional scheduler to restore state
        device:          Device to load tensors onto

    Returns:
        Dictionary with checkpoint metadata (epoch, best_metric, metrics)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"✅ Model weights loaded from {checkpoint_path}")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("✅ Optimizer state restored")

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print("✅ Scheduler state restored")

    info = {
        "epoch": checkpoint.get("epoch", 0),
        "best_metric": checkpoint.get("best_metric", None),
        "metrics": checkpoint.get("metrics", {}),
    }

    print(f"📌 Resuming from epoch {info['epoch']}")
    return info


def save_best_model(
    model: nn.Module,
    save_dir: str,
    metric_name: str = "ssim",
    metric_value: float = 0.0,
) -> None:
    """Save the best model weights (without optimizer state)."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "best_model.pth")

    torch.save({
        "model_state_dict": model.state_dict(),
        "metric_name": metric_name,
        "metric_value": metric_value,
    }, path)

    print(f"🏆 Best model saved ({metric_name}: {metric_value:.4f})")
