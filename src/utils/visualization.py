"""
Visualization utilities for IR-VIS fusion results.

Provides functions for displaying and saving comparison grids,
training curves, and individual fusion results.
"""

import os
from pathlib import Path
from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a CHW tensor to HWC numpy array in [0, 1]."""
    if isinstance(tensor, torch.Tensor):
        img = tensor.detach().cpu()
        if img.ndim == 4:
            img = img[0]  # Take first in batch
        img = img.permute(1, 2, 0).numpy()
    else:
        img = tensor

    img = np.clip(img, 0, 1)
    return img


def show_fusion_result(
    ir: np.ndarray,
    vis: np.ndarray,
    fused: np.ndarray,
    title: str = "Fusion Result",
    figsize: tuple = (15, 5),
) -> None:
    """Display IR, VIS, and fused images side by side."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(ir)
    axes[0].set_title("Infrared (IR)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(vis)
    axes[1].set_title("Visible (VIS)", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(fused)
    axes[2].set_title("Fused Output", fontsize=12, fontweight="bold", color="green")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def save_comparison_grid(
    ir_images: list,
    vis_images: list,
    fused_images: list,
    save_path: str,
    max_images: int = 8,
    figsize_per_image: tuple = (5, 5),
) -> None:
    """
    Save a grid of IR / VIS / Fused comparisons.

    Args:
        ir_images:    List of IR images (numpy HWC)
        vis_images:   List of VIS images
        fused_images: List of fused images
        save_path:    Output file path
        max_images:   Maximum number of rows
        figsize_per_image: Size per image cell
    """
    n = min(len(ir_images), max_images)
    fig_w = figsize_per_image[0] * 3
    fig_h = figsize_per_image[1] * n

    fig, axes = plt.subplots(n, 3, figsize=(fig_w, fig_h))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        axes[i, 0].imshow(np.clip(ir_images[i], 0, 1))
        axes[i, 0].axis("off")
        if i == 0:
            axes[i, 0].set_title("IR", fontsize=14, fontweight="bold")

        axes[i, 1].imshow(np.clip(vis_images[i], 0, 1))
        axes[i, 1].axis("off")
        if i == 0:
            axes[i, 1].set_title("VIS", fontsize=14, fontweight="bold")

        axes[i, 2].imshow(np.clip(fused_images[i], 0, 1))
        axes[i, 2].axis("off")
        if i == 0:
            axes[i, 2].set_title("Fused", fontsize=14, fontweight="bold", color="green")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"💾 Comparison grid saved to {save_path}")


def save_single_result(
    fused: np.ndarray,
    save_path: str,
) -> None:
    """Save a single fused image."""
    import cv2

    if fused.max() <= 1.0:
        fused = (fused * 255).astype(np.uint8)

    # RGB → BGR for cv2
    if fused.ndim == 3 and fused.shape[2] == 3:
        fused = cv2.cvtColor(fused, cv2.COLOR_RGB2BGR)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, fused)


def plot_training_curves(
    train_losses: list,
    val_metrics: dict,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot training loss and validation metrics curves.

    Args:
        train_losses: List of per-epoch training losses
        val_metrics:  Dict of {metric_name: [values_per_epoch]}
        save_path:    If provided, save plot to this path
    """
    n_metrics = len(val_metrics)
    fig, axes = plt.subplots(1, 1 + n_metrics, figsize=(6 * (1 + n_metrics), 5))

    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # Training loss
    axes[0].plot(train_losses, color="#e74c3c", linewidth=2)
    axes[0].set_title("Training Loss", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    # Validation metrics
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#f39c12", "#1abc9c"]
    for idx, (name, values) in enumerate(val_metrics.items()):
        ax = axes[1 + idx] if 1 + idx < len(axes) else axes[-1]
        color = colors[idx % len(colors)]
        ax.plot(values, color=color, linewidth=2)
        ax.set_title(f"Val {name.upper()}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📈 Training curves saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)
