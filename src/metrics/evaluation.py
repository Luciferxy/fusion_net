"""
Comprehensive evaluation metrics for image fusion.

Provides PSNR, Mutual Information, Average Gradient, and a combined
evaluation function that runs all metrics at once.
"""

import numpy as np
from typing import Dict

from .ssim import calculate_ssim
from .entropy import calculate_entropy


def calculate_psnr(img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio.

    PSNR = 10 * log10(data_range² / MSE)

    Args:
        img1, img2: Images as numpy arrays
        data_range: Maximum possible pixel value

    Returns:
        PSNR in dB (higher = better quality)
    """
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))


def calculate_mi(img1: np.ndarray, img2: np.ndarray, bins: int = 256) -> float:
    """
    Mutual Information between two images.

    MI(A, B) = H(A) + H(B) - H(A, B)

    Higher MI indicates more shared information between fused and source.

    Args:
        img1, img2: Images as numpy arrays (converted to grayscale internally)
        bins:       Histogram bins

    Returns:
        Mutual information value
    """
    # Convert to grayscale uint8
    def _to_gray(img):
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        if img.ndim == 3:
            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        return img.flatten()

    a = _to_gray(img1)
    b = _to_gray(img2)

    # Joint histogram
    hist_ab, _, _ = np.histogram2d(a, b, bins=bins, range=[[0, 256], [0, 256]])
    hist_a = hist_ab.sum(axis=1)
    hist_b = hist_ab.sum(axis=0)

    # Normalize
    p_ab = hist_ab / hist_ab.sum()
    p_a = hist_a / hist_a.sum()
    p_b = hist_b / hist_b.sum()

    # Compute MI
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if p_ab[i, j] > 0 and p_a[i] > 0 and p_b[j] > 0:
                mi += p_ab[i, j] * np.log2(p_ab[i, j] / (p_a[i] * p_b[j]))

    return float(mi)


def calculate_ag(image: np.ndarray) -> float:
    """
    Average Gradient — measures edge/detail preservation.

    AG = mean(sqrt(gx² + gy²)) where gx, gy are spatial gradients.

    Args:
        image: Image as numpy array (H, W) or (H, W, C)

    Returns:
        Average gradient value (higher = sharper/more detail)
    """
    if image.max() <= 1.0:
        image = (image * 255).astype(np.float64)
    else:
        image = image.astype(np.float64)

    if image.ndim == 3:
        image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

    gx = np.diff(image, axis=1)
    gy = np.diff(image, axis=0)

    # Align dimensions
    gx = gx[:-1, :]
    gy = gy[:, :-1]

    gradient = np.sqrt(gx ** 2 + gy ** 2)
    return float(gradient.mean())


def calculate_sd(image: np.ndarray) -> float:
    """
    Standard Deviation — measures contrast/dynamic range.

    Args:
        image: Image as numpy array

    Returns:
        Standard deviation value
    """
    if image.ndim == 3:
        image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    return float(np.std(image))


def evaluate_fusion(
    fused: np.ndarray,
    ir: np.ndarray,
    vis: np.ndarray,
    data_range: float = 1.0,
) -> Dict[str, float]:
    """
    Run all evaluation metrics on a fusion result.

    Args:
        fused: Fused image     (H, W, C) or (H, W)
        ir:    IR source image
        vis:   VIS source image
        data_range: Image value range (1.0 for normalized, 255.0 for uint8)

    Returns:
        Dictionary with all metric values
    """
    results = {}

    # SSIM with both sources
    results["ssim_ir"] = calculate_ssim(fused, ir, data_range=data_range)
    results["ssim_vis"] = calculate_ssim(fused, vis, data_range=data_range)
    results["ssim_avg"] = (results["ssim_ir"] + results["ssim_vis"]) / 2.0

    # PSNR with both sources
    results["psnr_ir"] = calculate_psnr(fused, ir, data_range=data_range)
    results["psnr_vis"] = calculate_psnr(fused, vis, data_range=data_range)

    # Information metrics
    results["entropy"] = calculate_entropy(fused)
    results["mi_ir"] = calculate_mi(fused, ir)
    results["mi_vis"] = calculate_mi(fused, vis)

    # Detail metrics
    results["ag"] = calculate_ag(fused)
    results["sd"] = calculate_sd(fused)

    return results


def print_metrics(metrics: Dict[str, float]) -> None:
    """Pretty-print evaluation metrics."""
    print("\n" + "=" * 50)
    print("  FUSION QUALITY METRICS")
    print("=" * 50)
    print(f"  SSIM (IR):      {metrics.get('ssim_ir', 0):.4f}")
    print(f"  SSIM (VIS):     {metrics.get('ssim_vis', 0):.4f}")
    print(f"  SSIM (Avg):     {metrics.get('ssim_avg', 0):.4f}")
    print(f"  PSNR (IR):      {metrics.get('psnr_ir', 0):.2f} dB")
    print(f"  PSNR (VIS):     {metrics.get('psnr_vis', 0):.2f} dB")
    print(f"  Entropy:        {metrics.get('entropy', 0):.4f}")
    print(f"  MI (IR):        {metrics.get('mi_ir', 0):.4f}")
    print(f"  MI (VIS):       {metrics.get('mi_vis', 0):.4f}")
    print(f"  Avg Gradient:   {metrics.get('ag', 0):.4f}")
    print(f"  Std Deviation:  {metrics.get('sd', 0):.4f}")
    print("=" * 50)
