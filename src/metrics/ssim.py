"""
SSIM metric for evaluation (non-differentiable, numpy-based).

Used for quantitative evaluation of fusion results.
"""

import numpy as np
from skimage.metrics import structural_similarity


def calculate_ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    data_range: float = 1.0,
    multichannel: bool = True,
) -> float:
    """
    Calculate Structural Similarity Index between two images.

    Args:
        img1, img2:    Images as numpy arrays (H, W) or (H, W, C) in [0, data_range]
        data_range:    Dynamic range of input images
        multichannel:  True if images have multiple channels

    Returns:
        SSIM score ∈ [0, 1] (higher = more similar)
    """
    if img1.ndim == 3 and img1.shape[2] == 3:
        return structural_similarity(
            img1, img2,
            data_range=data_range,
            channel_axis=2,
            win_size=11,
        )
    else:
        return structural_similarity(
            img1, img2,
            data_range=data_range,
            win_size=11,
        )
