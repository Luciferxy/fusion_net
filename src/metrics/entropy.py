"""
Shannon Entropy metric for image fusion evaluation.

Higher entropy indicates richer information content in the fused image.
"""

import numpy as np


def calculate_entropy(image: np.ndarray) -> float:
    """
    Calculate Shannon entropy of an image.

    Entropy = -Σ(p_i * log2(p_i))

    Args:
        image: Image as numpy array (H, W) or (H, W, C) in [0, 1] or [0, 255]

    Returns:
        Entropy value (higher = more information content)
    """
    # Convert to uint8 if needed
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    # Convert to grayscale if multichannel
    if image.ndim == 3:
        # Weighted average (standard luminance)
        image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

    # Calculate histogram
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))

    # Normalize to probabilities
    prob = hist / hist.sum()

    # Remove zero entries (log(0) undefined)
    prob = prob[prob > 0]

    # Shannon entropy
    entropy = -np.sum(prob * np.log2(prob))

    return float(entropy)
