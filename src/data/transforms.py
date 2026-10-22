"""
Data augmentation transforms for IR-VIS image fusion training.

Provides synchronized transforms that apply identical spatial
augmentations to both IR and VIS images in a pair.
"""

import random
from typing import Tuple

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


class PairedTransform:
    """
    Apply identical random augmentations to IR and VIS image pairs.

    Ensures spatial transforms (flip, rotate, crop) are synchronized
    across both images in the pair.
    """

    def __init__(self, image_size: int = 256, augment: bool = True):
        self.image_size = image_size
        self.augment = augment

    def __call__(
        self, ir: Image.Image, vis: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Resize
        ir = TF.resize(ir, (self.image_size, self.image_size))
        vis = TF.resize(vis, (self.image_size, self.image_size))

        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                ir = TF.hflip(ir)
                vis = TF.hflip(vis)

            # Random vertical flip
            if random.random() > 0.5:
                ir = TF.vflip(ir)
                vis = TF.vflip(vis)

            # Random rotation (0, 90, 180, 270)
            angle = random.choice([0, 90, 180, 270])
            if angle > 0:
                ir = TF.rotate(ir, angle)
                vis = TF.rotate(vis, angle)

        # To tensor [0, 1]
        ir = TF.to_tensor(ir)
        vis = TF.to_tensor(vis)

        return ir, vis
