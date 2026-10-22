"""
Paired IR-VIS dataset loader for image fusion.

Provides:
- FusionDataset: Dataset for loading matched IR/VIS image pairs
- get_dataloader: Helper to create DataLoader instances
"""

import os
from pathlib import Path
from typing import Optional, Callable, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class FusionDataset(Dataset):
    """
    Dataset for paired Infrared (IR) and Visible (VIS) images.

    Expects directory structure:
        root_dir/
            {split}/
                ir/    image_001.png, image_002.png, ...
                vis/   image_001.png, image_002.png, ...

    Files are matched by sorted filename.
    """

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

    def __init__(
        self,
        ir_dir: str,
        vis_dir: str,
        image_size: int = 256,
        transform: Optional[Callable] = None,
    ):
        self.ir_dir = Path(ir_dir)
        self.vis_dir = Path(vis_dir)
        self.image_size = image_size
        self.transform = transform

        self.ir_files = sorted(
            [f for f in self.ir_dir.iterdir() if f.suffix.lower() in self.SUPPORTED_EXTENSIONS]
        )
        self.vis_files = sorted(
            [f for f in self.vis_dir.iterdir() if f.suffix.lower() in self.SUPPORTED_EXTENSIONS]
        )

        # Match by filename stem
        ir_stems = {f.stem: f for f in self.ir_files}
        vis_stems = {f.stem: f for f in self.vis_files}
        common = sorted(set(ir_stems.keys()) & set(vis_stems.keys()))

        if not common:
            raise ValueError(
                f"No matching image pairs found.\n"
                f"  IR dir:  {self.ir_dir} ({len(self.ir_files)} files)\n"
                f"  VIS dir: {self.vis_dir} ({len(self.vis_files)} files)"
            )

        self.ir_files = [ir_stems[s] for s in common]
        self.vis_files = [vis_stems[s] for s in common]

        # Default transform: resize + to tensor [0, 1]
        if self.transform is None:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
            ])

    def __len__(self) -> int:
        return len(self.ir_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ir_img = Image.open(self.ir_files[idx]).convert("RGB")
        vis_img = Image.open(self.vis_files[idx]).convert("RGB")

        ir_tensor = self.transform(ir_img)
        vis_tensor = self.transform(vis_img)

        return ir_tensor, vis_tensor


def get_dataloader(
    root_dir: str,
    split: str = "train",
    batch_size: int = 2,
    image_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = True,
    shuffle: Optional[bool] = None,
    transform: Optional[Callable] = None,
) -> DataLoader:
    """
    Create a DataLoader for IR-VIS image pairs.

    Args:
        root_dir:    Root data directory containing split subdirectories
        split:       One of 'train', 'val', 'test'
        batch_size:  Batch size
        image_size:  Target image size (square)
        num_workers: Number of data loading workers
        pin_memory:  Pin memory for CUDA transfers
        shuffle:     Shuffle data (defaults to True for train, False otherwise)
        transform:   Optional custom transform

    Returns:
        DataLoader yielding (ir_tensor, vis_tensor) batches
    """
    ir_dir = os.path.join(root_dir, split, "ir")
    vis_dir = os.path.join(root_dir, split, "vis")

    if not os.path.isdir(ir_dir) or not os.path.isdir(vis_dir):
        raise FileNotFoundError(
            f"Dataset directories not found:\n"
            f"  IR:  {ir_dir} (exists={os.path.isdir(ir_dir)})\n"
            f"  VIS: {vis_dir} (exists={os.path.isdir(vis_dir)})"
        )

    dataset = FusionDataset(
        ir_dir=ir_dir,
        vis_dir=vis_dir,
        image_size=image_size,
        transform=transform,
    )

    if shuffle is None:
        shuffle = (split == "train")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == "train"),
    )

    print(f"📂 {split.upper()} dataset: {len(dataset)} pairs, {len(loader)} batches")
    return loader
