from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticSegDataset(Dataset):
    """Small learnable segmentation task used only for smoke tests."""

    def __init__(
        self,
        n: int = 24,
        size: int = 64,
        in_channels: int = 3,
        seed: int = 0,
    ) -> None:
        g = torch.Generator().manual_seed(seed)
        self.images = torch.rand(n, in_channels, size, size, generator=g) * 0.25
        self.masks = torch.zeros(n, size, size, dtype=torch.long)

        radius = max(4, size // 8)
        for i in range(n):
            cy = torch.randint(radius, size - radius, (1,), generator=g).item()
            cx = torch.randint(radius, size - radius, (1,), generator=g).item()
            self.images[i, :, cy - radius : cy + radius, cx - radius : cx + radius] += 1.5
            self.masks[i, cy - radius : cy + radius, cx - radius : cx + radius] = 1

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        return self.images[index], self.masks[index]


class PairedSegDataset(Dataset):
    """Generic 8-bit image/mask folder dataset.

    Expected layout::

        root/
          train/
            images/
            masks/
          val/
            images/
            masks/

    Images and masks are paired by filename stem. Masks must contain integer
    class ids. This adapter intentionally stays simple; dataset-specific
    adapters (for example LoveDA) can be added later without changing the
    training engine.
    """

    IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    def __init__(
        self,
        root: str,
        split: str,
        image_size: Optional[int] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        from PIL import Image  # noqa: F401

        split_root = Path(root) / split
        image_dir = split_root / "images"
        mask_dir = split_root / "masks"
        if not image_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {image_dir} and {mask_dir}. "
                "See UNIFIED_TRAINING.md for the required folder layout."
            )

        mask_by_stem = {
            p.stem: p
            for p in mask_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.IMAGE_EXTS
        }
        pairs = []
        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in self.IMAGE_EXTS:
                continue
            mask_path = mask_by_stem.get(image_path.stem)
            if mask_path is not None:
                pairs.append((image_path, mask_path))

        if not pairs:
            raise FileNotFoundError(f"No matched image/mask pairs found in {split_root}")

        if max_samples is not None:
            pairs = pairs[: max(1, min(max_samples, len(pairs)))]

        self.pairs = pairs
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        from PIL import Image

        image_path, mask_path = self.pairs[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.image_size is not None:
            size = (self.image_size, self.image_size)
            image = image.resize(size, Image.Resampling.BILINEAR)
            mask = mask.resize(size, Image.Resampling.NEAREST)

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        mask_np = np.asarray(mask)

        if mask_np.ndim != 2:
            raise ValueError(
                f"Mask {mask_path} has shape {mask_np.shape}; expected one channel "
                "with integer class ids."
            )

        image_t = torch.from_numpy(image_np.transpose(2, 0, 1).copy())
        mask_t = torch.from_numpy(mask_np.astype(np.int64, copy=True))
        return image_t, mask_t
