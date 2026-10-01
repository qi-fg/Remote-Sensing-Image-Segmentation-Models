from __future__ import annotations

from itertools import zip_longest
import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


LOVEDA_CLASS_NAMES = (
    "background",
    "building",
    "road",
    "water",
    "barren",
    "forest",
    "agricultural",
)
LOVEDA_NUM_CLASSES = len(LOVEDA_CLASS_NAMES)
LOVEDA_IGNORE_INDEX = -1


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
    """Generic 8-bit image/mask folder dataset."""

    IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    def __init__(
        self,
        root: str,
        split: str,
        image_size: Optional[int] = None,
        max_samples: Optional[int] = None,
    ) -> None:
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


def _resize_for_loveda_recipe(
    image: Image.Image,
    mask: Image.Image,
    crop_size: int,
    ratio: float,
) -> tuple[Image.Image, Image.Image]:
    """Mimic MMSeg's (2048, 512) keep-ratio scaling for a 512 crop.

    Generalized as max_long_edge=4*crop_size and max_short_edge=crop_size,
    followed by the sampled ratio. For square LoveDA tiles this yields a
    sampled side length of crop_size * ratio.
    """
    width, height = image.size
    long_edge = max(width, height)
    short_edge = min(width, height)
    max_long = 4.0 * crop_size * ratio
    max_short = float(crop_size) * ratio
    scale = min(max_long / max(long_edge, 1), max_short / max(short_edge, 1))
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return (
        image.resize(new_size, Image.Resampling.BILINEAR),
        mask.resize(new_size, Image.Resampling.NEAREST),
    )


def _pad_to_crop(
    image: Image.Image,
    mask: Image.Image,
    crop_size: int,
) -> tuple[Image.Image, Image.Image]:
    width, height = image.size
    target_w = max(width, crop_size)
    target_h = max(height, crop_size)
    if (target_w, target_h) == (width, height):
        return image, mask

    padded_image = Image.new("RGB", (target_w, target_h), color=(0, 0, 0))
    padded_mask = Image.new("L", (target_w, target_h), color=0)
    padded_image.paste(image, (0, 0))
    padded_mask.paste(mask, (0, 0))
    return padded_image, padded_mask


def _random_crop_loveda(
    image: Image.Image,
    mask: Image.Image,
    crop_size: int,
    cat_max_ratio: float,
    attempts: int = 10,
) -> tuple[Image.Image, Image.Image]:
    width, height = image.size
    if width < crop_size or height < crop_size:
        image, mask = _pad_to_crop(image, mask, crop_size)
        width, height = image.size

    last_box = (0, 0, crop_size, crop_size)
    for _ in range(max(1, attempts)):
        left = random.randint(0, width - crop_size)
        top = random.randint(0, height - crop_size)
        box = (left, top, left + crop_size, top + crop_size)
        last_box = box

        if cat_max_ratio >= 1.0:
            break

        crop_mask = np.asarray(mask.crop(box))
        labels, counts = np.unique(crop_mask, return_counts=True)
        counts = counts[labels != 0]  # LoveDA raw 0 is no-data / ignore.
        if len(counts) > 1 and counts.max() / counts.sum() < cat_max_ratio:
            break

    return image.crop(last_box), mask.crop(last_box)


def _photometric_distortion(image: Image.Image) -> Image.Image:
    """Approximate MMSeg PhotoMetricDistortion without adding OpenCV."""
    arr = np.asarray(image, dtype=np.float32)

    if random.random() < 0.5:
        arr += random.uniform(-32.0, 32.0)

    contrast_first = random.random() < 0.5
    if contrast_first and random.random() < 0.5:
        arr *= random.uniform(0.5, 1.5)

    image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")

    if random.random() < 0.5:
        image = ImageEnhance.Color(image).enhance(random.uniform(0.5, 1.5))

    if random.random() < 0.5:
        hsv = np.asarray(image.convert("HSV"), dtype=np.uint8).copy()
        shift = int(round(random.uniform(-18.0, 18.0) / 360.0 * 255.0))
        hue = (hsv[..., 0].astype(np.int16) + shift) % 256
        hsv[..., 0] = hue.astype(np.uint8)
        image = Image.fromarray(hsv, mode="HSV").convert("RGB")

    if not contrast_first and random.random() < 0.5:
        arr = np.asarray(image, dtype=np.float32)
        arr *= random.uniform(0.5, 1.5)
        image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")

    return image


class LoveDADataset(Dataset):
    """LoveDA semantic-segmentation adapter using official folder and label rules.

    Official raw masks use 0 for no-data and 1..7 for the seven semantic
    classes. This adapter maps 0 to -1 (ignore) and 1..7 to 0..6, matching
    the official LoveDA loader's mask-minus-one convention.
    """

    IMAGE_EXTS = (".png", ".tif", ".tiff")
    SPLIT_DIRS = {"train": "Train", "val": "Val"}
    DOMAIN_DIRS = {"urban": "Urban", "rural": "Rural"}

    def __init__(
        self,
        root: str,
        split: str,
        domains: Sequence[str] = ("urban", "rural"),
        image_size: Optional[int] = None,
        max_samples: Optional[int] = None,
        crop_size: Optional[int] = None,
        random_scale_range: Optional[tuple[float, float]] = None,
        flip_prob: float = 0.0,
        photometric_distortion: bool = False,
        cat_max_ratio: float = 1.0,
    ) -> None:
        if image_size is not None and crop_size is not None:
            raise ValueError("Use either image_size or crop_size augmentation, not both.")
        if random_scale_range is not None:
            lo, hi = random_scale_range
            if lo <= 0 or hi < lo:
                raise ValueError("random_scale_range must satisfy 0 < min <= max.")
        if not 0.0 <= flip_prob <= 1.0:
            raise ValueError("flip_prob must be in [0, 1].")
        if not 0.0 < cat_max_ratio <= 1.0:
            raise ValueError("cat_max_ratio must be in (0, 1].")

        split_key = split.lower()
        if split_key not in self.SPLIT_DIRS:
            raise ValueError("LoveDA training adapter supports split='train' or 'val'.")

        normalized_domains = tuple(d.lower() for d in domains)
        unknown = sorted(set(normalized_domains) - set(self.DOMAIN_DIRS))
        if unknown:
            raise ValueError(f"Unknown LoveDA domain(s): {unknown}; use urban/rural.")

        split_root = Path(root) / self.SPLIT_DIRS[split_key]
        domain_pairs: list[list[tuple[Path, Path, str]]] = []

        for domain in normalized_domains:
            domain_root = split_root / self.DOMAIN_DIRS[domain]
            image_dir = domain_root / "images_png"
            mask_dir = domain_root / "masks_png"
            if not image_dir.is_dir() or not mask_dir.is_dir():
                raise FileNotFoundError(
                    f"Expected LoveDA directories {image_dir} and {mask_dir}. "
                    "Point --data to the directory that contains Train/ and Val/."
                )

            masks = {
                p.stem: p
                for p in mask_dir.iterdir()
                if p.is_file() and p.suffix.lower() in self.IMAGE_EXTS
            }
            pairs = []
            for image_path in sorted(image_dir.iterdir()):
                if not image_path.is_file() or image_path.suffix.lower() not in self.IMAGE_EXTS:
                    continue
                mask_path = masks.get(image_path.stem)
                if mask_path is None:
                    raise FileNotFoundError(
                        f"Missing LoveDA mask for image {image_path.name} in {mask_dir}"
                    )
                pairs.append((image_path, mask_path, domain))

            if not pairs:
                raise FileNotFoundError(f"No LoveDA image/mask pairs found in {domain_root}")
            domain_pairs.append(pairs)

        # Interleave domains so debug caps inspect both Urban and Rural.
        pairs = []
        for group in zip_longest(*domain_pairs):
            pairs.extend(item for item in group if item is not None)

        if max_samples is not None:
            pairs = pairs[: max(1, min(max_samples, len(pairs)))]

        self.pairs = pairs
        self.image_size = image_size
        self.crop_size = crop_size
        self.random_scale_range = random_scale_range
        self.flip_prob = float(flip_prob)
        self.photometric_distortion = bool(photometric_distortion)
        self.cat_max_ratio = float(cat_max_ratio)
        self.class_names = LOVEDA_CLASS_NAMES
        self.num_classes = LOVEDA_NUM_CLASSES
        self.ignore_index = LOVEDA_IGNORE_INDEX

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        image_path, mask_path, _domain = self.pairs[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.image_size is not None:
            size = (self.image_size, self.image_size)
            image = image.resize(size, Image.Resampling.BILINEAR)
            mask = mask.resize(size, Image.Resampling.NEAREST)
        elif self.crop_size is not None:
            ratio = 1.0
            if self.random_scale_range is not None:
                ratio = random.uniform(*self.random_scale_range)
            image, mask = _resize_for_loveda_recipe(
                image, mask, self.crop_size, ratio
            )
            image, mask = _random_crop_loveda(
                image, mask, self.crop_size, self.cat_max_ratio
            )
            if random.random() < self.flip_prob:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if self.photometric_distortion:
                image = _photometric_distortion(image)

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        raw_mask = np.asarray(mask)

        if raw_mask.ndim != 2:
            raise ValueError(
                f"LoveDA mask {mask_path} has shape {raw_mask.shape}; expected one channel."
            )

        raw_mask = raw_mask.astype(np.int64, copy=False)
        if raw_mask.size:
            min_label = int(raw_mask.min())
            max_label = int(raw_mask.max())
            if min_label < 0 or max_label > 7:
                raise ValueError(
                    f"Unexpected LoveDA raw label range [{min_label}, {max_label}] "
                    f"in {mask_path}; expected only 0..7."
                )

        mapped_mask = raw_mask - 1
        image_t = torch.from_numpy(image_np.transpose(2, 0, 1).copy())
        mask_t = torch.from_numpy(mapped_mask.copy())
        return image_t, mask_t


def loveda_domains(value: str) -> tuple[str, ...]:
    value = value.lower()
    if value == "both":
        return ("urban", "rural")
    if value in ("urban", "rural"):
        return (value,)
    raise ValueError("LoveDA domain must be one of: both, urban, rural.")
