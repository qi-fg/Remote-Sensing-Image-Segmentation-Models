from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from rsseg.data import LOVEDA_CLASS_NAMES, LoveDADataset
from rsseg.engine import dice_loss


class LoveDAAdapterTest(unittest.TestCase):
    def _write_sample(self, root: Path, split: str, domain: str, name: str) -> None:
        image_dir = root / split / domain / "images_png"
        mask_dir = root / split / domain / "masks_png"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        image = np.zeros((4, 4, 3), dtype=np.uint8)
        image[..., 0] = 127
        mask = np.array(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [1, 2, 3, 4],
                [5, 6, 7, 0],
            ],
            dtype=np.uint8,
        )
        Image.fromarray(image).save(image_dir / name)
        Image.fromarray(mask).save(mask_dir / name)

    def test_official_layout_and_label_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_sample(root, "Train", "Urban", "urban.png")
            self._write_sample(root, "Train", "Rural", "rural.png")

            dataset = LoveDADataset(str(root), "train", domains=("urban", "rural"))
            self.assertEqual(len(dataset), 2)
            self.assertEqual(tuple(dataset.class_names), LOVEDA_CLASS_NAMES)

            image, mask = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 4, 4))
            self.assertEqual(set(mask.unique().tolist()), set(range(-1, 7)))
            self.assertEqual(int(mask[0, 0]), -1)
            self.assertEqual(int(mask[0, 1]), 0)
            self.assertEqual(int(mask[1, 3]), 6)


    def test_loveda_recipe_crop_preserves_shape_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_sample(root, "Train", "Urban", "urban.png")

            dataset = LoveDADataset(
                str(root),
                "train",
                domains=("urban",),
                crop_size=4,
                random_scale_range=(0.5, 0.5),
                flip_prob=0.0,
                photometric_distortion=False,
                cat_max_ratio=0.75,
            )
            image, mask = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 4, 4))
            self.assertEqual(tuple(mask.shape), (4, 4))
            self.assertTrue(set(mask.unique().tolist()).issubset(set(range(-1, 7))))
            self.assertIn(-1, mask.unique().tolist())

    def test_ignore_pixels_are_safe_in_dice(self) -> None:
        logits = torch.randn(1, 7, 2, 2, requires_grad=True)
        target = torch.tensor([[[-1, 0], [1, 6]]], dtype=torch.long)
        loss = dice_loss(logits, target, ignore_index=-1)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)


if __name__ == "__main__":
    unittest.main()
