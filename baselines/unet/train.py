"""Train a U-Net baseline for remote sensing semantic segmentation.

Two modes
---------
1) Demo (offline, no data, CPU-friendly)::

       python train.py --demo

   Builds a synthetic segmentation task (a bright square on random
   background) so you can verify the whole pipeline runs end-to-end and
   watch mIoU climb. Takes a few seconds on CPU.

2) Real data::

       python train.py --data /path/to/dataset --epochs 50 --batch-size 8

   Expected folder layout::

       dataset/
         images/   *.png|*.tif   (H x W x C, C = 1/3/N bands)
         masks/    *.png         (H x W, single channel, integer class ids)

   Masks are matched to images by file stem.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Allow `python baselines/unet/train.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import UNet  # noqa: E402


# --------------------------------------------------------------------------- #
# Losses / metrics
# --------------------------------------------------------------------------- #
def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft multi-class Dice loss. logits (B,C,H,W), target (B,H,W) long."""
    num_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    tgt = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * tgt).sum(dims)
    card = probs.sum(dims) + tgt.sum(dims)
    dice = (2.0 * inter + eps) / (card + eps)
    return 1.0 - dice.mean()


@torch.no_grad()
def mean_iou(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    pred = logits.argmax(1)
    ious = []
    for c in range(num_classes):
        p, t = pred == c, target == c
        union = (p | t).sum().item()
        if union > 0:
            ious.append((p & t).sum().item() / union)
    return sum(ious) / len(ious) if ious else 0.0


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class SyntheticSegData(Dataset):
    """Synthetic RS-like segmentation: a bright square blob on noisy background.

    The signal is learnable, so mIoU should rise well above the ~1/C random
    baseline within a few epochs -- this proves the pipeline is wired up.
    """

    def __init__(self, n: int = 64, size: int = 64, in_ch: int = 3,
                 num_classes: int = 2, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.imgs = torch.rand(n, in_ch, size, size, generator=g) * 0.3
        self.masks = torch.zeros(n, size, size, dtype=torch.long)
        r = max(4, size // 8)
        for i in range(n):
            cy = torch.randint(r, size - r, (1,), generator=g).item()
            cx = torch.randint(r, size - r, (1,), generator=g).item()
            self.imgs[i, :, cy - r:cy + r, cx - r:cx + r] += 2.0
            self.masks[i, cy - r:cy + r, cx - r:cx + r] = 1

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, i: int):
        return self.imgs[i], self.masks[i]


class FolderSegData(Dataset):
    """Real data from `images/` + `masks/` folders (needs pillow + numpy)."""

    def __init__(self, root: str, in_ch: int = 3) -> None:
        import numpy as np
        from PIL import Image

        self._np = np
        self._Image = Image
        self.in_ch = in_ch
        img_dir = os.path.join(root, "images")
        msk_dir = os.path.join(root, "masks")
        exts = (".png", ".tif", ".tiff", ".jpg")
        self.pairs = []
        for fn in sorted(os.listdir(img_dir)):
            if not fn.lower().endswith(exts):
                continue
            stem = os.path.splitext(fn)[0]
            for mext in exts:
                mp = os.path.join(msk_dir, stem + mext)
                if os.path.exists(mp):
                    self.pairs.append((os.path.join(img_dir, fn), mp))
                    break
        if not self.pairs:
            raise FileNotFoundError(f"no image/mask pairs found under {root}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        np, Image = self._np, self._Image
        ip, mp = self.pairs[i]
        img = np.asarray(Image.open(ip)).astype("float32")
        if img.ndim == 2:
            img = img[..., None]
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))[: self.in_ch]
        msk = np.asarray(Image.open(mp)).astype("int64")
        return torch.from_numpy(img.copy()), torch.from_numpy(msk.copy())


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, optim, device, num_classes):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y) + dice_loss(logits, y)
        loss.backward()
        optim.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    miou, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        miou += mean_iou(model(x), y, num_classes)
        n += 1
    return miou / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="U-Net baseline for RS segmentation")
    ap.add_argument("--demo", action="store_true", help="run on synthetic data (offline)")
    ap.add_argument("--data", type=str, default=None, help="dataset root (images/ + masks/)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--in-channels", type=int, default=3)
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--base", type=int, default=32, help="U-Net base width")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.demo or not args.data:
        if not args.demo:
            print("[warn] no --data given, falling back to --demo (synthetic data).")
        size = 64
        base_w = 8  # keep demo fast on CPU
        train_ds = SyntheticSegData(n=64, size=size, in_ch=args.in_channels,
                                    num_classes=args.num_classes, seed=0)
        val_ds = SyntheticSegData(n=16, size=size, in_ch=args.in_channels,
                                  num_classes=args.num_classes, seed=1)
        base_w = args.base if args.base != 32 else base_w
    else:
        train_ds = FolderSegData(args.data, in_ch=args.in_channels)
        val_ds = train_ds
        base_w = args.base

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = UNet(in_channels=args.in_channels, num_classes=args.num_classes, base=base_w).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"device={device}  params={n_params/1e6:.3f}M  "
          f"train={len(train_ds)}  val={len(val_ds)}")
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_ld, optim, device, args.num_classes)
        miou = evaluate(model, val_ld, device, args.num_classes)
        if ep == 1 or ep % max(1, args.epochs // 10) == 0 or ep == args.epochs:
            print(f"epoch {ep:3d}/{args.epochs}  loss={loss:.4f}  mIoU={miou:.4f}")
    print(f"done in {time.time() - t0:.1f}s  final mIoU={miou:.4f}")


if __name__ == "__main__":
    main()
