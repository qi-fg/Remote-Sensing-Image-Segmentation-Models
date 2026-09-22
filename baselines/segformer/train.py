"""Train a SegFormer baseline for remote sensing semantic segmentation.

Demo (offline, CPU-friendly)::

    python train.py --demo

Real data (same folder layout as the U-Net baseline)::

    python train.py --data /path/to/dataset --epochs 50 --batch-size 8

Data layout::

    dataset/images/*.png|*.tif    (H x W x C)
    dataset/masks/*.png           (H x W, integer class ids)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import SegFormer  # noqa: E402

# Reuse the dataset / loss utilities from the U-Net baseline (keeps the two
# baselines consistent). Fall back gracefully if paths differ.
_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS, "..", "unet"))
from train import SyntheticSegData, FolderSegData, dice_loss, mean_iou  # noqa: E402


def train_one_epoch(model, loader, optim, device):
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
    ap = argparse.ArgumentParser(description="SegFormer baseline for RS segmentation")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--in-channels", type=int, default=3)
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--variant", choices=["tiny", "b0"], default="tiny",
                    help="tiny = fast CPU demo; b0 = paper B0 channel widths")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)

    if args.demo or not args.data:
        if not args.demo:
            print("[warn] no --data given, falling back to --demo (synthetic data).")
        train_ds = SyntheticSegData(n=64, size=64, in_ch=args.in_channels,
                                    num_classes=args.num_classes, seed=0)
        val_ds = SyntheticSegData(n=16, size=64, in_ch=args.in_channels,
                                  num_classes=args.num_classes, seed=1)
    else:
        train_ds = FolderSegData(args.data, in_ch=args.in_channels)
        val_ds = train_ds

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.variant == "b0":
        model = SegFormer(in_channels=args.in_channels, num_classes=args.num_classes,
                          embed_dims=(32, 64, 160, 256), depths=(2, 2, 2, 2),
                          num_heads=(1, 2, 5, 8), sr_ratios=(8, 4, 2, 1))
    else:
        model = SegFormer.tiny(in_channels=args.in_channels, num_classes=args.num_classes)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    print(f"device={device}  variant={args.variant}  params={n_params/1e6:.3f}M  "
          f"train={len(train_ds)}  val={len(val_ds)}")
    t0 = time.time()
    miou = 0.0
    for ep in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_ld, optim, device)
        miou = evaluate(model, val_ld, device, args.num_classes)
        if ep == 1 or ep % max(1, args.epochs // 10) == 0 or ep == args.epochs:
            print(f"epoch {ep:3d}/{args.epochs}  loss={loss:.4f}  mIoU={miou:.4f}")
    print(f"done in {time.time() - t0:.1f}s  final mIoU={miou:.4f}")


if __name__ == "__main__":
    main()
