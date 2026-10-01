from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    num_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * one_hot).sum(dims)
    card = probs.sum(dims) + one_hot.sum(dims)
    dice = (2.0 * inter + eps) / (card + eps)
    return 1.0 - dice.mean()


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def train_one_epoch(model, loader, optimizer, device, scaler, amp: bool) -> float:
    model.train()
    total_loss = 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with _autocast(device, amp):
            logits = model(images)
            loss = F.cross_entropy(logits, masks) + dice_loss(logits, masks)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach()) * images.size(0)

    return total_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int, amp: bool = False) -> dict:
    model.eval()
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with _autocast(device, amp):
            logits = model(images)
        pred = logits.argmax(dim=1)

        valid = (masks >= 0) & (masks < num_classes)
        target = masks[valid].to(torch.int64).cpu()
        output = pred[valid].to(torch.int64).cpu()
        bins = torch.bincount(
            num_classes * target + output,
            minlength=num_classes * num_classes,
        )
        confusion += bins.reshape(num_classes, num_classes)

    tp = confusion.diag().float()
    denom = confusion.sum(1).float() + confusion.sum(0).float() - tp
    valid_classes = denom > 0

    per_class_iou = torch.full((num_classes,), float("nan"))
    per_class_iou[valid_classes] = tp[valid_classes] / denom[valid_classes]
    miou = per_class_iou[valid_classes].mean().item() if valid_classes.any() else 0.0
    oa = tp.sum().item() / max(confusion.sum().item(), 1)

    return {
        "miou": float(miou),
        "oa": float(oa),
        "per_class_iou": [
            None if torch.isnan(v) else float(v)
            for v in per_class_iou
        ],
        "confusion_matrix": confusion.tolist(),
    }
