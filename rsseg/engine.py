from __future__ import annotations

from contextlib import nullcontext
from typing import Optional, Sequence

import torch
import torch.nn.functional as F


def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Multi-class Dice loss with explicit ignored pixels."""
    num_classes = logits.shape[1]
    valid = (target != ignore_index) & (target >= 0) & (target < num_classes)
    if not valid.any():
        return logits.sum() * 0.0

    safe_target = target.clone()
    safe_target[~valid] = 0

    probs = F.softmax(logits, dim=1)
    one_hot = F.one_hot(safe_target, num_classes).permute(0, 3, 1, 2).float()
    valid_f = valid.unsqueeze(1).float()

    probs = probs * valid_f
    one_hot = one_hot * valid_f
    dims = (0, 2, 3)
    inter = (probs * one_hot).sum(dims)
    card = probs.sum(dims) + one_hot.sum(dims)

    present = card > 0
    if not present.any():
        return logits.sum() * 0.0

    dice = (2.0 * inter[present] + eps) / (card[present] + eps)
    return 1.0 - dice.mean()


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    scaler,
    amp: bool,
    ignore_index: int = -1,
    loss_name: str = "ce_dice",
    scheduler=None,
) -> float:
    model.train()
    total_loss = 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with _autocast(device, amp):
            logits = model(images)
            ce = F.cross_entropy(logits, masks, ignore_index=ignore_index)
            if loss_name == "ce":
                loss = ce
            elif loss_name == "ce_dice":
                loss = ce + dice_loss(logits, masks, ignore_index=ignore_index)
            else:
                raise ValueError(f"Unknown loss '{loss_name}'. Use ce or ce_dice.")

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += float(loss.detach()) * images.size(0)

    return total_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    num_classes: int,
    amp: bool = False,
    class_names: Optional[Sequence[str]] = None,
) -> dict:
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

    iou_values = [None if torch.isnan(v) else float(v) for v in per_class_iou]
    result = {
        "miou": float(miou),
        "oa": float(oa),
        "per_class_iou": iou_values,
        "confusion_matrix": confusion.tolist(),
    }

    if class_names is not None:
        if len(class_names) != num_classes:
            raise ValueError("class_names length must equal num_classes")
        result["per_class_iou_named"] = {
            str(name): value for name, value in zip(class_names, iou_values)
        }

    return result
