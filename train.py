"""Unified training entry for remote-sensing segmentation baselines."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from rsseg.data import (
    LOVEDA_CLASS_NAMES,
    LOVEDA_IGNORE_INDEX,
    LOVEDA_NUM_CLASSES,
    LoveDADataset,
    PairedSegDataset,
    SyntheticSegDataset,
    loveda_domains,
)
from rsseg.engine import evaluate, train_one_epoch
from rsseg.models import SUPPORTED_MODELS, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified RS segmentation trainer")
    parser.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
    parser.add_argument("--mode", choices=("smoke", "debug", "full"), default="smoke")
    parser.add_argument("--dataset", choices=("generic", "loveda"), default="generic")
    parser.add_argument("--data", type=str, default=None, help="dataset root for debug/full")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--domain", choices=("both", "urban", "rural"), default="both",
                        help="LoveDA domain selection")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--variant", choices=("tiny", "b0"), default=None,
                        help="repository SegFormer variant; smoke always uses tiny")
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="pretrained initialization; segformer_b0 defaults to NVIDIA ImageNet-1K weights",
    )
    parser.add_argument(
        "--loss",
        choices=("ce", "ce_dice"),
        default=None,
        help="training loss; segformer_b0 defaults to CE, repository models to CE+Dice",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=None,
                        help="optional square resize; debug defaults to 256")
    parser.add_argument("--debug-train-samples", type=int, default=128)
    parser.add_argument("--debug-val-samples", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="mixed precision; defaults on for CUDA debug/full and off for smoke/CPU",
    )
    parser.add_argument("--resume", type=str, default=None,
                        help="resume model/optimizer state from a checkpoint")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def resolve_num_classes(args: argparse.Namespace) -> int:
    if args.dataset == "loveda":
        if args.num_classes not in (None, LOVEDA_NUM_CLASSES):
            raise SystemExit(
                f"LoveDA has {LOVEDA_NUM_CLASSES} classes after label remapping; "
                f"got --num-classes {args.num_classes}."
            )
        return LOVEDA_NUM_CLASSES
    return args.num_classes if args.num_classes is not None else 2


def resolve_defaults(args: argparse.Namespace, device: torch.device) -> dict:
    epochs = args.epochs
    if epochs is None:
        epochs = {"smoke": 1, "debug": 3, "full": 100}[args.mode]

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = {"smoke": 4, "debug": 4, "full": 8}[args.mode]

    num_workers = args.num_workers
    if num_workers is None:
        num_workers = 0 if args.mode != "full" else 4

    image_size = args.image_size
    if image_size is None and args.mode == "debug":
        image_size = 256

    if args.lr is not None:
        lr = args.lr
    elif args.model == "unet":
        lr = 1e-3
    elif args.model == "segformer_b0":
        lr = 6e-5
    else:
        lr = 6e-4

    weight_decay = (
        args.weight_decay
        if args.weight_decay is not None
        else (1e-4 if args.model == "unet" else 1e-2)
    )

    pretrained = args.pretrained
    if pretrained is None:
        pretrained = args.model == "segformer_b0"

    loss_name = args.loss
    if loss_name is None:
        loss_name = "ce" if args.model == "segformer_b0" else "ce_dice"

    amp = args.amp
    if amp is None:
        amp = device.type == "cuda" and args.mode in ("debug", "full")
    if device.type != "cuda":
        amp = False

    output = args.output or f"runs/{args.dataset}-{args.model}-{args.mode}"

    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "image_size": image_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "pretrained": pretrained,
        "loss": loss_name,
        "amp": amp,
        "output": output,
    }


def build_datasets(args: argparse.Namespace, cfg: dict):
    if args.mode == "smoke":
        return (
            SyntheticSegDataset(n=24, size=64, in_channels=args.in_channels, seed=args.seed),
            SyntheticSegDataset(n=8, size=64, in_channels=args.in_channels, seed=args.seed + 1),
        )

    if not args.data:
        raise SystemExit("--data is required for debug/full mode")

    train_cap = args.debug_train_samples if args.mode == "debug" else None
    val_cap = args.debug_val_samples if args.mode == "debug" else None

    if args.dataset == "loveda":
        domains = loveda_domains(args.domain)
        train_ds = LoveDADataset(
            args.data,
            args.train_split,
            domains=domains,
            image_size=cfg["image_size"],
            max_samples=train_cap,
        )
        val_ds = LoveDADataset(
            args.data,
            args.val_split,
            domains=domains,
            image_size=cfg["image_size"],
            max_samples=val_cap,
        )
        return train_ds, val_ds

    train_ds = PairedSegDataset(
        args.data,
        args.train_split,
        image_size=cfg["image_size"],
        max_samples=train_cap,
    )
    val_ds = PairedSegDataset(
        args.data,
        args.val_split,
        image_size=cfg["image_size"],
        max_samples=val_cap,
    )
    return train_ds, val_ds


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    num_classes = resolve_num_classes(args)
    cfg = resolve_defaults(args, device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_ds, val_ds = build_datasets(args, cfg)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        args.model,
        num_classes=num_classes,
        in_channels=args.in_channels,
        mode=args.mode,
        variant=args.variant,
        pretrained=bool(cfg["pretrained"]) and not bool(args.resume),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    if cfg["amp"]:
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    ignore_index = LOVEDA_IGNORE_INDEX if args.dataset == "loveda" else -1
    class_names = LOVEDA_CLASS_NAMES if args.dataset == "loveda" else None

    output_dir = Path(cfg["output"])
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved = vars(args).copy()
    resolved.update(cfg)
    resolved["num_classes"] = num_classes
    resolved["ignore_index"] = ignore_index
    resolved["class_names"] = list(class_names) if class_names is not None else None
    resolved["device_resolved"] = str(device)
    resolved["train_samples"] = len(train_ds)
    resolved["val_samples"] = len(val_ds)
    resolved["parameters"] = sum(p.numel() for p in model.parameters())
    resolved["git_commit"] = git_commit()
    resolved["torch_version"] = str(torch.__version__)
    resolved["cuda_runtime"] = None if torch.version.cuda is None else str(torch.version.cuda)
    if args.model == "segformer_b0":
        resolved["pretrained_source"] = "nvidia/mit-b0" if cfg["pretrained"] else None
        try:
            import transformers
            resolved["transformers_version"] = transformers.__version__
        except ImportError:
            resolved["transformers_version"] = None
    save_json(output_dir / "config.json", resolved)

    start_epoch = 1
    best_miou = -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_miou = float(checkpoint.get("best_miou", -1.0))
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    print(
        f"dataset={args.dataset} model={args.model} mode={args.mode} "
        f"device={device} amp={cfg['amp']} pretrained={cfg['pretrained']} "
        f"loss={cfg['loss']} params={resolved['parameters']/1e6:.3f}M "
        f"train={len(train_ds)} val={len(val_ds)}"
    )

    final_metrics = None
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            cfg["amp"],
            ignore_index=ignore_index,
            loss_name=cfg["loss"],
        )
        metrics = evaluate(
            model,
            val_loader,
            device,
            num_classes,
            cfg["amp"],
            class_names=class_names,
        )
        final_metrics = metrics

        print(
            f"epoch {epoch:3d}/{cfg['epochs']} "
            f"loss={loss:.4f} mIoU={metrics['miou']:.4f} OA={metrics['oa']:.4f}"
        )

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_miou": max(best_miou, metrics["miou"]),
            "config": resolved,
        }
        torch.save(checkpoint, output_dir / "last.pt")

        if metrics["miou"] > best_miou:
            best_miou = metrics["miou"]
            checkpoint["best_miou"] = best_miou
            torch.save(checkpoint, output_dir / "best.pt")

    if final_metrics is None:
        final_metrics = evaluate(
            model,
            val_loader,
            device,
            num_classes,
            cfg["amp"],
            class_names=class_names,
        )

    result = {
        "dataset": args.dataset,
        "model": args.model,
        "mode": args.mode,
        "best_miou": best_miou,
        "final": final_metrics,
        "output": str(output_dir),
    }
    save_json(output_dir / "results.json", result)

    print(f"saved: {output_dir / 'results.json'}")
    print(f"best mIoU={best_miou:.4f}")


if __name__ == "__main__":
    main()
