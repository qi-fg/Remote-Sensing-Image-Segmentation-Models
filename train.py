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
    parser.add_argument(
        "--recipe",
        choices=("loveda_segformer_b0_512",),
        default=None,
        help="named repository-controlled training protocol",
    )
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
    parser.add_argument("--eval-every", type=int, default=None,
                        help="validation interval in epochs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--scheduler", choices=("constant", "poly"), default=None)
    parser.add_argument("--warmup-iters", type=int, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--poly-power", type=float, default=None)
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
    recipe = args.recipe
    if recipe == "loveda_segformer_b0_512":
        if args.dataset != "loveda" or args.model != "segformer_b0":
            raise SystemExit(
                "--recipe loveda_segformer_b0_512 requires "
                "--dataset loveda --model segformer_b0."
            )
        if args.mode == "smoke":
            raise SystemExit(
                "--recipe loveda_segformer_b0_512 is for real-data debug/full runs, "
                "not smoke mode."
            )

    epochs = args.epochs
    if epochs is None:
        epochs = {"smoke": 1, "debug": 3, "full": 100}[args.mode]

    eval_every = args.eval_every
    if eval_every is None:
        eval_every = 5 if recipe == "loveda_segformer_b0_512" and args.mode == "full" else 1
    if eval_every < 1:
        raise SystemExit("--eval-every must be >= 1.")

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = {"smoke": 4, "debug": 4, "full": 8}[args.mode]

    val_batch_size = args.val_batch_size
    if val_batch_size is None:
        val_batch_size = 1 if recipe == "loveda_segformer_b0_512" else batch_size

    num_workers = args.num_workers
    if num_workers is None:
        num_workers = 0 if args.mode != "full" else 4

    image_size = args.image_size
    if image_size is None and args.mode == "debug" and recipe is None:
        image_size = 256
    if recipe == "loveda_segformer_b0_512" and image_size is not None:
        raise SystemExit(
            "The LoveDA 512 recipe uses random 512 crops; do not combine it "
            "with --image-size."
        )

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

    scheduler = args.scheduler
    if scheduler is None:
        scheduler = "poly" if recipe == "loveda_segformer_b0_512" else "constant"
    warmup_iters = (
        args.warmup_iters
        if args.warmup_iters is not None
        else (1500 if recipe == "loveda_segformer_b0_512" else 0)
    )
    warmup_ratio = (
        args.warmup_ratio
        if args.warmup_ratio is not None
        else (1e-6 if recipe == "loveda_segformer_b0_512" else 1.0)
    )
    poly_power = args.poly_power if args.poly_power is not None else 1.0

    output = args.output
    if output is None:
        suffix = "-512" if recipe == "loveda_segformer_b0_512" else ""
        output = f"runs/{args.dataset}-{args.model}-{args.mode}{suffix}"

    return {
        "epochs": epochs,
        "eval_every": eval_every,
        "batch_size": batch_size,
        "val_batch_size": val_batch_size,
        "num_workers": num_workers,
        "image_size": image_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "pretrained": pretrained,
        "loss": loss_name,
        "amp": amp,
        "output": output,
        "scheduler": scheduler,
        "warmup_iters": warmup_iters,
        "warmup_ratio": warmup_ratio,
        "poly_power": poly_power,
        "crop_size": 512 if recipe == "loveda_segformer_b0_512" else None,
        "random_scale_range": (0.5, 2.0)
        if recipe == "loveda_segformer_b0_512"
        else None,
        "flip_prob": 0.5 if recipe == "loveda_segformer_b0_512" else 0.0,
        "photometric_distortion": recipe == "loveda_segformer_b0_512",
        "cat_max_ratio": 0.75 if recipe == "loveda_segformer_b0_512" else 1.0,
        "optimizer_profile": "segformer_official"
        if recipe == "loveda_segformer_b0_512"
        else "uniform",
        "head_lr_mult": 10.0 if recipe == "loveda_segformer_b0_512" else 1.0,
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
            crop_size=cfg["crop_size"],
            random_scale_range=cfg["random_scale_range"],
            flip_prob=cfg["flip_prob"],
            photometric_distortion=cfg["photometric_distortion"],
            cat_max_ratio=cfg["cat_max_ratio"],
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


def build_optimizer(model, cfg: dict) -> torch.optim.Optimizer:
    if cfg["optimizer_profile"] != "segformer_official":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )

    groups: dict[tuple[float, float], list[torch.nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        lower = name.lower()
        is_head = name.startswith("model.decode_head.")
        is_norm = any(
            key in lower
            for key in ("layer_norm", "layernorm", "batch_norm", "batchnorm")
        )
        lr = cfg["lr"] * (cfg["head_lr_mult"] if is_head else 1.0)
        weight_decay = 0.0 if is_norm else cfg["weight_decay"]
        groups.setdefault((lr, weight_decay), []).append(parameter)

    param_groups = [
        {"params": params, "lr": lr, "weight_decay": weight_decay}
        for (lr, weight_decay), params in groups.items()
    ]
    return torch.optim.AdamW(
        param_groups,
        lr=cfg["lr"],
        betas=(0.9, 0.999),
        weight_decay=cfg["weight_decay"],
    )


def build_scheduler(optimizer, cfg: dict, total_iters: int):
    if cfg["scheduler"] == "constant":
        return None
    if cfg["scheduler"] != "poly":
        raise ValueError(f"Unsupported scheduler: {cfg['scheduler']}")

    warmup_iters = min(max(int(cfg["warmup_iters"]), 0), max(total_iters - 1, 0))
    warmup_ratio = float(cfg["warmup_ratio"])
    power = float(cfg["poly_power"])

    def lr_lambda(step: int) -> float:
        if warmup_iters > 0 and step < warmup_iters:
            alpha = step / warmup_iters
            return warmup_ratio + alpha * (1.0 - warmup_ratio)

        decay_steps = max(total_iters - warmup_iters, 1)
        progress = min(max((step - warmup_iters) / decay_steps, 0.0), 1.0)
        return (1.0 - progress) ** power

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


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
        batch_size=cfg["val_batch_size"],
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

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(
        optimizer,
        cfg,
        total_iters=max(cfg["epochs"] * len(train_loader), 1),
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
    best_metrics = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        restore_rng_state(checkpoint.get("rng_state"))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_miou = float(checkpoint.get("best_miou", -1.0))
        best_metrics = checkpoint.get("best_metrics")
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    print(
        f"dataset={args.dataset} model={args.model} mode={args.mode} "
        f"device={device} amp={cfg['amp']} pretrained={cfg['pretrained']} "
        f"loss={cfg['loss']} scheduler={cfg['scheduler']} "
        f"params={resolved['parameters']/1e6:.3f}M "
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
            scheduler=scheduler,
        )

        should_evaluate = (
            epoch % cfg["eval_every"] == 0 or epoch == cfg["epochs"]
        )
        metrics = None
        is_best = False
        if should_evaluate:
            metrics = evaluate(
                model,
                val_loader,
                device,
                num_classes,
                cfg["amp"],
                class_names=class_names,
            )
            final_metrics = metrics
            is_best = metrics["miou"] > best_miou
            if is_best:
                best_miou = metrics["miou"]
                best_metrics = metrics

            print(
                f"epoch {epoch:3d}/{cfg['epochs']} "
                f"loss={loss:.4f} mIoU={metrics['miou']:.4f} OA={metrics['oa']:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )
        else:
            print(
                f"epoch {epoch:3d}/{cfg['epochs']} "
                f"loss={loss:.4f} val=skipped "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "rng_state": capture_rng_state(),
            "epoch": epoch,
            "metrics": metrics,
            "best_miou": best_miou,
            "best_metrics": best_metrics,
            "config": resolved,
        }
        torch.save(checkpoint, output_dir / "last.pt")

        if is_best:
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
        "best": best_metrics,
        "final": final_metrics,
        "output": str(output_dir),
    }
    save_json(output_dir / "results.json", result)

    print(f"saved: {output_dir / 'results.json'}")
    print(f"best mIoU={best_miou:.4f}")


if __name__ == "__main__":
    main()
