"""Evaluate a checkpoint produced by the unified trainer."""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from rsseg.data import (
    LOVEDA_CLASS_NAMES,
    LoveDADataset,
    PairedSegDataset,
    loveda_domains,
)
from rsseg.engine import evaluate
from rsseg.models import SUPPORTED_MODELS, build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an RS segmentation checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--dataset", choices=("generic", "loveda"), default=None)
    parser.add_argument("--domain", choices=("both", "urban", "rural"), default=None)
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    saved = checkpoint.get("config", {})

    model_name = args.model or saved.get("model")
    if model_name is None:
        raise SystemExit("Checkpoint has no model metadata; pass --model.")

    dataset_name = args.dataset or saved.get("dataset", "generic")
    domain = args.domain or saved.get("domain", "both")

    num_classes = args.num_classes or saved.get("num_classes")
    if num_classes is None:
        raise SystemExit("Checkpoint has no class-count metadata; pass --num-classes.")

    image_size = args.image_size
    if image_size is None:
        image_size = saved.get("image_size")

    if dataset_name == "loveda":
        dataset = LoveDADataset(
            args.data,
            args.split,
            domains=loveda_domains(domain),
            image_size=image_size,
        )
        class_names = LOVEDA_CLASS_NAMES
    else:
        dataset = PairedSegDataset(args.data, args.split, image_size=image_size)
        class_names = saved.get("class_names")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        model_name,
        num_classes=int(num_classes),
        in_channels=int(saved.get("in_channels", 3)),
        mode=saved.get("mode", "full"),
        variant=saved.get("variant"),
    ).to(device)
    model.load_state_dict(checkpoint["model"])

    use_amp = bool(saved.get("amp", False)) and device.type == "cuda"
    metrics = evaluate(
        model,
        loader,
        device,
        int(num_classes),
        amp=use_amp,
        class_names=class_names,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
