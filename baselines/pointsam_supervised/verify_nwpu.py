#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED = {
    "train_images": 520,
    "val_images": 130,
    "train_instances": 3178,
    "val_instances": 743,
    "union_images": 650,
}


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Missing annotation file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify PointSAM NWPU split and images")
    parser.add_argument("--pointsam-dir", required=True)
    args = parser.parse_args()

    root = Path(args.pointsam_dir).expanduser().resolve()
    ann_dir = root / "data" / "NWPU" / "Annotations"
    image_dir = root / "data" / "NWPU" / "Images"

    train = load_json(ann_dir / "NWPU_instances_train.json")
    val = load_json(ann_dir / "NWPU_instances_val.json")

    train_files = {str(x["file_name"]) for x in train.get("images", [])}
    val_files = {str(x["file_name"]) for x in val.get("images", [])}
    overlap = train_files & val_files
    union = train_files | val_files

    summary = {
        "train_images": len(train_files),
        "val_images": len(val_files),
        "train_instances": len(train.get("annotations", [])),
        "val_instances": len(val.get("annotations", [])),
        "overlap_images": len(overlap),
        "union_images": len(union),
        "first_image": min(union) if union else None,
        "last_image": max(union) if union else None,
    }

    failures = []
    for key, expected in EXPECTED.items():
        if summary[key] != expected:
            failures.append(f"{key}: expected {expected}, got {summary[key]}")
    if overlap:
        failures.append(f"train/val overlap is not empty: {sorted(overlap)[:10]}")

    missing = []
    if not image_dir.is_dir():
        failures.append(f"image directory does not exist: {image_dir}")
    else:
        missing = sorted(name for name in union if not (image_dir / name).is_file())
        if missing:
            failures.append(
                f"missing {len(missing)} split images; first missing: {missing[:10]}"
            )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if failures:
        print("\nFAILED:")
        for item in failures:
            print(f"- {item}")
        raise SystemExit(1)

    print("\nNWPU PointSAM split and image set: OK")


if __name__ == "__main__":
    main()
