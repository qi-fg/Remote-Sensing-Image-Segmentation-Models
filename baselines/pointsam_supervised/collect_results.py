#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PAPER_TARGETS = {
    1: {"iou": 78.73, "f1": 86.74},
    2: {"iou": 80.88, "f1": 88.58},
    3: {"iou": 81.12, "f1": 88.79},
}


def read_best(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)

    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                epoch = int(float(row["epoch"]))
                iou = float(row["IoU"])
                f1 = float(row["F1"])
            except (KeyError, TypeError, ValueError):
                continue
            if epoch <= 0:
                continue
            rows.append({"epoch": epoch, "iou": iou, "f1": f1})

    if not rows:
        raise ValueError(f"No trained validation rows found in {path}")
    return max(rows, key=lambda row: row["iou"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect PointSAM supervised NWPU results")
    parser.add_argument("--pointsam-dir", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.pointsam_dir).expanduser().resolve()
    results = {}
    missing = []

    for points in (1, 2, 3):
        csv_path = root / "work_dir" / "nwpu" / "supervise" / f"point_{points}" / "metrics.csv"
        try:
            best = read_best(csv_path)
        except FileNotFoundError:
            missing.append(str(csv_path))
            continue

        target = PAPER_TARGETS[points]
        results[str(points)] = {
            "best_epoch": best["epoch"],
            "iou": best["iou"],
            "f1": best["f1"],
            "paper_target_iou": target["iou"],
            "paper_target_f1": target["f1"],
            "delta_iou": best["iou"] * 100.0 - target["iou"]
            if best["iou"] <= 1.0
            else best["iou"] - target["iou"],
            "delta_f1": best["f1"] * 100.0 - target["f1"]
            if best["f1"] <= 1.0
            else best["f1"] - target["f1"],
        }

    payload = {
        "dataset": "NWPU VHR-10",
        "method": "PointSAM supervised / SAM ViT-B",
        "upstream_commit": "a4b9253bca3db4932dbaea33d2bf1fee66d92e28",
        "results": results,
        "missing_metrics": missing,
    }

    output = (
        Path(args.output)
        if args.output
        else root / "work_dir" / "nwpu" / "supervise" / "summary.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nsaved: {output}")


if __name__ == "__main__":
    main()
