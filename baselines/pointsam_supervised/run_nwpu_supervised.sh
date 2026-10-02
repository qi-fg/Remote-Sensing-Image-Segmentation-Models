#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
POINTSAM_DIR="${POINTSAM_DIR:-$(cd "$REPO_ROOT/.." && pwd)/PointSAM}"
POINTS="${POINTS:-1 2 3}"
GPU="${GPU:-0}"

if [[ ! -f "$POINTSAM_DIR/train_supervise.py" ]]; then
    echo "PointSAM not found at $POINTSAM_DIR" >&2
    echo "Run baselines/pointsam_supervised/setup_upstream.sh first." >&2
    exit 1
fi

python "$SCRIPT_DIR/verify_nwpu.py" --pointsam-dir "$POINTSAM_DIR"

if [[ ! -f "$POINTSAM_DIR/pretrain/sam_vit_b_01ec64.pth" ]]; then
    echo "Missing SAM ViT-B checkpoint under $POINTSAM_DIR/pretrain" >&2
    exit 1
fi

cd "$POINTSAM_DIR"

for num_points in $POINTS; do
    if [[ "$num_points" != "1" && "$num_points" != "2" && "$num_points" != "3" ]]; then
        echo "POINTS must contain only 1, 2, or 3; got $num_points" >&2
        exit 1
    fi

    out_dir="work_dir/nwpu/supervise/point_${num_points}"
    echo "=== NWPU supervised SAM ViT-B: ${num_points}-point ==="
    CUDA_VISIBLE_DEVICES="$GPU" python train_supervise.py       --cfg configs.config_nwpu       --prompt point       --num_points "$num_points"       --out_dir "$out_dir"       --load_type load
done
