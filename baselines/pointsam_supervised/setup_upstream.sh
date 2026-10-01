#!/usr/bin/env bash
set -euo pipefail

PIN_SHA="a4b9253bca3db4932dbaea33d2bf1fee66d92e28"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
POINTSAM_DIR="${POINTSAM_DIR:-$(cd "$REPO_ROOT/.." && pwd)/PointSAM}"

python - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(
        f"PointSAM/ReSAM reproduction env must use Python 3.10; got {sys.version.split()[0]}"
    )
print("Python:", sys.version.split()[0])
PY

python - <<'PY'
try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "PyTorch is not installed. For RTX 5090 use a CUDA 12.8+ build, "
        "for example the PyTorch 2.7.0+cu128 environment documented in README."
    ) from exc
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))
    archs = set(torch.cuda.get_arch_list())
    print("Compiled CUDA archs:", sorted(archs))
    if torch.cuda.get_device_capability(0) >= (12, 0) and "sm_120" not in archs:
        raise SystemExit(
            "This PyTorch build does not contain sm_120 support required by RTX 5090. "
            "Install a CUDA 12.8+ PyTorch build."
        )
PY

if [[ ! -d "$POINTSAM_DIR/.git" ]]; then
    git clone https://github.com/Lans1ng/PointSAM.git "$POINTSAM_DIR"
fi

git -C "$POINTSAM_DIR" fetch origin main
git -C "$POINTSAM_DIR" checkout "$PIN_SHA"

python -m pip install -r "$POINTSAM_DIR/requirements.txt"

mkdir -p "$POINTSAM_DIR/pretrain"
SAM_CKPT="$POINTSAM_DIR/pretrain/sam_vit_b_01ec64.pth"
if [[ ! -f "$SAM_CKPT" ]]; then
    wget -c       -O "$SAM_CKPT"       https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
fi

echo
echo "PointSAM pinned at: $PIN_SHA"
echo "PointSAM dir:       $POINTSAM_DIR"
echo "SAM ViT-B:          $SAM_CKPT"
echo
echo "Next: put NWPU images 001.jpg ... 650.jpg in:"
echo "  $POINTSAM_DIR/data/NWPU/Images"
echo "Then run verify_nwpu.py before training."
