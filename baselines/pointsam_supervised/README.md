# PointSAM supervised reproduction adapter

This folder is a **thin reproduction adapter** around the upstream PointSAM repository. It does not vendor or rewrite PointSAM.

The goal is to reproduce the fully-mask-supervised SAM-ViT-B reference that ReSAM cites as `Supervised [16]`, starting with NWPU VHR-10 and the same PointSAM train/validation split.

## Pinned upstream

- Repository: `Lans1ng/PointSAM`
- Commit: `a4b9253bca3db4932dbaea33d2bf1fee66d92e28`
- Backbone: SAM ViT-B
- Training entry: `train_supervise.py`
- NWPU launcher mirrored by this adapter: `scripts/train_nwpu_supervise.sh`

The pin is intentional: later upstream changes should not silently alter a paper reproduction.

## Environment for RTX 5090

PointSAM's older README suggests PyTorch 2.3.1/CUDA 11.8, which is too old for Blackwell/RTX 5090. ReSAM's official setup uses Python 3.10 with PyTorch 2.7.0 + CUDA 12.8, so use that environment for the shared PointSAM/ReSAM experiment line:

```bash
conda create -n pointsam python=3.10 -y
conda activate pointsam

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

Then, from this repository root:

```bash
bash baselines/pointsam_supervised/setup_upstream.sh
```

The setup script clones the pinned PointSAM commit, installs its Python requirements, and downloads the official SAM ViT-B checkpoint with `wget`.

## NWPU images

PointSAM already ships the exact COCO train/val JSON annotations. You only need the 650 positive NWPU VHR-10 images named `001.jpg` ... `650.jpg`.

Place or symlink them at:

```text
/private/SEG/PointSAM/data/NWPU/Images/
```

The pinned split contains:

- train: 520 images / 3178 instances
- validation: 130 images / 743 instances
- overlap: 0 images
- union: 650 images

Verify before training:

```bash
python baselines/pointsam_supervised/verify_nwpu.py \
  --pointsam-dir /private/SEG/PointSAM
```

## Run the full-supervision reference

The official PointSAM supervised launcher trains three separate point-prompt settings. This wrapper preserves the same config and calls the same upstream `train_supervise.py`:

```bash
POINTSAM_DIR=/private/SEG/PointSAM \
bash baselines/pointsam_supervised/run_nwpu_supervised.sh
```

To run only one setting first:

```bash
POINTS="1" POINTSAM_DIR=/private/SEG/PointSAM \
bash baselines/pointsam_supervised/run_nwpu_supervised.sh
```

Expected output directories are:

```text
PointSAM/work_dir/nwpu/supervise/
├── point_1/
├── point_2/
└── point_3/
```

Each run writes `metrics.csv` and saves its best checkpoint according to validation IoU.

## Collect results

```bash
python baselines/pointsam_supervised/collect_results.py \
  --pointsam-dir /private/SEG/PointSAM
```

The collector selects the trained row with the highest validation IoU for each point budget and keeps IoU/F1 from the same epoch.

The ReSAM paper reports the following SAM-based supervised reference on NWPU VHR-10:

| Prompt budget | IoU | F1 |
| :--: | --: | --: |
| 1 point | 78.73 | 86.74 |
| 2 points | 80.88 | 88.58 |
| 3 points | 81.12 | 88.79 |

These are **paper-reported targets**, not repository-controlled results. Do not copy them into a reproduced-results table until the local runs finish.

See [POINTSAM_SUPERVISED_PROTOCOL.md](../../POINTSAM_SUPERVISED_PROTOCOL.md) for provenance and reporting rules.
