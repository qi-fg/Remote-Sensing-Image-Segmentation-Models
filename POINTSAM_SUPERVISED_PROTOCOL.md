# PointSAM Supervised NWPU Reproduction Protocol

This protocol reproduces the **full-mask-supervised SAM reference used by PointSAM and cited by ReSAM** on NWPU VHR-10.

It is intentionally separate from the LoveDA SegFormer baseline:

- LoveDA SegFormer-B0: prompt-free dense semantic segmentation.
- PointSAM supervised: full instance-mask supervision during adaptation, but evaluation still uses 1/2/3 point prompts.

That distinction is why PointSAM's supervised reference can appear in the same 1-/2-/3-point table as ReSAM.

## Provenance

Pinned upstream PointSAM:

- repository: `Lans1ng/PointSAM`
- commit: `a4b9253bca3db4932dbaea33d2bf1fee66d92e28`
- license: MIT
- official training entry: `train_supervise.py`
- official NWPU launcher: `scripts/train_nwpu_supervise.sh`

ReSAM uses the same NWPU file paths in its configuration:

```text
data/NWPU/Images
data/NWPU/Annotations/NWPU_instances_train.json
data/NWPU/Annotations/NWPU_instances_val.json
```

The PointSAM annotation files contain an exact, disjoint split:

| Split | Images | Instances |
| :-- | --: | --: |
| train | 520 | 3178 |
| val | 130 | 743 |
| union | 650 | 3921 |

There is no train/val image overlap. The image filenames cover `001.jpg` through `650.jpg`.

## What “Supervised” means here

The upstream `train_supervise.py` does **not** turn SAM into a prompt-free segmentation network.

For each training sample it:

1. derives 1, 2, or 3 point prompts from the ground-truth instance masks;
2. feeds the image and point prompts through SAM;
3. supervises predicted masks against the **full ground-truth instance masks**;
4. optimizes focal loss, Dice loss, and IoU-regression loss.

The SAM image encoder, prompt encoder, and mask decoder base parameters are frozen in the provided configuration. PointSAM inserts rank-4 LoRA adapters into SAM ViT-B image-encoder attention blocks starting from block 6, and those adapters are trained.

So the most precise label for a paper table is:

```text
SAM ViT-B + LoRA (Full Mask Supervision)
```

or, when matching ReSAM's table wording:

```text
Supervised [PointSAM]
```

## Upstream NWPU training settings

The pinned PointSAM NWPU configuration specifies:

| Setting | Value |
| :-- | :-- |
| backbone | SAM ViT-B |
| SAM checkpoint | `sam_vit_b_01ec64.pth` |
| input transform | resize-and-pad to 1024 |
| prompt | point |
| point budgets | 1 / 2 / 3 |
| batch size | 1 |
| validation batch | 1 |
| epochs | 10 |
| optimizer | Adam |
| learning rate | `5e-4` |
| weight decay | `1e-4` |
| warmup | 250 iterations |
| LR decay steps | 2000 / 4000 iterations |
| LoRA rank | 4 |
| first LoRA block | 6 |
| seed | 1337 + distributed rank |
| model selection | best validation IoU |

The training objective is:

```text
20 × Focal + Dice + IoU-regression
```

These values come from the pinned upstream code and config; this repository's wrapper does not silently change them.

## RTX 5090 environment

PointSAM's README contains an older PyTorch 2.3.1/CUDA 11.8 example. That build cannot target RTX 5090 / Blackwell.

ReSAM's official repository, which derives from the same experiment family, uses:

```bash
conda create -n pointsam python=3.10 -y
conda activate pointsam

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

This is the environment used by this reproduction adapter because CUDA 12.8+ PyTorch builds support Blackwell.

Then:

```bash
bash baselines/pointsam_supervised/setup_upstream.sh
```

## Dataset preparation

The pinned PointSAM repository already includes the exact NWPU COCO annotation JSON files.

Download the NWPU VHR-10 positive images and place or symlink all 650 files under:

```text
/private/SEG/PointSAM/data/NWPU/Images/
```

Before training, run:

```bash
python baselines/pointsam_supervised/verify_nwpu.py \
  --pointsam-dir /private/SEG/PointSAM
```

Do not create a new random 80/20 split. Reusing the provided 520/130 split is required for comparison with PointSAM/ReSAM.

## Training

Run all three supervised point budgets:

```bash
POINTSAM_DIR=/private/SEG/PointSAM \
bash baselines/pointsam_supervised/run_nwpu_supervised.sh
```

For a first GPU sanity check, run only 1-point:

```bash
POINTS="1" POINTSAM_DIR=/private/SEG/PointSAM \
bash baselines/pointsam_supervised/run_nwpu_supervised.sh
```

This wrapper calls the same upstream training entry and arguments as PointSAM's official NWPU supervised script.

## Paper-reported target

ReSAM's CVPR 2026 Table 1 cites PointSAM for the SAM-based supervised reference on NWPU VHR-10:

| Setting | IoU | F1 |
| :--: | --: | --: |
| 1 point | 78.73 | 86.74 |
| 2 points | 80.88 | 88.58 |
| 3 points | 81.12 | 88.79 |

These values are **paper-reported targets**. They must remain separate from locally reproduced values.

After local runs finish:

```bash
python baselines/pointsam_supervised/collect_results.py \
  --pointsam-dir /private/SEG/PointSAM
```

The collector excludes epoch 0 (direct SAM), selects the trained epoch with the highest validation IoU for each point budget, and records F1 from the same epoch.

## Reporting rule

For the eventual ReSAM-style comparison table, use two layers:

```text
Paper-reported:
Supervised [PointSAM]   78.73/86.74   80.88/88.58   81.12/88.79

Repository-controlled:
Supervised [PointSAM]   <our run>     <our run>     <our run>
```

Do not replace the literature numbers with local results or vice versa. If the reproduced numbers differ materially, investigate environment, image set, upstream commit, prompt sampling, and checkpoint-selection behavior before drawing conclusions.
