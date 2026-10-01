# LoveDA SegFormer-B0 512 Protocol

This document defines the repository-controlled full-supervision reference used for LoveDA experiments.

> This is **not** claimed to be a bit-for-bit reproduction of an official LoveDA or NVLabs SegFormer result. It deliberately combines a LoveDA 512×512 semantic-segmentation data pipeline with the standard SegFormer optimization recipe, then records every resolved setting in `config.json`.

## Why this protocol exists

The earlier `debug` path resizes a small subset to 256×256 and is only for pipeline checks. It is useful for finding data/label bugs, but its scores must not be used as a paper benchmark.

The named recipe:

```text
loveda_segformer_b0_512
```

is the first controlled protocol intended for repeatable full-supervision experiments.

## Model

- Model: standard SegFormer-B0 implementation from Hugging Face Transformers.
- Encoder initialization: `nvidia/mit-b0` ImageNet-1K pretrained weights.
- Classes: 7 LoveDA classes.
- Loss: cross entropy.
- Input normalization: ImageNet RGB mean/std inside the model wrapper.

## LoveDA training transform

The recipe follows the structure used by MMSegmentation-style LoveDA 512×512 configs:

1. random keep-ratio resize with ratio sampled from `[0.5, 2.0]`;
2. random 512×512 crop;
3. reject highly single-class crops when possible with `cat_max_ratio=0.75`;
4. horizontal flip with probability 0.5;
5. photometric distortion;
6. no-data pixels remain ignored.

For a square 1024×1024 LoveDA tile, the resize stage samples a square side from about 256 to 1024 before cropping/padding to 512×512.

Validation uses the original image resolution, no random transform, and no TTA.

## Optimizer and schedule

The SegFormer-oriented optimizer profile uses:

| Setting | Value |
| :-- | :-- |
| optimizer | AdamW |
| backbone LR | `6e-5` |
| decode-head LR multiplier | `10×` |
| betas | `(0.9, 0.999)` |
| weight decay | `0.01` |
| normalization weight decay | `0` |
| scheduler | polynomial |
| poly power | `1.0` |
| warmup | linear, 1500 iterations |
| warmup ratio | `1e-6` |
| loss | CE |
| default train batch | 8 |
| default val batch | 1 |
| default epochs | 100 |
| validation interval | every 5 epochs |
| AMP | on for CUDA |

The 100-epoch training length is a **repository-controlled choice**. The original SegFormer configs are iteration-based and commonly use substantially longer schedules, while MMSegmentation LoveDA examples also use iteration-based schedules. Do not describe this 100-epoch run as an exact reproduction of those schedules.

## Sources behind the recipe

- NVLabs SegFormer official repository:
  <https://github.com/NVlabs/SegFormer>
- OpenMMLab MMSegmentation LoveDA dataset/config family:
  <https://github.com/open-mmlab/mmsegmentation>
- LoveDA official repository:
  <https://github.com/Junjue-Wang/LoveDA>

The important distinction is provenance: data handling, architecture/pretraining, and optimizer conventions come from established sources, while this repository fixes them into one controlled protocol for its own comparisons.

## Preflight run

Before spending hours on the full run, verify the exact 512 recipe on a small subset:

```bash
python train.py \
  --dataset loveda \
  --model segformer_b0 \
  --mode debug \
  --recipe loveda_segformer_b0_512 \
  --data /private/SEG/LoveDA \
  --epochs 2 \
  --debug-train-samples 64 \
  --debug-val-samples 16 \
  --batch-size 4 \
  --device cuda \
  --output runs/loveda-segformer_b0-recipe-preflight
```

Unlike the legacy debug path, this named recipe does **not** force 256×256 resize; it exercises the real random 512 crop pipeline.

## Full run

```bash
python train.py \
  --dataset loveda \
  --model segformer_b0 \
  --mode full \
  --recipe loveda_segformer_b0_512 \
  --data /private/SEG/LoveDA \
  --device cuda \
  --seed 0 \
  --output runs/loveda-segformer_b0-full-512-seed0
```

For a paper result, repeat with multiple fixed seeds (for example 0, 1, and 2) and report mean ± standard deviation rather than selecting a favorable seed.

## Resume

```bash
python train.py \
  --dataset loveda \
  --model segformer_b0 \
  --mode full \
  --recipe loveda_segformer_b0_512 \
  --data /private/SEG/LoveDA \
  --device cuda \
  --seed 0 \
  --output runs/loveda-segformer_b0-full-512-seed0 \
  --resume runs/loveda-segformer_b0-full-512-seed0/last.pt
```

Checkpoints preserve model, optimizer, scheduler, AMP scaler, RNG state, best metrics, and resolved configuration.

## Evaluation

```bash
python evaluate.py \
  --checkpoint runs/loveda-segformer_b0-full-512-seed0/best.pt \
  --data /private/SEG/LoveDA \
  --dataset loveda \
  --split val \
  --batch-size 1 \
  --device cuda
```

The training `results.json` now stores both `best` and `final` metrics, so per-class IoU is tied to the correct best-validation checkpoint rather than accidentally mixing the best mIoU with final-epoch class scores.

## What this row means in a point-supervised paper

Use a label such as:

```text
SegFormer-B0 (Full Sup., no prompt)
```

It is a full-pixel-supervision reference. It does not consume 1/2/3 point prompts, so it should not be presented as though it were a point-budget method or as a renamed literature `Supervised` row.
