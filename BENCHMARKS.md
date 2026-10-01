# 🧪 Benchmarks & Reproducibility

This page turns the repository from a paper index into a reproducible research workspace.

> **Important:** scores copied from different papers are usually **not directly comparable**. Remote-sensing segmentation results can change materially with crop size, overlap, data split, pretraining, augmentation, test-time augmentation (TTA), class mapping, and evaluation scripts.

## 1. Two benchmark layers

We keep two kinds of results separate:

### A. Paper-reported results

Use this table only to **look up what a paper reported**.

| Method | Dataset | Metric | Score | Backbone | Pretraining | Input / crop | Split | TTA | Source |
| :-- | :-- | :-- | --: | :-- | :-- | :-- | :-- | :--: | :-- |
| *Example* | LoveDA | mIoU | — | — | — | — | official | — | paper/table |

Rules:

- Every number must point to the original paper, supplement, or official repository.
- Do not compare rows as an apples-to-apples leaderboard unless the protocol columns match.
- If a value cannot be traced to a primary source, do not add it.

### B. Repository-controlled results

These are the results that can eventually support fair comparison.

| Method | Dataset | Seed(s) | Crop | Epochs | Pretraining | mIoU | Params | Config / commit |
| :-- | :-- | :-- | :-- | --: | :-- | --: | --: | :-- |
| U-Net | TBD | 0/1/2 | TBD | TBD | none | — | — | TBD |
| SegFormer-B0 | LoveDA | 0/1/2 | 512 | 100 | ImageNet-1K (`nvidia/mit-b0`) | — | 3.716M | `loveda_segformer_b0_512` |

A controlled result should record:

1. exact dataset version and split;
2. preprocessing, crop size, overlap and resize policy;
3. optimizer, learning rate, schedule, batch size and epochs;
4. pretrained weights and their source;
5. random seed(s);
6. evaluation script and class mapping;
7. hardware only when it affects the protocol;
8. git commit or config file needed to reproduce the run.

## 2. Recommended first controlled benchmark

For a compact and useful first benchmark, start with one semantic-segmentation dataset and only a few representative model families:

- **CNN:** U-Net
- **Transformer:** SegFormer
- **Foundation model:** SAM/SAM2-derived method only when the prompting protocol is clearly defined

Do not expand to many models until the data pipeline and evaluation script are shared across methods.

## 3. Metric conventions

For semantic segmentation, report at least:

- **mIoU** — mean intersection over union;
- **per-class IoU** — exposes failures hidden by a mean;
- **mF1 / Dice** when commonly used by the dataset or task;
- **OA** only as a secondary metric for strongly imbalanced segmentation datasets.

For instance segmentation, use the dataset's standard AP/AR protocol. For change detection, explicitly state whether metrics are pixel-level, object-level, or semantic-change metrics.

## 4. Reproducibility checklist

Before adding a result, verify:

- [ ] dataset download source is recorded;
- [ ] train/val/test split is explicit;
- [ ] class IDs and ignored labels are documented;
- [ ] preprocessing and tiling are documented;
- [ ] pretrained weights are identified;
- [ ] seed is recorded;
- [ ] evaluation command is recorded;
- [ ] result comes from an exact commit/config;
- [ ] paper-reported and locally reproduced scores are clearly distinguished.

## 5. Suggested experiment record

A future run can be stored in a lightweight structure such as:

```text
experiments/
  loveda/
    unet/
      config.yaml
      results.json
    segformer_b0/
      config.yaml
      results.json
```

Recommended `results.json` fields:

```json
{
  "dataset": "LoveDA",
  "split": "official",
  "method": "SegFormer-B0",
  "seed": 0,
  "miou": null,
  "per_class_iou": {},
  "commit": "<git-sha>",
  "notes": ""
}
```

This keeps future thesis experiments, ablations, and public benchmark results traceable instead of leaving numbers only in a README or spreadsheet.

The first named repository-controlled protocol is documented in **[LOVEDA_SEGFORMER_PROTOCOL.md](LOVEDA_SEGFORMER_PROTOCOL.md)**. Its row should remain scoreless until a full controlled run is completed; debug/preflight numbers must not be copied into the benchmark table.
