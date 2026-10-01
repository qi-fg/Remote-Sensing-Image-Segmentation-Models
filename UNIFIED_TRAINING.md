# 🧪 Unified Training Workflow

这一层的目的不是让完整训练“瞬间变快”，而是减少**跑很久以后才发现代码、数据或指标有问题**的情况。

目前统一入口支持 **U-Net** 和 **SegFormer**。

## 三种模式

| 模式 | 数据 | 默认训练 | 用途 | 建议运行位置 |
| :-- | :-- | :-- | :-- | :-- |
| `smoke` | 合成小数据 | 1 epoch | 检查代码能否完整跑通 | GitHub CI / 本地 CPU |
| `debug` | 少量真实数据 | 3 epochs，最多 128/64 张 train/val | 检查真实数据、标签、loss、metric | 服务器 GPU |
| `full` | 完整真实数据 | 100 epochs（可改） | 正式科研实验 | 服务器 GPU |

## 1. Smoke：不消耗真实数据和 5090 时间

```bash
python train.py --model unet --mode smoke
python train.py --model segformer --mode smoke
```

GitHub Actions 只运行这一层，不会下载或完整训练 LoveDA。

## 2. Debug：先用少量真实数据排错

当前通用数据适配器要求：

```text
dataset/
├── train/
│   ├── images/
│   └── masks/
└── val/
    ├── images/
    └── masks/
```

图像和 mask 按**同名文件 stem** 配对，例如 `0001.png` 对 `0001.png`。mask 必须是单通道整数类别 ID。

示例：

```bash
python train.py \
  --model segformer \
  --mode debug \
  --data /data/my_dataset \
  --num-classes 7 \
  --device cuda
```

默认只取前 128 张训练图和 64 张验证图，并 resize 到 256×256，目的是尽快暴露数据读取、类别 ID、显存、loss 和指标问题，而不是追求最终精度。

也可以调整：

```bash
python train.py --model unet --mode debug --data /data/my_dataset \
  --num-classes 7 --debug-train-samples 256 --debug-val-samples 128
```

## 3. Full：确认无误后再正式跑

```bash
python train.py \
  --model segformer \
  --mode full \
  --data /data/my_dataset \
  --num-classes 7 \
  --epochs 100 \
  --batch-size 8 \
  --amp \
  --device cuda \
  --output runs/loveda-segformer-b0
```

`full` 默认不强制 resize，避免无意中改变正式实验协议。正式 benchmark 应明确记录 crop/resize、split、pretraining、augmentation、TTA 等设置。

## 自动保存

每次运行会在输出目录保存：

```text
runs/...
├── config.json
├── best.pt
├── last.pt
└── results.json
```

`results.json` 包含：

- mIoU
- OA
- per-class IoU
- confusion matrix

这样以后论文实验不需要靠手工复制终端数字。

## 断点续训

服务器训练中断后：

```bash
python train.py ... --resume runs/loveda-segformer-b0/last.pt
```

## 单独评测 checkpoint

```bash
python evaluate.py \
  --checkpoint runs/loveda-segformer-b0/best.pt \
  --data /data/my_dataset \
  --split val
```

## 当前边界

这一版先统一**训练/评测机制**，还没有把 LoveDA 的原始官方目录结构硬编码进来。下一步会单独增加 LoveDA adapter，并明确官方 train/val/test 与类别映射；这样不会为了“方便”偷偷改变实验协议。
