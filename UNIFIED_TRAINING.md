# 🧪 Unified Training Workflow

这一层的目的不是让完整训练“瞬间变快”，而是减少**跑很久以后才发现代码、数据或指标有问题**的情况。

目前统一入口支持 **U-Net**、仓库自实现 **SegFormer**，以及用于正式全监督参考的 **standard SegFormer-B0**。

## 两种 SegFormer 的用途

仓库现在故意保留两条 SegFormer 路线：

| CLI | 初始化 | 主要用途 |
| :-- | :-- | :-- |
| `--model segformer` | 随机初始化，仓库自包含实现 | 快速开发、消融、检查统一框架 |
| `--model segformer_b0` | NVIDIA `nvidia/mit-b0` ImageNet-1K 预训练编码器 + 标准 SegFormer 解码头 | **全监督 benchmark / 论文参考** |

标准版首次使用前：

```bash
pip install -r baselines/segformer_standard/requirements.txt
```

首次启用预训练时会下载并缓存 `nvidia/mit-b0`。统一训练器会记录 `pretrained`、`loss`、学习率、Git commit、PyTorch/CUDA/Transformers 版本等实验信息。

默认配置上，`segformer_b0` 使用 ImageNet-1K 预训练、AdamW、学习率 `6e-5`、weight decay `0.01` 和 CE loss；这些设置用于建立一个比随机初始化自实现版更合适的全监督参考。正式论文仍应把数据增强、crop、scheduler、训练长度等协议固定并完整报告。

> 点监督论文里，标准 SegFormer-B0 应标为 **Full Supervision / no prompt**。它不是 ReSAM/PointSAM 表中依赖 1/2/3 个点提示的 `Supervised` 方法，不能直接改名替换文献原结果。

---

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

## LoveDA：直接读取官方目录

LoveDA 不需要你手工整理成通用 `train/images` 结构。统一入口可以直接读取官方解压后的目录：

```text
LoveDA/
├── Train/
│   ├── Urban/
│   │   ├── images_png/
│   │   └── masks_png/
│   └── Rural/
│       ├── images_png/
│       └── masks_png/
└── Val/
    ├── Urban/
    │   ├── images_png/
    │   └── masks_png/
    └── Rural/
        ├── images_png/
        └── masks_png/
```

官方原始 mask 的标签是：

| 原始值 | 训练值 | 类别 |
| :--: | :--: | :-- |
| 0 | -1 | no-data / ignore |
| 1 | 0 | background |
| 2 | 1 | building |
| 3 | 2 | road |
| 4 | 3 | water |
| 5 | 4 | barren |
| 6 | 5 | forest |
| 7 | 6 | agricultural |

adapter 会自动完成这个映射；no-data 不参与 CE、Dice、mIoU、OA 或混淆矩阵。

先跑 debug：

```bash
python train.py \
  --dataset loveda \
  --model segformer_b0 \
  --mode debug \
  --data /data/LoveDA \
  --device cuda
```

LoveDA 的类别数会自动设为 7，不需要再写 `--num-classes 7`。默认同时读取 Urban 和 Rural；也可以只检查一个域：

```bash
python train.py --dataset loveda --model unet --mode debug \
  --data /data/LoveDA --domain urban --device cuda
```

> `debug` 默认 resize 到 256×256，只用于快速排错，不应该把它的精度当正式 benchmark 结果。

LoveDA 官方 Train/Val 可以用于开发；Test 标签不公开，正式 Test 分数需要通过挑战赛评测。因此当前训练/评测 adapter 只支持 `train` / `val`。

---

## 3. Full：确认无误后再正式跑

```bash
python train.py \
  --dataset loveda \
  --model segformer_b0 \
  --mode full \
  --data /data/LoveDA \
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

当前 LoveDA adapter 解决的是**官方目录读取、标签映射、Urban/Rural 选择和 ignore 像素处理**。它没有声称复现某篇论文的完整训练 recipe。

正式对比时还需要明确并固定：crop/resize、数据增强、预训练权重、学习率策略、batch size、训练轮数、TTA 等。尤其不要把 `debug` 的 256×256 resize 结果与论文表格直接比较。

LoveDA 数据用于学术研究时还应遵守其数据许可与 Google Earth 相关使用条款。
