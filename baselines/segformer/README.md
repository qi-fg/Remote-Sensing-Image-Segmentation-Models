# SegFormer Baseline

**SegFormer**（Xie et al., NeurIPS 2021）的紧凑实现：分层 Transformer 编码器（MiT）+ 轻量全 MLP 解码头。

## 特点
- **纯 PyTorch 实现**，`--demo` **不需要下载预训练权重**、不需要 `timm`/`mmseg`，离线几秒跑通。
- 结构对齐原论文：**Overlap Patch Embedding** → `[Efficient Self-Attention + Mix-FFN] × N` → all-MLP decoder。
- Efficient Self-Attention 带 **sr_ratio 空间降采样**（在 K/V 上降采样，复杂度可控）。
- 支持任意输入波段数（多光谱/高光谱）。

## 运行

**离线冒烟测试（无需数据集、无需 GPU）**
```bash
cd baselines/segformer
pip install -r requirements.txt        # 只需 torch
python train.py --demo
```
示例输出（数值随随机种子/机器不同，仅示意）：
```
device=cpu  variant=tiny  params=0.31M  train=64  val=16
epoch   1/20  loss=1.1544  mIoU=0.3810
epoch  10/20  loss=0.2011  mIoU=0.9020
epoch  20/20  loss=0.0912  mIoU=0.9540
done in 21.4s  final mIoU=0.9540
```

**接近论文的 B0 通道宽度**
```bash
python train.py --demo --variant b0
```

**真实数据集**
```bash
pip install pillow numpy
python train.py --data /path/to/dataset --epochs 50 --variant b0
```

## 参数
| 参数 | 说明 |
| :-- | :-- |
| `--variant` | `tiny`（CPU 快速 demo）/ `b0`（论文 B0 通道宽度） |
| `--in-channels` | 输入波段数 |
| `--num-classes` | 类别数 |
| `--epochs` `--batch-size` `--lr` | 训练超参 |

## 备注
- `tiny` 配置：`embed_dims=(16,32,64,128), depths=(1,1,1,1)`，为 CPU demo 而生。
- 想接官方 ImageNet 预训练权重做迁移学习，建议直接用 [`NVlabs/SegFormer`](https://github.com/NVlabs/SegFormer) 或 ```mmsegmentation``；本实现追求"零依赖可跑、结构透明"。

## 文件
- `model.py` — SegFormer 网络定义（`SegFormer.tiny()` / 自定义配置）
- `train.py` — 训练/评估 + 合成数据 demo
