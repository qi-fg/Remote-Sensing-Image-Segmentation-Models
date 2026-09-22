# U-Net Baseline

经典 **U-Net**（Ronneberger et al., MICCAI 2015）的干净实现，作为遥感语义分割的通用基线。

## 为什么用它当基线
编码-解码 + 跳跃连接结构简单、稳定、收敛快，是几乎所有 RS 分割论文都要对比的 **CNN 基线**。支持任意波段数（RGB=3、多光谱/高光谱=N）。

## 运行

**离线冒烟测试（无需数据集、无需 GPU，几秒跑完）**
```bash
cd baselines/unet
pip install -r requirements.txt        # 只需 torch
python train.py --demo
```
示例输出（数值随随机种子/机器不同，仅示意）：
```
device=cpu  params=1.9M  train=64  val=16
epoch   1/20  loss=1.0312  mIoU=0.4521
epoch  10/20  loss=0.1287  mIoU=0.9314
epoch  20/20  loss=0.0455  mIoU=0.9783
done in 6.2s  final mIoU=0.9783
```

**真实数据集**
```bash
pip install pillow numpy
python train.py --data /path/to/dataset --epochs 50 --batch-size 8
```
数据目录结构：
```
dataset/
  images/   *.png|*.tif     # H x W x C
  masks/    *.png           # H x W，单通道，整型类别 id
```
images 与 masks 按文件名（去掉扩展名）配对。

## 常用参数
| 参数 | 说明 |
| :-- | :-- |
| `--in-channels` | 输入波段数（RGB=3，多光谱按实际改） |
| `--num-classes` | 类别数（二分类=2） |
| `--base` | 基础通道宽度（默认 32；demo 自动降到 8 求快） |
| `--epochs` `--batch-size` `--lr` | 训练超参 |

## 损失
`CrossEntropy + Dice` 联合损失（`train.py` 中 `dice_loss`）——类别不均衡的 RS 场景下比纯 CE 更稳。

## 文件
- `model.py` — U-Net 网络定义（可直接 `from model import UNet`）
- `train.py` — 训练/评估 + 合成数据 demo
