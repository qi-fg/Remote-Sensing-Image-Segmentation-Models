# SAM-RS Baseline

**SAM**（Segment Anything, Kirillov et al., ICCV 2023）在遥感分割上的最小封装。

## 为什么单独放一个 SAM 基线
SAM 是**提示式（prompt-driven）**基础模型：给点 / 框，返回掩码。遥感影像动辄上万像素，**预处理与坐标变换**（长边缩放到 1024、prompt 同步缩放）才是落地时最容易踩坑的地方。本封装把这条链路显式拆出来，可直接复用。

## 运行

**离线链路检查（不需要权重、不需要装 segment-anything、不需要 GPU）**
```bash
cd baselines/sam_rs
pip install -r requirements.txt        # numpy (+pillow)
python predict.py --demo
```
输出示例：
```
== SAM-RS pipeline demo (no checkpoint required) ==
input image        : (384, 512, 3) uint8
resized (longest=1024) : (768, 1024, 3) (scale=2.000)
point prompt orig  : [(225, 150)]
point prompt scaled: [(450.0, 300.0)]
box prompt orig    : [150, 100, 300, 200]
box prompt scaled  : [300.0, 200.0, 600.0, 400.0]
[info] `segment-anything` not installed. To run real masks: ...
```

**真实推理**
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git torch pillow
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# 框提示
python predict.py --image scene.tif --checkpoint sam_vit_h_4b8939.pth --box 150 100 300 200
# 点提示（x y [x y ...]）
python predict.py --image scene.tif --checkpoint sam_vit_h_4b8939.pth --point 225 150
# 全自动掩码生成
python predict.py --image scene.tif --checkpoint sam_vit_h_4b8939.pth --auto
```
结果保存为 `mask.npy`（`--out` 可改）。

## 参数
| 参数 | 说明 |
| :-- | :-- |
| `--model-type` | `vit_h` / `vit_l` / `vit_b`（显存不够就用 `vit_b`） |
| `--point` / `--box` / `--auto` | 三种提示模式 |
| `--device` | `cpu` / `cuda` |

## 遥感落地建议
- **超大图**：先切瓦片（tile）再喂 SAM，最后拼回；直接整图缩到 1024 会丢小目标。
- **零样本不够用**：RS 的类别与自然图像分布差异大，通常需要 **微调 / 加 adapter / 学 prompt**——见主 README 的 Foundation Models 一节的 `SAMRS` / `RSPrompter` / `HSA-SAM`。
- **点监督**：结合少量点标注做弱监督微调，是降低成本的主流路线（见主 README 弱监督一节）。

## 文件
- `predict.py` — 预处理/坐标变换工具 + `SAMRSPredictor` 封装 + CLI
