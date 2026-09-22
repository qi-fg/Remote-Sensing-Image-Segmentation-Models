# 📦 遥感图像分割常用数据集

> 覆盖语义分割 / 实例分割 / 建筑物·道路提取 / 变化检测。链接以官方页面为准，若失效欢迎 PR 修正。

---

## 🏙️ 语义分割（Semantic Segmentation）

| 数据集 | 规模 | 分辨率 / 特点 | 任务 | 入口 |
| :-- | :-- | :-- | :-- | :-- |
| **ISPRS Potsdam** | 38 图，6 类 | 5 cm GSD，城市正射影像 | 语义分割（RS 分割最常用基准之一） | [🔗](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx) |
| **ISPRS Vaihingen** | 33 图，6 类 | 9 cm GSD，含近红外 | 语义分割 | [🔗](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx) |
| **LoveDA** | 5987 图，7 类 | 0.3 m，城/乡双域，NeurIPS 2021 D&B | 语义分割（域自适应） | [🔗](https://github.com/Junjue-Wang/LoveDA) |
| **iSAID** | 2806 图，15 类 | 高分辨率航拍，CVPR 2019 | 语义 + 实例分割 | [🔗](https://captain-whu.github.io/iSAID/) |
| **DeepGlobe Land Cover** | 1146 图，7 类 | 0.5 m，CVPR 2018 | 地表覆盖分类 | [🔗](http://deepglobe.org/) |
| **GID (Gaofen Image Dataset)** | 150 图，15 类 | 高分二号，大规模 | 语义分割 | [🔗](https://captain-whu.github.io/GID/) |
| **OpenEarthMap** | 5000 图，8 类 | 0.25–0.5 m，全球覆盖 | 语义分割 | 🔗 官网 |

## 🎯 实例分割 / 目标分割（Instance Segmentation）

| 数据集 | 规模 | 特点 | 入口 |
| :-- | :-- | :-- | :-- |
| **SAMRS** | 105,090 图 / 1,668,241 实例 | 用 SAM + 检测数据生成，NeurIPS 2023 | [🔗](https://github.com/ViTAE-Transformer/SAMRS) |
| **iSAID** | 655,451 实例 | 基于 DOTA 像素级标注 | [🔗](https://captain-whu.github.io/iSAID/) |
| **NWPU VHR-10** | 800 图，10 类 | 小规模、经典目标检测/分割集 | [🔗](https://github.com/chaozhong2010/VHR-10_dataset_coco) |
| **DOTA-v2.0** | 11,268 图，18 类 | 定向目标检测（SAMRS 转换源） | [🔗](https://captain-whu.github.io/DOTA/) |

## 🏗️ 建筑物 / 道路提取（Building & Road Extraction）

| 数据集 | 规模 | 特点 | 入口 |
| :-- | :-- | :-- | :-- |
| **WHU Building Dataset** | 8,180 图 | 新西兰 Christchurch，航空影像 | [🔗](http://gpcv.whu.edu.cn/data/building_dataset.html) |
| **Massachusetts Buildings** | 151 图 | 1 m，航拍 | [🔗](https://www.cs.toronto.edu/~vmnih/data/) |
| **Massachusetts Roads** | 1171 图 | 1 m，航拍道路 | [🔗](https://www.cs.toronto.edu/~vmnih/data/) |
| **SpaceNet** | 多城市多期 | 商业卫星，建筑/道路 | [🔗](https://spacenet.ai/) |

## 🔄 变化检测（Change Detection）

| 数据集 | 规模 | 特点 | 入口 |
| :-- | :-- | :-- | :-- |
| **LEVIR-CD** | 637 对，0.5 m | 建筑变化检测基准 | [🔗](https://chen-yongrui.github.io/) |
| **WHU-CD** | 1 对大幅影像 | 建筑物变化 | [🔗](http://gpcv.whu.edu.cn/data/) |
| **CDD (Change Detection Dataset)** | 16,000 对 | 季节变化 | 🔗 官网 |

---

## 🗂️ 高光谱分类数据集（附：主仓库未覆盖，但常一起用）

> 若你后续要做高光谱方向，这些是标配：

| 数据集 | 类别数 | 场景 |
| :-- | :--: | :-- |
| Indian Pines | 16 | 农业区，145×145×200 |
| Pavia University | 9 | 城市，610×340×103 |
| Salinas | 16 | 农业，512×217×204 |
| Houston 2013 / 2018 | 15 / 20 | 城市，GRSS 竞赛 |

---

## ⚠️ 使用提示

- **坐标系 / 配准**：多源数据（光学 + SAR + 高光谱）务必先检查地理配准与重采样。
- **类别不均衡**：RS 场景里道路、小目标占比极低，训练时建议用 **Dice / Focal + 类别权重**。
- **大图切瓦片**：训练前通常要 tile 化 + 重叠裁剪，推理后再拼回。
- **标注规范**：不同数据集的类别定义不一致（如"车"是否含卡车），跨集训练前要统一。

---

*发现链接失效或想补充数据集，欢迎提 [Issue](../../issues) / PR。*
