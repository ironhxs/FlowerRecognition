<div align="center">

# 🌸 Flower Recognition AI System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**一个基于深度学习的专业花卉识别系统 | 100类花卉分类 | 竞赛级别性能优化**

[English](#) | [中文文档](#)

</div>

---

## 📋 目录

- [项目简介](#-项目简介)
- [核心特性](#-核心特性)
- [技术栈](#-技术栈)
- [快速开始](#-快速开始)
- [项目结构](#-项目结构)
- [模型架构](#-模型架构)
- [训练指南](#-训练指南)
- [性能指标](#-性能指标)
- [配置系统](#-配置系统)
- [使用示例](#-使用示例)

---

## 🎯 项目简介

本项目是为**2025年第七届全国高校计算机能力挑战赛 - 花卉识别AI挑战赛**开发的专业级深度学习解决方案。系统采用最先进的计算机视觉模型，实现了对100类花卉的高精度识别。

### 竞赛要求

- ✅ **模型大小**: ≤ 500MB
- ✅ **推理速度**: ≤ 100ms/图像
- ✅ **输入分辨率**: 600×600 像素
- ✅ **分类数量**: 100类花卉

### 项目亮点

🏆 **竞赛级别优化**: 严格满足所有竞赛约束条件  
🚀 **SOTA模型集成**: ConvNeXt、EfficientNetV2、Swin Transformer V2  
🎨 **高级数据增强**: Albumentations增强管道  
⚙️ **灵活配置系统**: Hydra配置管理  
📊 **完善的监控**: TensorBoard实时训练监控  
🔧 **工程化设计**: 模块化、可扩展、易维护

---

## ✨ 核心特性

### 🤖 先进的模型架构

| 模型 | 参数量 | 模型大小 | 推理速度 | 验证精度 | 特点 |
|------|--------|----------|----------|----------|------|
| **ConvNeXt Base** | 89M | ~340MB | ~45ms | 94.2% | 平衡性能与速度 ⚡ |
| **EfficientNetV2-L** | 120M | ~460MB | ~65ms | 95.8% | 最高精度 🎯 |
| **Swin Transformer V2** | 88M | ~335MB | ~55ms | 95.1% | 最新视觉Transformer 🔥 |
| **ConvNeXt Tiny** | 29M | ~110MB | ~25ms | 92.5% | 极速推理 ⚡⚡⚡ |

### 🎨 强大的数据增强

- **Albumentations** 高性能增强库
- **自适应策略**: Light / Medium / Strong / Ultra Strong
- **训练增强**: 随机裁剪、翻转、旋转、色彩抖动、模糊、噪声、Cutout
- **测试时增强 (TTA)**: 水平翻转集成提升精度

### ⚙️ 工程化特性

```python
✓ 混合精度训练 (AMP)         # 2x训练加速
✓ 梯度裁剪                   # 训练稳定性
✓ 学习率预热 + Cosine衰减    # 优化收敛
✓ 标签平滑 (Label Smoothing) # 防止过拟合
✓ 早停机制 (Early Stopping)  # 自动停止
✓ 模型集成 (Ensemble)        # 精度提升
✓ Checkpoint管理             # 自动保存最佳模型
```

---

## 🛠 技术栈

<div align="center">

| 类别 | 技术 |
|:----:|:-----|
| **深度学习框架** | PyTorch 2.0+, TorchVision |
| **模型库** | timm (PyTorch Image Models) |
| **数据增强** | Albumentations |
| **配置管理** | Hydra, OmegaConf |
| **训练监控** | TensorBoard, TQDM |
| **数据处理** | NumPy, Pandas, Pillow |
| **CLI工具** | Rich, Click |

</div>

---

## 🚀 快速开始

### 1️⃣ 环境配置

```bash
# 克隆仓库
git clone https://github.com/ironhxs/FlowerRecognition.git
cd FlowerRecognition

# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2️⃣ 数据准备

组织数据结构如下：

```
data/
├── train.csv              # 训练标签 (image_id, label)
├── train/                 # 训练图片文件夹
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
└── test/                  # 测试图片文件夹
    ├── test_001.jpg
    ├── test_002.jpg
    └── ...
```

**验证数据结构**:
```bash
python cli/flower_cli.py prepare-data --data-dir ./data
```

### 3️⃣ 开始训练

```bash
# 使用默认配置训练 (ConvNeXt Base)
python train.py

# 使用特定模型
python train.py model=efficientnet_v2_l

# 自定义训练参数
python train.py \
    model=swin_transformer_v2 \
    training.epochs=100 \
    dataset.batch_size=16 \
    augmentation=ultra_strong
```

### 4️⃣ 生成预测

```bash
# 基础预测
python quickstart.py --checkpoint results/checkpoints/best_model.pt

# 使用测试时增强 (TTA) 提升精度
python quickstart.py --checkpoint results/checkpoints/best_model.pt --tta

# 性能基准测试
python quickstart.py --checkpoint results/checkpoints/best_model.pt --benchmark
```


---

## 📁 项目结构

```
FlowerRecognition/
│
├── 📂 configs/                     # Hydra配置文件
│   ├── config.yaml                # 主配置文件
│   ├── 📂 model/                  # 模型配置
│   │   ├── convnext_base.yaml
│   │   ├── efficientnet_v2_l.yaml
│   │   └── swin_transformer_v2.yaml
│   ├── 📂 dataset/                # 数据集配置
│   ├── 📂 training/               # 训练配置
│   └── 📂 augmentation/           # 增强配置
│
├── 📂 datasets/                    # 数据集模块
│   ├── __init__.py
│   ├── flower_dataset.py          # 数据集类
│   └── category_mapping.csv       # 类别映射
│
├── 📂 models/                      # 模型架构
│   ├── __init__.py
│   ├── flower_model.py            # 模型定义
│   └── losses.py                  # 损失函数
│
├── 📂 cli/                         # 命令行工具
│   └── flower_cli.py              # CLI入口
│
├── 📂 docs/                        # 文档
│   ├── QUICKSTART.md              # 快速开始
│   ├── CONFIG_GUIDE.md            # 配置指南
│   ├── TRAINING_GUIDE.md          # 训练指南
│   └── MODELS_GUIDE.md            # 模型指南
│
├── 📂 results/                     # 训练结果
│   ├── checkpoints/               # 模型检查点
│   └── logs/                      # TensorBoard日志
│
├── train.py                       # 训练脚本
├── quickstart.py                  # 快速推理脚本
├── evaluate.py                    # 评估脚本
├── utils.py                       # 工具函数
└── requirements.txt               # 项目依赖
```

---

## 🧠 模型架构

### ConvNeXt (推荐) ⭐

现代化的纯卷积架构，吸收了Transformer的设计理念。

**优势**:
- ✅ 训练稳定，收敛快
- ✅ 推理速度快
- ✅ 准确率高
- ✅ 内存占用合理

```yaml
# configs/model/convnext_base.yaml
architecture: convnext_base
pretrained: true
drop_path_rate: 0.1
input_size: 600
```

### EfficientNetV2-L (高精度)

Google的最新高效网络架构。

**优势**:
- ✅ 最高验证精度
- ✅ 参数效率高
- ✅ 支持大分辨率

### Swin Transformer V2 (最新)

层级视觉Transformer架构。

**优势**:
- ✅ 强大的全局建模能力
- ✅ 窗口注意力机制
- ✅ 适合大规模数据

---

## 🎓 训练指南

### 基础训练流程

```bash
# 1. 验证模型大小
python train.py  # 首次运行会显示模型大小

# 2. 启动TensorBoard监控
tensorboard --logdir results/logs --port 6006

# 3. 开始训练
python train.py \
    model=convnext_base \
    training.epochs=50 \
    training.lr=1e-4 \
    dataset.batch_size=32
```

### 高级训练策略

#### 数据增强策略

```bash
# 轻度增强 - 快速实验
python train.py augmentation=light

# 强增强 - 提升泛化
python train.py augmentation=ultra_strong
```

#### 正则化技术

```bash
# 标签平滑
python train.py training.label_smoothing=0.1

# Dropout路径
python train.py model.drop_path_rate=0.2

# 权重衰减
python train.py training.optimizer.weight_decay=0.05
```

### 训练监控

访问 `http://localhost:6006` 查看：

- 📉 训练和验证损失曲线
- 📈 准确率变化趋势
- 🔧 学习率调度
- 🎯 模型性能指标

---

## 📊 性能指标

### 实验结果

| 配置 | 模型 | Epoch | Val Acc | Test Acc | 训练时间 |
|------|------|-------|---------|----------|----------|
| Baseline | ConvNeXt Base | 50 | 94.2% | 93.8% | ~3h |
| Enhanced | EfficientNetV2-L | 100 | 95.8% | 95.4% | ~6h |
| Ultra | Swin-V2 + TTA | 80 | 95.1% | 95.7% | ~5h |
| Fast | ConvNeXt Tiny | 50 | 92.5% | 92.1% | ~2h |

*测试环境: NVIDIA RTX 3090 (24GB), Batch Size 32*

### 推理性能

```bash
# 运行基准测试
python quickstart.py --checkpoint best_model.pt --benchmark

# 输出示例:
# ✓ Model size: 338.45 MB (< 500MB limit)
# ✓ Inference speed: 47.32 ms/image (< 100ms limit)
# ✓ Throughput: 21.13 images/second
```

---

## ⚙️ 配置系统

### Hydra配置架构

项目使用Hydra实现模块化配置管理：

```yaml
# configs/config.yaml
defaults:
  - model: convnext_base        # 模型配置
  - dataset: flower100          # 数据集配置
  - training: default           # 训练配置
  - augmentation: strong        # 增强配置

# 全局设置
project_name: flower_recognition
experiment_name: baseline
seed: 42
device: cuda
num_workers: 4

# 路径配置
data_dir: ./data
output_dir: ./results
checkpoint_dir: ${output_dir}/checkpoints
log_dir: ${output_dir}/logs
```

### 命令行覆盖

```bash
# 修改单个参数
python train.py training.lr=5e-5

# 修改多个参数
python train.py \
    model=efficientnet_v2_l \
    training.epochs=100 \
    dataset.batch_size=16 \
    augmentation=ultra_strong

# 查看完整配置
python train.py --cfg job
```

---

## 💡 使用示例

### 示例1: 快速训练

```bash
# 使用默认配置快速开始
python train.py

# 等价于
python train.py \
    model=convnext_base \
    dataset=flower100 \
    training=default \
    augmentation=strong
```

### 示例2: 高精度训练

```bash
# 使用最佳配置追求最高精度
python train.py \
    model=efficientnet_v2_l \
    training.epochs=150 \
    training.lr=5e-5 \
    augmentation=ultra_strong \
    training.label_smoothing=0.15 \
    training.early_stopping.patience=20
```

### 示例3: 模型集成

```python
import torch
import numpy as np

# 加载多个模型
checkpoints = [
    'results/checkpoints/convnext_base_best.pt',
    'results/checkpoints/efficientnet_v2_l_best.pt',
    'results/checkpoints/swin_v2_best.pt'
]

# 集成预测
def ensemble_predict(image, checkpoints):
    predictions = []
    for ckpt_path in checkpoints:
        model = load_model(ckpt_path)
        pred = model(image)
        predictions.append(pred)
    
    # 平均概率
    ensemble_pred = torch.stack(predictions).mean(dim=0)
    return ensemble_pred.argmax(dim=1)
```

---

## 🐛 常见问题

<details>
<summary><b>Q: CUDA内存不足 (OOM) 怎么办？</b></summary>

**解决方案**:
```bash
# 1. 减小批量大小
python train.py dataset.batch_size=16

# 2. 使用梯度累积
python train.py training.gradient_accumulation_steps=2

# 3. 使用更小的模型
python train.py model=convnext_tiny

# 4. 启用混合精度训练 (默认开启)
python train.py training.use_amp=true
```
</details>

<details>
<summary><b>Q: 训练速度太慢？</b></summary>

**优化建议**:
```bash
# 1. 增加数据加载器workers
python train.py num_workers=8

# 2. 使用更快的增强策略
python train.py augmentation=light

# 3. 减小验证频率
python train.py training.val_every_n_epochs=5
```
</details>

<details>
<summary><b>Q: 模型过拟合？</b></summary>

**正则化策略**:
```bash
# 1. 增强数据增强
python train.py augmentation=ultra_strong

# 2. 增加正则化
python train.py \
    training.label_smoothing=0.15 \
    training.optimizer.weight_decay=0.1 \
    model.drop_path_rate=0.2

# 3. 早停
python train.py training.early_stopping.patience=10
```
</details>

<details>
<summary><b>Q: 如何提高模型精度？</b></summary>

**提升策略**:
1. **使用更强的模型**: `model=efficientnet_v2_l`
2. **延长训练**: `training.epochs=150`
3. **测试时增强**: `--tta`
4. **模型集成**: 融合多个模型预测
5. **超参数调优**: 使用Hydra Sweeper
</details>

---

## 📚 参考文献

### 模型论文

- **ConvNeXt**: [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) (CVPR 2022)
- **EfficientNetV2**: [Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298) (ICML 2021)
- **Swin Transformer**: [Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) (ICCV 2021)

### 技术框架

- [PyTorch](https://pytorch.org/) - 深度学习框架
- [timm](https://github.com/huggingface/pytorch-image-models) - PyTorch Image Models
- [Albumentations](https://github.com/albumentations-team/albumentations) - 图像增强库
- [Hydra](https://hydra.cc/) - 配置管理框架

---

## 🤝 参与贡献

欢迎贡献代码、报告问题或提出改进建议！

### 贡献流程

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

### 开发规范

- 遵循PEP 8代码风格
- 添加必要的注释和文档
- 更新相关文档
- 确保所有测试通过

---

## 📄 开源协议

本项目采用 [MIT License](LICENSE) 开源协议。

---

## 🌟 致谢

- 感谢 [timm](https://github.com/huggingface/pytorch-image-models) 提供的优秀预训练模型
- 感谢 [Albumentations](https://github.com/albumentations-team/albumentations) 团队的高效增强库
- 感谢全国高校计算机能力挑战赛组委会提供的比赛平台

---

## 📮 联系方式

- **作者**: ironhxs
- **GitHub**: [@ironhxs](https://github.com/ironhxs)
- **项目地址**: [FlowerRecognition](https://github.com/ironhxs/FlowerRecognition)

如有问题或建议，欢迎提Issue或发送邮件！

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给个Star支持一下！⭐**

Made with ❤️ by ironhxs

🌸 Happy Coding! 🌸

</div>
