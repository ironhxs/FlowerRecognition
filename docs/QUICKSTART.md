# Flower Recognition - Quick Start Guide

## 快速开始指南

### 1. 环境准备

```bash
# 克隆仓库
git clone https://github.com/ironhxs/FlowerRecognition.git
cd FlowerRecognition

# 安装依赖
pip install -r requirements.txt

# 或者使用setup脚本
chmod +x setup.sh
./setup.sh
```

### 2. 数据准备

#### 下载比赛数据
从比赛官网下载训练数据和测试数据。

#### 组织数据结构
```
data/
├── train.csv          # 格式: image_id, label
├── train/             # 训练图片文件夹
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── test/              # 测试图片文件夹
    ├── test_001.jpg
    ├── test_002.jpg
    └── ...
```

#### 生成示例数据（用于测试）
```bash
python generate_sample_data.py --num-classes 100 --samples-per-class 10
```

### 3. 训练模型

#### 使用默认配置训练
```bash
python train.py
```

#### 选择不同的模型
```bash
# ConvNeXt Base (推荐，速度快)
python train.py model=convnext_base

# EfficientNetV2-L (准确率高)
python train.py model=efficientnet_v2_l

# Swin Transformer V2 (最新架构)
python train.py model=swin_transformer_v2
```

#### 自定义训练参数
```bash
python train.py \
    model=convnext_base \
    training.epochs=100 \
    training.optimizer.lr=2e-4 \
    dataset.batch_size=16
```

### 4. 监控训练过程

在另一个终端窗口启动TensorBoard：
```bash
tensorboard --logdir results/logs
```

然后在浏览器中访问 http://localhost:6006

### 5. 生成预测

#### 基础预测
```bash
python inference.py \
    --checkpoint results/checkpoints/best_model.pt \
    --output predictions.csv
```

#### 使用测试时增强 (TTA)
```bash
python inference.py \
    --checkpoint results/checkpoints/best_model.pt \
    --output predictions.csv \
    --tta
```

#### 测试推理速度
```bash
python inference.py \
    --checkpoint results/checkpoints/best_model.pt \
    --benchmark
```

### 6. 准备提交

#### 验证预测格式
```bash
python prepare_submission.py \
    --checkpoint results/checkpoints/best_model.pt \
    --predictions predictions.csv \
    --verify-only
```

#### 创建提交包
```bash
python prepare_submission.py \
    --checkpoint results/checkpoints/best_model.pt \
    --predictions predictions.csv \
    --output submission.zip
```

### 7. 使用CLI工具

```bash
# 查看可用模型
python cli/flower_cli.py models

# 查看系统信息
python cli/flower_cli.py info

# 验证数据集
python cli/flower_cli.py prepare-data --data-dir ./data

# 训练
python cli/flower_cli.py train

# 预测
python cli/flower_cli.py predict \
    --checkpoint results/checkpoints/best_model.pt \
    --output predictions.csv
```

## 常见问题

### Q: CUDA内存不足
A: 减小批次大小
```bash
python train.py dataset.batch_size=8
```

### Q: 训练速度太慢
A: 
1. 确认混合精度训练已启用（默认开启）
2. 减少数据加载工作进程
```bash
python train.py num_workers=2
```

### Q: 模型太大超过500MB
A: 使用较小的模型
```bash
python train.py model=convnext_base
```

### Q: 推理时间超过100ms
A: 
1. 使用更快的模型架构
2. 使用PyTorch 2.0编译优化
3. 确保使用GPU推理

## 高级技巧

### 模型集成
```python
# 训练多个模型
python train.py model=convnext_base experiment_name=model1
python train.py model=efficientnet_v2_l experiment_name=model2

# 在代码中实现集成预测
# 参考README.md中的集成示例
```

### 自定义数据增强
编辑 `configs/augmentation/custom.yaml`:
```yaml
train:
  - RandomResizedCrop:
      size: 600
      scale: [0.8, 1.0]
  - RandomHorizontalFlip:
      p: 0.5
  # 添加更多增强...
```

使用自定义增强：
```bash
python train.py augmentation=custom
```

### 学习率查找
```bash
# 使用较小的epochs来测试不同学习率
python train.py training.epochs=10 training.optimizer.lr=1e-5
python train.py training.epochs=10 training.optimizer.lr=5e-5
python train.py training.epochs=10 training.optimizer.lr=1e-4
```

## 性能基准

### 推荐配置（在单个V100 GPU上）

| 模型 | 批次大小 | 训练时间/epoch | 验证准确率 | 推理时间 |
|------|---------|----------------|-----------|----------|
| ConvNeXt Base | 32 | ~3分钟 | 90%+ | ~50ms |
| EfficientNetV2-L | 16 | ~5分钟 | 92%+ | ~70ms |
| Swin-V2 Base | 24 | ~4分钟 | 91%+ | ~60ms |

*注：实际性能取决于硬件配置和数据集

## 比赛提交检查清单

- [ ] 模型大小 < 500MB
- [ ] 推理时间 < 100ms
- [ ] 预测CSV格式正确（image_id, label）
- [ ] CSV编码为UTF-8
- [ ] 所有测试图片都有预测
- [ ] 标签范围在0-99之间
- [ ] 提交包结构正确
- [ ] 技术报告完成
- [ ] 代码可复现

## 获取帮助

1. 查看完整文档：README.md
2. 查看技术报告模板：docs/technical_report_template.md
3. 提交Issue到GitHub仓库
4. 参考示例代码和配置文件

---

祝比赛顺利！🌸🏆
