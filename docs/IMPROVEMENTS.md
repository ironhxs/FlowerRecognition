# 🔬 模型架构与改进建议

## 📊 当前使用的模型

### 支持的 6 个模型架构（基于 timm 库）

| 模型 | 参数量 | 模型大小 | 架构类型 | 适用场景 | 比赛约束 |
|------|--------|----------|----------|----------|----------|
| **ConvNeXt Tiny** | 29M | ~110MB | Modern CNN | 快速测试/基线 | ✅ 符合 |
| **ConvNeXt Base** ⭐ | 89M | ~340MB | Modern CNN | 平衡性能（默认） | ✅ 符合 |
| **ConvNeXt Large** | 198M | ~760MB | Modern CNN | 最高精度 | ❌ **超限！** |
| **EfficientNet B3** | 12M | ~45MB | Efficient CNN | 轻量级/快速 | ✅ 符合 |
| **EfficientNetV2-L** ⭐ | 120M | ~460MB | Efficient CNN | 高精度 | ✅ 符合 |
| **Swin Transformer V2** | 88M | ~330MB | Vision Transformer | 最新架构 | ✅ 符合 |

⭐ = 推荐用于比赛

---

## 🎯 当前训练策略

### 1. 数据增强（Albumentations）
**Strong 级别（默认）:**
```yaml
几何变换:
  - RandomResizedCrop (scale: 0.7-1.0)
  - RandomHorizontalFlip (p=0.5)
  - RandomVerticalFlip (p=0.3)
  - ShiftScaleRotate (rotate: ±45°, p=0.5)

颜色增强:
  - ColorJitter (亮度/对比度/饱和度: 0.3)
  - RandomBrightnessContrast (p=0.5)

高级增强:
  - GaussianBlur (p=0.2)
  - GaussNoise (p=0.2)
  - CoarseDropout/Cutout (8个洞, p=0.3)

标准化: ImageNet 统计值
```

### 2. 优化器配置
- **优化器**: AdamW
- **学习率**: 1e-4
- **权重衰减**: 0.05
- **调度器**: Cosine Annealing + 5 epoch Warmup

### 3. 训练技巧
- ✅ 混合精度训练（AMP）- 加速 + 省内存
- ✅ Label Smoothing (0.1) - 提升泛化
- ✅ 梯度裁剪 (1.0) - 稳定训练
- ✅ Early Stopping (patience=10)
- ✅ Drop Path/Stochastic Depth (0.1)

### 4. 推理增强
- ✅ Test-Time Augmentation (TTA) - 水平翻转
- ✅ 混合精度推理

---

## 🚀 改进建议（按优先级排序）

### 🔥 高优先级改进（立即可做）

#### 1. **模型集成（Ensemble）** - 预计提升 2-5%
**现状**: 单模型预测  
**改进**: 
```python
# 训练多个不同架构的模型
models = [
    'convnext_base',      # 89M, CNN
    'efficientnet_v2_l',  # 120M, Efficient CNN
    'swin_transformer_v2' # 88M, Transformer
]

# 集成策略
1. 简单平均: predictions = mean([model1, model2, model3])
2. 加权平均: predictions = w1*model1 + w2*model2 + w3*model3
3. 投票机制: predictions = mode([model1, model2, model3])
```

**实现方式**:
```bash
# 训练3个模型
python train.py model=convnext_base experiment_name=model1
python train.py model=efficientnet_v2_l experiment_name=model2
python train.py model=swin_transformer_v2 experiment_name=model3

# 修改 inference.py 支持集成
```

---

#### 2. **增强 TTA 策略** - 预计提升 1-3%
**现状**: 仅水平翻转  
**改进**: 
```python
TTA_transforms = [
    原图,
    水平翻转,
    垂直翻转,
    旋转 ±5°,
    亮度调整 (±10%)
]
# 取所有变换的平均预测
```

**需要修改**: `inference.py` 的 TTA 部分

---

#### 3. **优化数据增强** - 针对花卉特征
**现状**: 通用增强  
**改进**:
```yaml
# 添加花卉特定增强
- RandomRotate90:  # 花卉可能从任意角度拍摄
    p: 0.5
    
- ElasticTransform:  # 模拟自然形变
    alpha: 1
    sigma: 50
    p: 0.3

- HueSaturationValue:  # 花朵颜色是关键特征
    hue_shift_limit: 20
    sat_shift_limit: 30
    val_shift_limit: 20
    p: 0.5

- MultiplicativeNoise:  # 模拟不同光照
    multiplier: [0.9, 1.1]
    p: 0.3
```

**需要修改**: `configs/augmentation/strong.yaml`

---

#### 4. **学习率调优** - 可能提升 1-2%
**现状**: 固定 lr=1e-4  
**改进尝试**:
```yaml
# 方案A: 更高的初始学习率 + 更长 warmup
lr: 2e-4
warmup_epochs: 10

# 方案B: 使用 OneCycleLR
scheduler:
  name: onecycle
  max_lr: 3e-4
  pct_start: 0.3
```

---

### 🔧 中优先级改进（需要额外实验）

#### 5. **更大的输入尺寸** - 可能提升 1-3%
**现状**: 600×600  
**风险**: 可能影响推理速度（需验证 <100ms）

```yaml
# 尝试更大尺寸（如果 GPU 内存允许）
input_size: 768  # 或 640

# 需要同时调整：
- batch_size: 减小到 16 或 8
- 验证推理速度是否符合要求
```

**需要测试**: 
```bash
python inference.py --checkpoint <model> --benchmark
```

---

#### 6. **知识蒸馏（Knowledge Distillation）**
**策略**: 
```
Teacher: EfficientNetV2-L (120M, 高精度)
Student: ConvNeXt Base (89M, 快速)

目标: Student 达到接近 Teacher 的精度，但更快
```

**需要新增**: 知识蒸馏训练脚本

---

#### 7. **Focal Loss 替代交叉熵**
**场景**: 如果发现某些类别样本预测困难  
**改进**:
```python
# 替换 CrossEntropyLoss
loss = FocalLoss(
    alpha=0.25,  # 平衡因子
    gamma=2.0    # 聚焦参数
)
```

**需要修改**: `train.py` 的损失函数部分

---

#### 8. **渐进式训练（Progressive Training）**
```bash
# 阶段1: 冻结 backbone，只训练分类头
python train.py training.freeze_backbone=true training.epochs=10

# 阶段2: 解冻全部，低学习率微调
python train.py training.lr=5e-5 training.epochs=40
```

**需要新增**: freeze_backbone 功能（代码中已有 `freeze_backbone()` 方法）

---

### 💡 低优先级改进（实验性）

#### 9. **注意力机制增强**
```python
# 在模型中添加
- Squeeze-and-Excitation (SE) blocks
- CBAM (Convolutional Block Attention Module)
- ECA (Efficient Channel Attention)
```

**需要修改**: `models/flower_model.py`

---

#### 10. **Mix-up / Cut-mix 数据增强**
```python
# 训练时混合两张图片
alpha = 0.2
lam = np.random.beta(alpha, alpha)
mixed_image = lam * image1 + (1 - lam) * image2
mixed_label = lam * label1 + (1 - lam) * label2
```

---

#### 11. **自监督预训练**
如果有额外的无标注花卉图片：
```
1. SimCLR / MoCo 在无标注数据上预训练
2. 在比赛数据上微调
```

---

#### 12. **更好的后处理**
```python
# 温度缩放（Temperature Scaling）
predictions = softmax(logits / T)  # T=1.5 或 2.0

# 可以提升预测概率的校准
```

---

## 📈 预估提升潜力

| 改进项 | 难度 | 时间成本 | 预期提升 | 推荐指数 |
|--------|------|----------|----------|----------|
| 模型集成 | 低 | 3倍训练时间 | 2-5% | ⭐⭐⭐⭐⭐ |
| 增强 TTA | 低 | 1小时 | 1-3% | ⭐⭐⭐⭐⭐ |
| 优化数据增强 | 低 | 2小时 | 1-2% | ⭐⭐⭐⭐ |
| 学习率调优 | 低 | 多次实验 | 1-2% | ⭐⭐⭐⭐ |
| 更大输入尺寸 | 中 | 需测试速度 | 1-3% | ⭐⭐⭐ |
| 知识蒸馏 | 高 | 1-2天 | 2-4% | ⭐⭐⭐ |
| Focal Loss | 中 | 1小时 | 0-2% | ⭐⭐ |
| 渐进式训练 | 中 | 额外时间 | 1-2% | ⭐⭐⭐ |

---

## 🎯 推荐实施路线

### 阶段1: 快速提升（1-2天）
1. ✅ 训练 ConvNeXt Base（已准备好）
2. ✅ 训练 EfficientNetV2-L
3. ✅ 训练 Swin Transformer V2
4. ✅ 实现 3 模型集成
5. ✅ 增强 TTA（5种变换）

**预期提升**: 基线 + 3-8%

---

### 阶段2: 优化调参（1-2天）
1. 🔧 优化数据增强（添加花卉特定增强）
2. 🔧 学习率网格搜索
3. 🔧 尝试更大输入尺寸（测试速度）
4. 🔧 Label Smoothing 调参 (0.05, 0.1, 0.15)

**预期提升**: 额外 1-3%

---

### 阶段3: 高级技术（可选，2-3天）
1. 💡 知识蒸馏
2. 💡 Focal Loss / 加权损失
3. 💡 渐进式训练
4. 💡 更多模型架构实验

**预期提升**: 额外 1-2%

---

## 🛠️ 立即可执行的命令

### 训练集成所需的3个模型
```bash
# 模型1: ConvNeXt Base
python train.py model=convnext_base experiment_name=convnext_base_v1

# 模型2: EfficientNetV2-L
python train.py model=efficientnet_v2_l experiment_name=efficientnetv2_v1

# 模型3: Swin Transformer V2
python train.py model=swin_transformer_v2 experiment_name=swin_v1
```

### 检查推理速度
```bash
python inference.py --checkpoint results/checkpoints/best_model.pt --benchmark
```

---

## 📚 需要创建的新脚本

1. **`ensemble.py`** - 模型集成脚本
2. **`enhanced_tta.py`** - 增强 TTA 实现
3. **`distillation.py`** - 知识蒸馏训练（可选）

---

## ⚠️ 注意事项

### 必须遵守的约束
- ✅ 单模型大小 < 500MB（集成时每个模型单独检查）
- ✅ 推理速度 < 100ms（如果集成，需要优化或选择更快的模型）
- ✅ 参数量 < 10B（所有当前模型都符合）

### 风险控制
- 模型集成会增加推理时间（需要测试）
- 更大输入尺寸可能违反速度约束
- 过度增强可能导致欠拟合

---

**建议优先实施**: 模型集成 + 增强TTA，这两项改进成本低、效果明显！ 🚀
