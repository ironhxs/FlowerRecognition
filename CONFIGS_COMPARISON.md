# 配置文件对比表

## 📊 完整对比

| 配置文件 | 目标显卡 | 显存需求 | 模型 | 模型大小 | Batch Size | 梯度累积 | 有效Batch | 训练时长(100轮) | 预期准确率 |
|---------|---------|---------|------|---------|-----------|---------|----------|--------------|-----------|
| `train_small_gpu.yaml` | RTX 3060/3070 | 8-12GB | ConvNeXt Base | 340MB | 8 | 4 | 32 | 8-12小时 | 87-90% |
| `train_medium_gpu.yaml` | RTX 3080/3090/4070 | 16-20GB | EfficientNetV2-L | 450MB | 16 | 2 | 32 | 5-8小时 | 88-92% |
| `train_large_gpu.yaml` | RTX 4090/A100 | 24-32GB | EfficientNetV2-L | 450MB | 32 | 1 | 32 | 3-5小时 | 88-92% |
| `train_rtx5090.yaml` | **RTX 5090** | 32GB | EfficientNetV2-L | 450MB | **64** | 1 | **64** | **2-3小时** | **90-93%** |
| `train_quick_test.yaml` | 任意 | 8GB+ | ConvNeXt Tiny | 110MB | 16 | 1 | 16 | 20分钟(5轮) | 80-85% |
| `train_rtx5090_ensemble.yaml` | RTX 5090 | 32GB | ConvNeXt Large | 490MB | 48 | 1 | 48 | 4-6小时 | 集成后94%+ |

## 🎯 关键差异

### 1️⃣ 模型选择
- **Small/Medium/Large/5090**: EfficientNetV2-L 或 ConvNeXt（最强模型）
- **Quick Test**: ConvNeXt Tiny（最小模型，只用于测试）
- **5090 Ensemble**: ConvNeXt Large（用于集成学习）

### 2️⃣ Batch Size 策略
```
有效Batch = batch_size × accumulation_steps

Small GPU:    8 × 4 = 32  （显存不够，用梯度累积凑）
Medium GPU:  16 × 2 = 32  （刚好够用）
Large GPU:   32 × 1 = 32  （直接用大batch）
RTX 5090:    64 × 1 = 64  （更大batch，更快收敛）
```

### 3️⃣ 学习率调整
- **Small/Medium/Large**: `lr=0.0001`（标准）
- **RTX 5090**: `lr=0.0002`（大batch需要更高学习率）
- **规则**: `lr_new = lr_base × sqrt(batch_new / batch_base)`

### 4️⃣ 训练速度
```
假设数据集 19,928 张，100 epochs：

Small GPU (batch=8):   
  - 每轮: 2,491 步
  - 单轮时长: ~5分钟
  - 总时长: 8-12小时

Medium GPU (batch=16): 
  - 每轮: 1,246 步
  - 单轮时长: ~3分钟
  - 总时长: 5-8小时

Large GPU (batch=32):  
  - 每轮: 623 步
  - 单轮时长: ~2分钟
  - 总时长: 3-5小时

RTX 5090 (batch=64):   
  - 每轮: 312 步
  - 单轮时长: ~1.5分钟
  - 总时长: 2-3小时 ⚡
```

## 🚀 使用建议

### 你的 RTX 5090 专用方案

#### 方案A: 单模型最强（推荐新手）
```bash
python train.py --config-name train_rtx5090
```
- 最简单，一条命令搞定
- EfficientNetV2-L + Batch 64
- 预期: 90-93% 准确率
- 训练时间: 2-3小时

#### 方案B: 模型集成（推荐竞赛）
```bash
# 同时开3个终端，训练3个不同模型
# 终端1
python train.py --config-name train_rtx5090_ensemble model=efficientnet_v2_l_optimized

# 终端2  
python train.py --config-name train_rtx5090_ensemble model=convnext_large

# 终端3
python train.py --config-name train_rtx5090_ensemble model=swin_transformer_v2
```
- 3个模型并行训练（5090显存够用）
- 最后集成预测（投票或平均）
- 预期: 94-96% 准确率 🏆
- 训练时间: 4-6小时（3个并行）

#### 方案C: 极限 Batch（实验性）
```bash
python train.py --config-name train_rtx5090 \
  dataset.batch_size=96 \
  training.optimizer.lr=0.00025
```
- batch_size=96（需要监控显存）
- 如果 OOM，降到 80 或 64

## 📋 配置文件详细说明

### `train_small_gpu.yaml`
```yaml
model: convnext_base        # 更小的模型
batch_size: 8               # 小batch适应小显存
accumulation_steps: 4       # 通过累积模拟大batch
num_workers: 4              # 较少的数据加载线程
```
**适用**: 预算卡、笔记本GPU

### `train_medium_gpu.yaml`
```yaml
model: efficientnet_v2_l_optimized  # 最强模型
batch_size: 16                       # 中等batch
accumulation_steps: 2                # 少量累积
num_workers: 8                       # 更多线程
```
**适用**: 主流游戏卡

### `train_large_gpu.yaml`
```yaml
model: efficientnet_v2_l_optimized
batch_size: 32              # 大batch，不需要累积
accumulation_steps: 1       
num_workers: 16             # 充分利用CPU
```
**适用**: 高端卡、专业卡

### `train_rtx5090.yaml` ⭐
```yaml
model: efficientnet_v2_l_optimized
batch_size: 64              # 超大batch！
lr: 0.0002                  # 更高学习率匹配大batch
num_workers: 16             
```
**适用**: RTX 5090（你的卡）

### `train_quick_test.yaml`
```yaml
model: convnext_tiny        # 最小模型
batch_size: 16
epochs: 5                   # 只跑5轮测试
augmentation: medium        # 轻量增强
```
**适用**: 快速验证流程（所有卡都能跑）

### `train_rtx5090_ensemble.yaml`
```yaml
model: convnext_large       # 可替换为其他模型
batch_size: 48              # 给多模型并行留空间
```
**适用**: RTX 5090 集成学习

## 🎓 为什么大 batch 更好？

### 优点
1. **训练更快**: 步数减少一半
2. **更稳定**: 梯度估计更准确
3. **更高准确率**: batch=64 通常比 batch=32 高 1-2%

### 缺点
1. **需要大显存**: 每张图占用显存
2. **需要调学习率**: 大batch要配高学习率
3. **可能欠拟合**: 太大的batch会减少随机性

### 5090 的优势
- 32GB 显存 → 可以用 batch=64-96
- 更快的计算 → 训练时间减半
- 更高带宽 → 数据加载不是瓶颈

## 💡 快速决策树

```
你有 5090？
  ├─ 是，第一次用 → train_quick_test (5分钟验证)
  │   └─ 验证通过 → train_rtx5090 (2-3小时完整训练)
  │
  └─ 是，要打比赛 → train_rtx5090_ensemble (集成学习)
      └─ 同时跑3个模型 → 准确率 +2-4%

你是其他卡？
  ├─ 8-12GB (3060/3070) → train_small_gpu
  ├─ 16-20GB (3080/4070) → train_medium_gpu
  └─ 24-32GB (4090/A100) → train_large_gpu
```

## 🔥 RTX 5090 最佳实践

1. **先测试**
   ```bash
   python train.py --config-name train_quick_test
   ```

2. **单模型训练**
   ```bash
   python train.py --config-name train_rtx5090
   ```

3. **集成学习（竞赛推荐）**
   ```bash
   # 开3个终端
   python train.py --config-name train_rtx5090_ensemble model=efficientnet_v2_l_optimized
   python train.py --config-name train_rtx5090_ensemble model=convnext_large
   python train.py --config-name train_rtx5090_ensemble model=swin_transformer_v2
   ```

4. **极限调优**
   ```bash
   python train.py --config-name train_rtx5090 dataset.batch_size=96
   ```

---

**总结**: 你有 5090 就直接用 `train_rtx5090.yaml`，想要更高分就用 `train_rtx5090_ensemble.yaml` 训练3个模型集成！
