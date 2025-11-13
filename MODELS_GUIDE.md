# Model Configuration Quick Reference

## 🚀 所有可用配置（按模型命名）

```bash
configs/
├── convnext_tiny.yaml           # 最快，测试用
├── convnext_base.yaml           # 平衡，主力模型
├── convnext_large.yaml          # 接近大小上限
├── efficientnet_b3.yaml         # 推理最快
├── efficientnetv2_l.yaml        # 最高准确率 ⭐
└── swin_transformer_v2.yaml     # Transformer架构
```

## 📊 模型对比

| 配置文件 | 模型大小 | 训练时长 | 推理速度 | 准确率 | 显存需求 | 推荐用途 |
|---------|---------|---------|---------|--------|---------|---------|
| `convnext_tiny` | 110MB | 3-4h | ~30ms | 82-86% | 6GB | 快速测试 |
| `efficientnet_b3` | 180MB | 4-6h | ~40ms | 85-88% | 8GB | 推理优化 |
| `convnext_base` | 340MB | 5-8h | ~50ms | 87-90% | 12GB | 主力模型 |
| `swin_transformer_v2` | 350MB | 6-9h | ~60ms | 87-90% | 14GB | 集成多样性 |
| **`efficientnetv2_l`** | **450MB** | **5-8h** | **~70ms** | **88-92%** | **16GB** | **竞赛首选** |
| `convnext_large` | 490MB | 6-10h | ~90ms | 88-91% | 20GB | 最大容量 |

## 🎯 使用方法

### 单模型训练
```bash
# 直接用模型名
python train.py --config-name efficientnetv2_l

# 或者
python train.py --config-name convnext_base

# 或者
python train.py --config-name swin_transformer_v2
```

### 调整参数
```bash
# 小显存：降低 batch size
python train.py --config-name efficientnetv2_l dataset.batch_size=8

# 大显存：提高 batch size
python train.py --config-name efficientnetv2_l dataset.batch_size=32

# 快速测试：减少 epochs
python train.py --config-name convnext_base training.epochs=10
```

### 模型集成训练（3个并行）
```bash
# 终端1
python train.py --config-name efficientnetv2_l

# 终端2  
python train.py --config-name convnext_base

# 终端3
python train.py --config-name swin_transformer_v2
```

## 💡 选择建议

### 按目标选择

**最高准确率（竞赛）**:
```bash
python train.py --config-name efficientnetv2_l
```
- 单模型准确率: 88-92%
- 模型大小: 450MB < 500MB ✓
- 推理速度: ~70ms < 100ms ✓

**快速迭代（开发）**:
```bash
python train.py --config-name convnext_tiny
```
- 3-4小时完成训练
- 快速验证想法

**推理速度优先**:
```bash
python train.py --config-name efficientnet_b3
```
- 推理仅需 40ms
- 准确率仍有 85-88%

**模型集成（冲榜）**:
```bash
# 同时训练3个不同架构
efficientnetv2_l + convnext_base + swin_transformer_v2
# 集成后准确率 +2-4%
```

### 按显卡选择

**8GB (RTX 3060)**:
```bash
python train.py --config-name efficientnet_b3
# 或
python train.py --config-name convnext_tiny
```

**12GB (RTX 3060Ti/4060Ti)**:
```bash
python train.py --config-name convnext_base dataset.batch_size=12
```

**16GB (RTX 3080/4070)**:
```bash
python train.py --config-name efficientnetv2_l
# 默认配置就是针对 16GB 优化的
```

**24GB (RTX 4090)**:
```bash
python train.py --config-name efficientnetv2_l dataset.batch_size=32
# 可以用更大的 batch
```

**32GB (RTX 5090)**:
```bash
python train.py --config-name efficientnetv2_l \
  dataset.batch_size=64 \
  training.optimizer.lr=0.0002
# 极限性能，2-3小时完成
```

## 🔧 配置文件内容说明

每个配置文件都包含：
```yaml
defaults:
  - model: xxx              # 模型架构
  - dataset: flower100      # 数据集配置
  - training: focal         # Focal Loss + 优化器
  - augmentation: ultra_strong  # 数据增强

dataset:
  batch_size: 16           # 根据模型大小调整
  
training:
  epochs: 100              # 训练轮数
  accumulation_steps: 2    # 梯度累积
```

## 📝 命名规则

- **文件名 = 模型名**
- 简洁明了，直接对应 `configs/model/` 下的模型配置
- 不再用 `train_small_gpu` 这种抽象名称

## 🎓 常见操作

### 1. 查看配置
```bash
python train.py --config-name efficientnetv2_l --cfg job
```

### 2. 覆盖单个参数
```bash
python train.py --config-name efficientnetv2_l dataset.batch_size=8
```

### 3. 覆盖多个参数
```bash
python train.py --config-name efficientnetv2_l \
  dataset.batch_size=12 \
  training.epochs=50 \
  augmentation=medium
```

### 4. 后台训练
```bash
nohup python train.py --config-name efficientnetv2_l > train.log 2>&1 &
tail -f train.log
```

## 🏆 竞赛推荐流程

1. **快速验证** (5分钟)
   ```bash
   python train.py --config-name convnext_tiny training.epochs=2
   ```

2. **单模型训练** (5-8小时)
   ```bash
   python train.py --config-name efficientnetv2_l
   ```

3. **模型集成** (并行训练)
   ```bash
   # 3个终端同时跑
   python train.py --config-name efficientnetv2_l
   python train.py --config-name convnext_base
   python train.py --config-name swin_transformer_v2
   ```

4. **预测与提交**
   ```bash
   python inference.py --checkpoint results/checkpoints/best_model.pt --output predictions.csv --tta
   python prepare_submission.py --checkpoint <path> --predictions predictions.csv
   ```

---

**现在配置文件直接用模型名，一看就懂！** 🎯
