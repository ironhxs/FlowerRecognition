# Training Configuration Quick Reference

## 📋 使用方法

### 根据显卡选择配置

```bash
# 小显卡 (8GB-12GB) - ConvNeXt Base
python train.py --config-name train_small_gpu

# 中等显卡 (16GB-20GB) - EfficientNetV2-L  
python train.py --config-name train_medium_gpu

# 大显卡 (24GB+) - EfficientNetV2-L 全速
python train.py --config-name train_large_gpu

# 快速测试 (验证流程)
python train.py --config-name train_quick_test
```

## 🎯 Hydra 配置优先级（从高到低）

1. **命令行参数** (最高优先级)
   ```bash
   python train.py --config-name train_medium_gpu training.epochs=50
   ```

2. **--config-name 指定的配置文件**
   ```bash
   train_medium_gpu.yaml  # 会覆盖 config.yaml
   ```

3. **defaults 列表中的配置**
   ```yaml
   defaults:
     - model: efficientnet_v2_l_optimized  # 覆盖默认 model
     - training: focal                      # 覆盖默认 training
   ```

4. **基础配置文件**
   ```bash
   configs/config.yaml  # 最低优先级
   ```

## 🔧 常见参数覆盖示例

### 只改 batch size
```bash
python train.py --config-name train_medium_gpu dataset.batch_size=8
```

### 改多个参数
```bash
python train.py --config-name train_medium_gpu \
  dataset.batch_size=12 \
  training.epochs=50 \
  training.optimizer.lr=0.00005
```

### 换模型但保持其他设置
```bash
python train.py --config-name train_medium_gpu model=convnext_base
```

### 关闭数据增强测试
```bash
python train.py --config-name train_quick_test augmentation=light
```

## 📊 配置文件对比

| 配置文件 | 显存需求 | 模型 | Batch Size | 累积步数 | 有效Batch |
|---------|---------|------|-----------|---------|----------|
| train_small_gpu | 8-12GB | ConvNeXt Base | 8 | 4 | 32 |
| train_medium_gpu | 16-20GB | EfficientNetV2-L | 16 | 2 | 32 |
| train_large_gpu | 24GB+ | EfficientNetV2-L | 32 | 1 | 32 |
| train_quick_test | 任意 | ConvNeXt Tiny | 16 | 1 | 16 |

## 🎓 梯度累积说明

**有效 Batch Size = batch_size × accumulation_steps**

- `batch_size=8, accumulation_steps=4` → 效果等同 `batch_size=32`
- 显存占用：只按 `batch_size=8` 计算
- 训练效果：和 `batch_size=32` 几乎一样，只是稍慢

## ⚡ 服务器训练完整示例

### Linux/Mac (bash)
```bash
#!/bin/bash
conda activate flower
export HF_ENDPOINT=https://hf-mirror.com

# 根据显卡选择
python train.py --config-name train_medium_gpu

# 后台运行
nohup python train.py --config-name train_medium_gpu > train.log 2>&1 &
tail -f train.log
```

### Windows (PowerShell)
```powershell
conda activate flower
$env:HF_ENDPOINT="https://hf-mirror.com"

# 直接运行
python train.py --config-name train_medium_gpu

# 监控日志（另开终端）
tensorboard --logdir results/logs
```

## 🐛 常见问题

### Q: 还是显存不够？
```bash
# 方案1: 减小 batch_size
python train.py --config-name train_small_gpu dataset.batch_size=4

# 方案2: 增加累积步数
python train.py --config-name train_small_gpu \
  dataset.batch_size=4 \
  training.accumulation_steps=8  # 有效batch=32
```

### Q: 想用不同的数据增强？
```bash
# 使用轻量增强（更快）
python train.py --config-name train_medium_gpu augmentation=light

# 使用中等增强
python train.py --config-name train_medium_gpu augmentation=medium

# 使用超强增强（当前默认）
python train.py --config-name train_medium_gpu augmentation=ultra_strong
```

### Q: 想改学习率？
```bash
python train.py --config-name train_medium_gpu training.optimizer.lr=0.00005
```

### Q: 想从 checkpoint 恢复？
编辑 `train.py` 第 60 行附近，取消注释：
```python
# 加载 checkpoint 恢复训练
checkpoint = torch.load('results/checkpoints/checkpoint_epoch_50.pt')
self.model.load_state_dict(checkpoint['model_state_dict'])
self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
self.current_epoch = checkpoint['epoch']
```

## 📁 配置文件位置

```
configs/
├── train_small_gpu.yaml      # 小显卡配置
├── train_medium_gpu.yaml     # 中等显卡配置  
├── train_large_gpu.yaml      # 大显卡配置
├── train_quick_test.yaml     # 快速测试配置
└── high_performance.yaml     # 之前的高性能配置（已废弃，用上面的）
```

---

**推荐用法：先用 `train_quick_test` 验证流程，再根据显卡用对应配置训练！**
