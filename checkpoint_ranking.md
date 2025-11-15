# Checkpoint 排序报告

**生成时间**: 2025-11-15 18:27:20  
**总数**: 15 个 checkpoints  
**基础路径**: `/root/autodl-tmp/`

---

## 📊 按 Val Acc 排序 (降序)

| 排名 | Val Acc | Epoch | LR | MixUp | 实验路径 | 大小 | 日期 |
|------|---------|-------|-----|-------|----------|------|------|
| 1 | **97.66%** | 19 | 2.6e-05 | ✅/✅ | `checkpoints_sweep_lr_mixup/lr_2.6e-5_mixup` | 332MB | 2025-11-15 17:35 |
| 2 | **97.56%** | 16 | 2.6e-05 | ❌/❌ | `checkpoints_sweep_lr/lr_2.6e-5` | 332MB | 2025-11-15 16:08 |
| 3 | **97.53%** | 9 | 2.2e-05 | ❌/❌ | `checkpoints_sweep_lr/lr_2.2e-5` | 332MB | 2025-11-15 15:12 |
| 4 | **97.53%** | 13 | 2.4e-05 | ❌/❌ | `checkpoints_sweep_lr/lr_2.4e-5` | 332MB | 2025-11-15 15:40 |
| 5 | **97.49%** | 14 | 1.6e-05 | ❌/❌ | `checkpoints_sweep_lr/lr_1.6e-5` | 332MB | 2025-11-15 13:49 |
| 6 | **97.46%** | 10 | 1.4e-05 | ❌/❌ | `checkpoints_sweep_lr/lr_1.4e-5` | 332MB | 2025-11-15 13:21 |
| 7 | **97.46%** | 4 | 1.8e-05 | ❌/❌ | `checkpoints_sweep_lr/lr_1.8e-5` | 332MB | 2025-11-15 14:17 |
| 8 | **97.42%** | 9 | 1.0e-05 | ❌/❌ | `checkpoints_sweep_lr/lr_1.0e-5` | 332MB | 2025-11-15 12:24 |
| 9 | **97.42%** | 9 | 1.2e-05 | ❌/❌ | `checkpoints_sweep_lr/lr_1.2e-5` | 332MB | 2025-11-15 12:53 |
| 10 | **97.39%** | 9 | 2.0e-05 | ❌/❌ | `checkpoints_sweep_lr/lr_2.0e-5` | 332MB | 2025-11-15 14:45 |
| 11 | **97.36%** | 14 | 8.0e-06 | ❌/❌ | `checkpoints_sweep_lr/lr_8.0e-6` | 332MB | 2025-11-15 11:56 |
| 12 | **97.26%** | 9 | 2.0e-05 | ❌/❌ | `checkpoints_swin_anti_overfit/fast_decay_20ep` | 332MB | 2025-11-15 03:01 |
| 13 | **96.99%** | 11 | 4.0e-05 | ❌/❌ | `checkpoints_swin_anti_overfit/default` | 996MB | 2025-11-15 02:18 |
| 14 | **96.96%** | 14 | 6.0e-06 | ❌/❌ | `checkpoints_sweep_lr/lr_6.0e-6` | 332MB | 2025-11-15 03:43 |
| 15 | **95.65%** | 1 | 3.0e-05 | ✅/✅ | `checkpoints_sweep_lr/lr_3.0e-5` | 996MB | 2025-11-15 16:11 |


---

## 🎯 Top 5 推荐


### 1. checkpoints_sweep_lr_mixup/lr_2.6e-5_mixup - **97.66%**

- **完整路径**: `/root/autodl-tmp/checkpoints_sweep_lr_mixup/lr_2.6e-5_mixup/best_model_weights_only.pt`
- **训练轮数**: Epoch 19
- **学习率**: 2.6e-05
- **MixUp/CutMix**: 启用 / 启用
- **数据增强**: 6 transforms
- **模型**: ms_in22k_ft_in1k
- **文件大小**: 332.0 MB

**使用命令**:
```bash
python inference.py --checkpoint "/root/autodl-tmp/checkpoints_sweep_lr_mixup/lr_2.6e-5_mixup/best_model_weights_only.pt" --output predictions.csv
```

### 2. checkpoints_sweep_lr/lr_2.6e-5 - **97.56%**

- **完整路径**: `/root/autodl-tmp/checkpoints_sweep_lr/lr_2.6e-5/best_model_weights_only.pt`
- **训练轮数**: Epoch 16
- **学习率**: 2.6e-05
- **MixUp/CutMix**: 未启用 / 未启用
- **数据增强**: 6 transforms
- **模型**: ms_in22k_ft_in1k
- **文件大小**: 332.0 MB

**使用命令**:
```bash
python inference.py --checkpoint "/root/autodl-tmp/checkpoints_sweep_lr/lr_2.6e-5/best_model_weights_only.pt" --output predictions.csv
```

### 3. checkpoints_sweep_lr/lr_2.2e-5 - **97.53%**

- **完整路径**: `/root/autodl-tmp/checkpoints_sweep_lr/lr_2.2e-5/best_model_weights_only.pt`
- **训练轮数**: Epoch 9
- **学习率**: 2.2e-05
- **MixUp/CutMix**: 未启用 / 未启用
- **数据增强**: 6 transforms
- **模型**: ms_in22k_ft_in1k
- **文件大小**: 332.0 MB

**使用命令**:
```bash
python inference.py --checkpoint "/root/autodl-tmp/checkpoints_sweep_lr/lr_2.2e-5/best_model_weights_only.pt" --output predictions.csv
```

### 4. checkpoints_sweep_lr/lr_2.4e-5 - **97.53%**

- **完整路径**: `/root/autodl-tmp/checkpoints_sweep_lr/lr_2.4e-5/best_model_weights_only.pt`
- **训练轮数**: Epoch 13
- **学习率**: 2.4e-05
- **MixUp/CutMix**: 未启用 / 未启用
- **数据增强**: 6 transforms
- **模型**: ms_in22k_ft_in1k
- **文件大小**: 332.0 MB

**使用命令**:
```bash
python inference.py --checkpoint "/root/autodl-tmp/checkpoints_sweep_lr/lr_2.4e-5/best_model_weights_only.pt" --output predictions.csv
```

### 5. checkpoints_sweep_lr/lr_1.6e-5 - **97.49%**

- **完整路径**: `/root/autodl-tmp/checkpoints_sweep_lr/lr_1.6e-5/best_model_weights_only.pt`
- **训练轮数**: Epoch 14
- **学习率**: 1.6e-05
- **MixUp/CutMix**: 未启用 / 未启用
- **数据增强**: 6 transforms
- **模型**: ms_in22k_ft_in1k
- **文件大小**: 332.0 MB

**使用命令**:
```bash
python inference.py --checkpoint "/root/autodl-tmp/checkpoints_sweep_lr/lr_1.6e-5/best_model_weights_only.pt" --output predictions.csv
```


---

## 📈 实验对比分析

### MixUp/CutMix 效果对比


- **有 MixUp**: 平均 Val Acc = 96.66% (2 个实验)
- **无 MixUp**: 平均 Val Acc = 97.37% (13 个实验)
- **提升**: -0.72%


### 学习率分布

| 学习率 | 最佳 Val Acc | 实验 |
|--------|-------------|------|
| 1.0e-05 | 97.42% | `checkpoints_sweep_lr/lr_1.0e-5` |
| 1.2e-05 | 97.42% | `checkpoints_sweep_lr/lr_1.2e-5` |
| 1.4e-05 | 97.46% | `checkpoints_sweep_lr/lr_1.4e-5` |
| 1.6e-05 | 97.49% | `checkpoints_sweep_lr/lr_1.6e-5` |
| 1.8e-05 | 97.46% | `checkpoints_sweep_lr/lr_1.8e-5` |
| 2.0e-05 | 97.39% | `checkpoints_sweep_lr/lr_2.0e-5` |
| 2.2e-05 | 97.53% | `checkpoints_sweep_lr/lr_2.2e-5` |
| 2.4e-05 | 97.53% | `checkpoints_sweep_lr/lr_2.4e-5` |
| 2.6e-05 | 97.66% | `checkpoints_sweep_lr_mixup/lr_2.6e-5_mixup` |
| 3.0e-05 | 95.65% | `checkpoints_sweep_lr/lr_3.0e-5` |
| 4.0e-05 | 96.99% | `checkpoints_swin_anti_overfit/default` |
| 6.0e-06 | 96.96% | `checkpoints_sweep_lr/lr_6.0e-6` |
| 8.0e-06 | 97.36% | `checkpoints_sweep_lr/lr_8.0e-6` |


---

## 🔍 筛选指南

### 按实验类型


- **LR Sweep (无 MixUp)**: 11 个实验
  - 最佳: 97.56%
  
- **LR Sweep (有 MixUp)**: 2 个实验
  - 最佳: 97.66%

- **Anti-Overfit**: 2 个实验
  - 最佳: 97.26%

- **Optimized**: 0 个实验
  - 最佳: 0.00%

---

**总结**: 基于 Val Acc，推荐使用 **排名前 3** 的模型进行推理或进一步训练。
