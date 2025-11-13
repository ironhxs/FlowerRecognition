# Checkpoint权重提取指南

## 背景

PyTorch训练保存的完整checkpoint包含:
- `model_state_dict`: 模型权重 (~332MB)
- `optimizer_state_dict`: 优化器状态 (~664MB, AdamW为每个参数存储momentum和variance)
- `scheduler_state_dict`: 学习率调度器状态
- `epoch`, `best_val_acc`等元数据

**问题**: 推理时不需要optimizer/scheduler,完整checkpoint浪费66%存储空间。

## 解决方案

### 方法1: 使用提供的脚本

```bash
# 提取权重+配置(推荐用于推理)
python extract_weights.py results/checkpoints/best_model.pt

# 仅提取权重(用于模型共享/迁移学习)
python extract_weights.py results/checkpoints/best_model.pt --weights-only

# 自定义输出路径
python extract_weights.py results/checkpoints/best_model.pt -o my_weights.pt

# 提取后删除原文件(谨慎使用!)
python extract_weights.py results/checkpoints/best_model.pt --delete-original
```

### 方法2: Python代码

```python
import torch

# 加载完整checkpoint
checkpoint = torch.load('best_model.pt', map_location='cpu', weights_only=False)

# 选项A: 保存纯权重(最小体积)
torch.save(checkpoint['model_state_dict'], 'weights_only.pt')

# 选项B: 保存权重+元数据(推荐)
inference_checkpoint = {
    'model_state_dict': checkpoint['model_state_dict'],
    'config': checkpoint['config'],
    'epoch': checkpoint['epoch'],
    'best_val_acc': checkpoint['best_val_acc']
}
torch.save(inference_checkpoint, 'inference.pt')
```

## 文件大小对比

| 类型 | 大小 | 用途 |
|------|------|------|
| 完整checkpoint | 997MB | 恢复训练 |
| 推理checkpoint | 332MB | 模型推理 |
| 纯权重 | 332MB | 模型共享/迁移学习 |

**节省空间: 66.7%** (665MB)

## 加载权重

### 推理时加载

```python
# inference.py 已支持两种格式
python inference.py \
    --checkpoint results/checkpoints/best_model_inference.pt \
    --output predictions.csv
```

### 代码中加载

```python
from models import build_model
from omegaconf import OmegaConf
import torch

# 方式1: 加载推理checkpoint
checkpoint = torch.load('best_model_inference.pt', weights_only=False)
config = OmegaConf.create(checkpoint['config'])
model = build_model(config)
model.load_state_dict(checkpoint['model_state_dict'])

# 方式2: 加载纯权重
weights = torch.load('weights_only.pt', weights_only=False)
model = build_model(config)  # 需要单独提供config
model.load_state_dict(weights)
```

## 恢复训练

如果需要恢复训练,**必须使用完整checkpoint**:

```python
# train.py中的示例
checkpoint = torch.load('best_model.pt', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

## 最佳实践

1. **训练期间**: 保存完整checkpoint
   ```python
   torch.save({
       'epoch': epoch,
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict(),
       'scheduler_state_dict': scheduler.state_dict(),
       'best_val_acc': best_val_acc,
       'config': OmegaConf.to_container(cfg)
   }, 'checkpoint.pt')
   ```

2. **训练完成**: 提取推理checkpoint
   ```bash
   python extract_weights.py results/checkpoints/best_model.pt
   ```

3. **提交比赛/部署**: 使用推理checkpoint (332MB)

4. **备份**: 保留一份完整checkpoint用于可能的继续训练

## PyTorch官方支持

PyTorch 2.6+引入了`weights_only`参数提升安全性:

```python
# 安全加载(仅加载张量,不执行任意代码)
torch.load('checkpoint.pt', weights_only=True)  # 新版默认

# 兼容旧格式(如包含OmegaConf等自定义对象)
torch.load('checkpoint.pt', weights_only=False)
```

**注意**: 我们的checkpoint包含OmegaConf配置对象,需要`weights_only=False`。

## 比赛提交

花卉识别比赛要求:
- ✅ 模型大小 < 500MB: 推理checkpoint 332MB符合要求
- ✅ 推理速度 < 100ms/图: 不受checkpoint格式影响
- ✅ 可复现性: 推理checkpoint包含完整config

```bash
# 准备提交
cp results/checkpoints/best_model_inference.pt submission/model.pt
python inference.py --checkpoint submission/model.pt --output submission/predictions.csv
```
