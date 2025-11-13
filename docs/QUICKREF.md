# 🌸 花卉识别 AI 挑战赛 - 快速参考

## 🎯 比赛核心约束
- ✅ 模型大小 ≤ 500MB
- ✅ 推理速度 ≤ 100ms/图片  
- ✅ 输入尺寸 600×600
- ✅ 100 类花卉识别
- ✅ UTF-8 编码

## 🚀 快速开始

### 1. 激活环境
```bash
conda activate flower
```

### 2. 检查比赛要求
```bash
python check_competition.py
```

### 3. 快速训练测试（5分钟）
```bash
python train.py model=convnext_tiny training.epochs=5 dataset.batch_size=16
```

### 4. 正式训练
```bash
# 方式1: 使用快速启动脚本（推荐）
python quickstart.py

# 方式2: 直接训练
python train.py  # 默认使用 ConvNeXt Base
```

### 5. 监控训练
```bash
# 新开终端
conda activate flower
tensorboard --logdir results/logs
# 访问 http://localhost:6006
```

### 6. 生成预测
```bash
python inference.py \
    --checkpoint results/checkpoints/best_model.pt \
    --output predictions.csv \
    --tta
```

### 7. 验证约束
```bash
# 模型大小会在训练时自动显示
# 检查推理速度
python inference.py --checkpoint results/checkpoints/best_model.pt --benchmark
```

### 8. 准备提交
```bash
python prepare_submission.py \
    --checkpoint results/checkpoints/best_model.pt \
    --predictions predictions.csv
```

## 📊 推荐训练方案

### 方案A：快速基线（2-3小时）
```bash
python train.py \
    model=convnext_tiny \
    augmentation=medium \
    training.epochs=30
```

### 方案B：平衡性能（6-8小时）
```bash
python train.py \
    model=convnext_base \
    augmentation=strong \
    training.epochs=50
```

### 方案C：最高精度（10-12小时）
```bash
python train.py \
    model=efficientnet_v2_l \
    augmentation=strong \
    training.epochs=60
```

## 📁 关键文件

| 文件 | 说明 |
|------|------|
| `COMPETITION_REQUIREMENTS.md` | 完整比赛需求文档 |
| `check_competition.py` | 比赛要求检查脚本 |
| `quickstart.py` | 交互式快速启动 |
| `train.py` | 训练脚本 |
| `inference.py` | 推理脚本 |
| `prepare_submission.py` | 提交准备脚本 |
| `docs/technical_report_template.md` | 技术报告模板 |

## 🎓 可用模型

| 模型 | 参数量 | 模型大小 | 速度 | 推荐场景 |
|------|--------|----------|------|----------|
| ConvNeXt Tiny | 29M | ~110MB | 快 | 快速测试 |
| ConvNeXt Base | 89M | ~340MB | 中 | 平衡性能 ⭐ |
| ConvNeXt Large | 198M | ~760MB | 慢 | ❌ 超限 |
| EfficientNet B3 | 12M | ~45MB | 快 | 轻量级 |
| EfficientNetV2-L | 120M | ~460MB | 中 | 高精度 ⭐ |
| Swin Transformer V2 | 88M | ~330MB | 中 | 最新架构 |

⭐ = 推荐使用

## ⚠️ 常见问题

### 模型大小超限？
```bash
# 使用更小的模型
python train.py model=convnext_tiny
# 或
python train.py model=efficientnet_b3
```

### 推理速度超时？
- 确保使用 `use_amp=true`（默认开启）
- 减小 batch_size 到 1
- 使用 `torch.compile()`（PyTorch 2.0+）

### CUDA 内存不足？
```bash
python train.py dataset.batch_size=16  # 减小 batch size
```

## 📝 提交检查清单

- [ ] 模型大小 < 500MB
- [ ] 推理速度 < 100ms/图片
- [ ] predictions.csv 使用 UTF-8 编码
- [ ] 提交 ZIP 包含所有必需文件
- [ ] 技术报告完整（≤10页）

## 🔗 相关链接

- **详细需求**: `COMPETITION_REQUIREMENTS.md`
- **快速开始**: `docs/QUICKSTART.md`
- **使用示例**: `docs/USAGE_EXAMPLES.md`
- **技术报告模板**: `docs/technical_report_template.md`
- **完整文档**: `README.md`

---

**祝比赛顺利！Good luck! 🌸🏆**
