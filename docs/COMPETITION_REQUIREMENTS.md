# 🌸 花卉识别 AI 挑战赛 - 项目需求文档

## 📋 比赛信息
- **赛事名称**: 2025年第七届全国高校计算机能力挑战赛 - 大数据挑战赛
- **赛题**: 花卉识别 AI 挑战赛
- **任务**: 构建高精度花卉识别模型，识别 100 种花卉

---

## 🎯 核心任务要求

### 1. 识别任务
- ✅ **类别数量**: 100 种花卉（包含稀有品种和形态相似花卉）
- ✅ **图片尺寸**: 统一 600×600 像素
- ✅ **图片格式**: JPG/PNG
- ✅ **标注格式**: CSV 文件（图片路径 + 类别标签）

### 2. 数据集
#### 训练集（主办方提供）
- ✅ 100 个类别
- ✅ 每个类别 100-150 张高质量图片
- ✅ **实际数据**: 19,928 张图片，每类约 199 张（已准备完成）

#### 测试集（服务器端）
- 📌 100 个类别
- 📌 每个类别 50 张图片
- 📌 不提供下载，保存在服务器后台用于最终评测

---

## ⚙️ 技术约束（必须满足）

### 硬性限制
| 约束项 | 要求 | 当前状态 |
|--------|------|----------|
| 模型大小 | **≤ 500MB** | ✅ 需在训练时验证 |
| 推理速度 | **≤ 100ms/图片** | ✅ 需在推理时测试 |
| 模型参数量 | **< 10B** | ✅ 所有模型均符合 |
| 图片尺寸 | **600×600** | ✅ 已配置 |

### 环境要求
- ✅ Python 3.8+（当前: 3.10）
- ✅ PyTorch 1.9+（当前: 2.5.1）
- ✅ CUDA 11.3+ / 12.1（当前: 12.1）
- ✅ UTF-8 编码

---

## 📦 提交要求

### 1. 提交格式
必须为 **ZIP 压缩包**，包含以下结构：

```
submission.zip
├── code/                    # 源代码
│   ├── train.py            # 训练脚本
│   ├── inference.py        # 推理脚本
│   ├── datasets/           # 数据处理模块
│   ├── models/             # 模型定义
│   └── configs/            # 配置文件
├── model/                   # 模型文件
│   └── best_model.pt       # 最佳模型权重
├── result/                  # 预测结果
│   └── predictions.csv     # 预测文件（UTF-8）
└── requirements.txt         # 依赖列表
```

✅ **已实现**: `prepare_submission.py` 脚本可自动生成此结构

### 2. 预测文件格式
CSV 文件，UTF-8 编码，格式：
```csv
image_id,label
test_001.jpg,42
test_002.jpg,15
...
```

✅ **已实现**: `inference.py` 自动生成符合格式的 CSV

---

## 📊 评审标准

### 技术评审（70%）
1. **模型性能（50%）**
   - 测试集准确率（主要指标）
   - 模型鲁棒性
   - 📌 目标：准确率尽可能高

2. **技术创新（20%）**
   - 算法改进
   - 技术方案创新性
   - 📌 可考虑：集成学习、注意力机制、新型数据增强等

### 报告评审（30%）
1. **技术报告（20%）**
   - ✅ 已提供模板：`docs/technical_report_template.md`
   - 必须包含：
     - 模型架构描述
     - 训练策略（数据预处理、增强、优化器）
     - 实验结果（验证集性能）
     - 创新点说明
   - 📌 页数限制：≤ 10 页

2. **实验分析（10%）**
   - 实验设计合理性
   - 结果分析深度

---

## ✅ 当前项目符合性检查

### 已满足的要求
- ✅ 数据格式：600×600，JPG 格式
- ✅ 数据准备：19,928 张训练图片，100 类
- ✅ 标签格式：CSV（image_id, label）
- ✅ Python 版本：3.10（≥3.8）
- ✅ 深度学习框架：PyTorch 2.5.1（≥1.9）
- ✅ CUDA 版本：12.1（≥11.3）
- ✅ 模型参数量：所有模型 < 10B
- ✅ 提交脚本：`prepare_submission.py`
- ✅ UTF-8 编码：已在代码中强制使用

### 需要在训练/测试时验证
- ⚠️ 模型大小 ≤ 500MB（训练时自动检查）
- ⚠️ 推理速度 ≤ 100ms/图片（使用 `--benchmark` 测试）

### 可选增强项
- 🔄 数据增强策略优化（已有 strong/medium/light 三级）
- 🔄 模型集成（ensemble）
- 🔄 测试时增强（TTA）已实现
- 🔄 超参数调优

---

## 🚀 推荐训练策略

### 阶段一：基线模型（快速验证）
```bash
# ConvNeXt Tiny - 快速测试流程
python train.py model=convnext_tiny augmentation=medium training.epochs=30
```

### 阶段二：性能优化
```bash
# ConvNeXt Base - 平衡性能
python train.py model=convnext_base augmentation=strong training.epochs=50

# EfficientNetV2-L - 最高精度
python train.py model=efficientnet_v2_l augmentation=strong training.epochs=60
```

### 阶段三：模型集成
- 训练多个不同架构的模型
- 使用 ensemble 策略组合预测结果
- 使用 TTA 提升单模型性能

---

## 📝 技术报告准备清单

使用模板：`docs/technical_report_template.md`

### 必须包含的内容
- [ ] 1. 模型架构详细描述
  - [ ] 选择的基础架构（ConvNeXt/EfficientNet 等）
  - [ ] 网络结构说明
  - [ ] 参数量统计
  
- [ ] 2. 训练策略
  - [ ] 数据预处理方法
  - [ ] 数据增强技术（Albumentations）
  - [ ] 优化器选择（AdamW）
  - [ ] 学习率调度策略（Cosine + Warmup）
  - [ ] 正则化技术（Label Smoothing, Drop Path）
  
- [ ] 3. 实验结果
  - [ ] 验证集准确率曲线
  - [ ] 混淆矩阵
  - [ ] 每类别准确率分析
  - [ ] 模型大小和推理速度
  
- [ ] 4. 创新点
  - [ ] 使用的特殊技术
  - [ ] 算法改进
  - [ ] 独特的训练策略

---

## 🛠️ 快速操作指南

### 1. 环境激活
```bash
conda activate flower
```

### 2. 开始训练
```bash
# 使用快速启动脚本
python quickstart.py

# 或直接训练
python train.py
```

### 3. 监控训练
```bash
# 新终端
tensorboard --logdir results/logs
# 访问 http://localhost:6006
```

### 4. 生成预测
```bash
python inference.py --checkpoint results/checkpoints/best_model.pt --output predictions.csv --tta
```

### 5. 验证约束
```bash
# 检查模型大小（训练时自动显示）
# 检查推理速度
python inference.py --checkpoint results/checkpoints/best_model.pt --benchmark
```

### 6. 准备提交
```bash
python prepare_submission.py --checkpoint results/checkpoints/best_model.pt --predictions predictions.csv
```

---

## 📌 重要注意事项

1. **模型大小控制**
   - 训练时会自动显示模型大小
   - 如果超过 500MB，选择更小的模型或降低 drop_path_rate

2. **推理速度优化**
   - 使用混合精度（AMP）
   - 考虑使用 `torch.compile()`（PyTorch 2.0+）
   - 测试时使用 `--benchmark` 参数

3. **数据使用规则**
   - ✅ 允许：主办方数据 + 数据增强 + 预训练模型
   - ❌ 不允许：外部标注数据

4. **UTF-8 编码**
   - 所有 CSV 文件必须使用 UTF-8 编码
   - `inference.py` 已强制使用 UTF-8

---

## 📚 相关文档

- **快速开始**: `docs/QUICKSTART.md`
- **使用示例**: `docs/USAGE_EXAMPLES.md`
- **技术报告模板**: `docs/technical_report_template.md`
- **项目总结**: `docs/PROJECT_SUMMARY.md`
- **Copilot 指南**: `.github/copilot-instructions.md`

---

## ✨ 成功标准

- ✅ 模型准确率尽可能高
- ✅ 模型大小 < 500MB
- ✅ 推理速度 < 100ms/图片
- ✅ 提交格式正确（ZIP + UTF-8）
- ✅ 技术报告完整（≤10 页）
- ✅ 代码可复现

---

**祝比赛顺利！Good luck! 🌸🏆**
