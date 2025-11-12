# FlowerRecognition | 花卉识别

这是参加花卉识别AI挑战赛的代码库。本项目使用深度学习技术进行花卉图像分类。

This is a code repository for participating in a flower recognition AI challenge. This project uses deep learning techniques for flower image classification.

## 项目结构 | Project Structure

```
FlowerRecognition/
├── data/                   # 数据目录 | Data directory
│   ├── train/             # 训练数据 | Training data
│   ├── val/               # 验证数据 | Validation data
│   └── test/              # 测试数据 | Test data
├── models/                # 模型保存目录 | Model checkpoints directory
├── src/                   # 源代码 | Source code
│   ├── __init__.py
│   ├── dataset.py         # 数据加载 | Data loading
│   ├── model.py           # 模型定义 | Model definitions
│   └── utils.py           # 工具函数 | Utility functions
├── notebooks/             # Jupyter notebooks
├── train.py               # 训练脚本 | Training script
├── predict.py             # 预测脚本 | Inference script
├── evaluate.py            # 评估脚本 | Evaluation script
├── config.yaml            # 配置文件 | Configuration file
└── requirements.txt       # 依赖库 | Dependencies
```

## 功能特性 | Features

- 🌸 支持多种深度学习模型（ResNet, EfficientNet, VGG, DenseNet）
- 📊 完整的训练、验证和测试流程
- 🔄 数据增强技术提升模型泛化能力
- 📈 TensorBoard 可视化训练过程
- 💾 自动保存最佳模型
- 🎯 详细的评估指标和混淆矩阵

- 🌸 Support for multiple deep learning models (ResNet, EfficientNet, VGG, DenseNet)
- 📊 Complete training, validation, and testing pipeline
- 🔄 Data augmentation techniques for better generalization
- 📈 TensorBoard visualization for training process
- 💾 Automatic best model saving
- 🎯 Detailed evaluation metrics and confusion matrix

## 安装 | Installation

### 环境要求 | Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (可选，用于GPU加速 | Optional, for GPU acceleration)

### 安装步骤 | Installation Steps

1. 克隆仓库 | Clone the repository:
```bash
git clone https://github.com/ironhxs/FlowerRecognition.git
cd FlowerRecognition
```

2. 安装依赖 | Install dependencies:
```bash
pip install -r requirements.txt
```

## 数据准备 | Data Preparation

数据应按以下结构组织 | Data should be organized in the following structure:

```
data/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── class2/
│   │   └── ...
│   └── ...
├── val/
│   └── (same structure as train)
└── test/
    └── (same structure as train)
```

每个类别的图像应放在对应的子目录中。| Images for each class should be placed in corresponding subdirectories.

## 使用方法 | Usage

### 1. 配置 | Configuration

编辑 `config.yaml` 文件以设置模型参数、训练参数等。

Edit `config.yaml` to set model parameters, training parameters, etc.

### 2. 训练模型 | Train Model

```bash
python train.py --config config.yaml
```

从检查点继续训练 | Resume training from a checkpoint:
```bash
python train.py --config config.yaml --resume models/checkpoint_epoch_10.pth
```

### 3. 评估模型 | Evaluate Model

```bash
python evaluate.py --checkpoint models/best_model.pth --config config.yaml
```

生成混淆矩阵 | Generate confusion matrix:
```bash
python evaluate.py --checkpoint models/best_model.pth --config config.yaml --plot-cm
```

### 4. 预测 | Prediction

单张图片预测 | Predict a single image:
```bash
python predict.py --image path/to/image.jpg --checkpoint models/best_model.pth --config config.yaml
```

可视化预测结果 | Visualize predictions:
```bash
python predict.py --image path/to/image.jpg --checkpoint models/best_model.pth --config config.yaml --visualize
```

## 支持的模型 | Supported Models

本项目支持以下预训练模型 | This project supports the following pretrained models:

- ResNet (resnet18, resnet34, resnet50, resnet101)
- EfficientNet (efficientnet_b0, efficientnet_b3)
- VGG (vgg16)
- DenseNet (densenet121)

在 `config.yaml` 中修改 `model.name` 来选择不同的模型。

Modify `model.name` in `config.yaml` to select different models.

## 训练监控 | Training Monitoring

使用 TensorBoard 监控训练过程 | Use TensorBoard to monitor training:

```bash
tensorboard --logdir logs
```

然后在浏览器中打开 `http://localhost:6006`

Then open `http://localhost:6006` in your browser.

## 配置说明 | Configuration Options

主要配置选项 | Main configuration options:

- `model.name`: 模型架构 | Model architecture
- `model.num_classes`: 类别数量 | Number of classes
- `training.batch_size`: 批次大小 | Batch size
- `training.num_epochs`: 训练轮数 | Number of epochs
- `training.learning_rate`: 学习率 | Learning rate
- `data.image_size`: 图像大小 | Image size

## 性能优化建议 | Performance Optimization Tips

1. 使用更大的批次大小（如果GPU内存允许）| Use larger batch size (if GPU memory allows)
2. 尝试不同的学习率 | Try different learning rates
3. 使用学习率调度器 | Use learning rate scheduler
4. 增加数据增强 | Increase data augmentation
5. 尝试不同的模型架构 | Try different model architectures

## 常见问题 | FAQ

### Q: 如何处理类别不平衡？| How to handle class imbalance?
A: 可以使用加权损失函数或过采样技术。| Use weighted loss function or oversampling techniques.

### Q: 如何提高模型准确率？| How to improve model accuracy?
A: 尝试更深的网络、更多的训练数据、数据增强和迁移学习。| Try deeper networks, more training data, data augmentation, and transfer learning.

### Q: GPU内存不足怎么办？| What to do with insufficient GPU memory?
A: 减小批次大小或使用更小的模型。| Reduce batch size or use a smaller model.

## 贡献 | Contributing

欢迎提交问题和拉取请求！| Issues and pull requests are welcome!

## 许可证 | License

MIT License

## 联系方式 | Contact

如有问题，请提交 Issue。| For questions, please submit an Issue.

---

祝你在花卉识别AI挑战赛中取得好成绩！🌺

Good luck in the Flower Recognition AI Challenge! 🌺