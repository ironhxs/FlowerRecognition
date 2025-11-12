# Flower Recognition AI Challenge

A comprehensive deep learning solution for the Flower Recognition AI Challenge using state-of-the-art computer vision models.

## 🌟 Features

- **State-of-the-art Models**: ConvNeXt, EfficientNetV2, Swin Transformer V2
- **Advanced Data Augmentation**: Albumentations-based augmentation pipeline
- **Configuration Management**: Hydra for flexible experiment configuration
- **Training Monitoring**: TensorBoard integration for real-time monitoring
- **Progress Tracking**: TQDM progress bars for all operations
- **Easy CLI**: Command-line interface for common tasks
- **Competition Ready**: Meets all competition requirements (model size < 500MB, inference < 100ms)

## 📁 Project Structure

```
FlowerRecognition/
├── configs/                 # Hydra configuration files
│   ├── config.yaml         # Main configuration
│   ├── model/              # Model configurations
│   ├── dataset/            # Dataset configurations
│   ├── training/           # Training configurations
│   └── augmentation/       # Augmentation configurations
├── datasets/               # Dataset module
│   ├── __init__.py
│   └── flower_dataset.py   # Dataset class and utilities
├── models/                 # Model architectures
│   ├── __init__.py
│   └── flower_model.py     # Model definitions
├── cli/                    # Command-line interface
│   ├── __init__.py
│   └── flower_cli.py       # CLI commands
├── docs/                   # Documentation
├── results/                # Training results and checkpoints
├── train.py               # Training script
├── inference.py           # Inference script
└── requirements.txt       # Python dependencies
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ironhxs/FlowerRecognition.git
cd FlowerRecognition

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

Organize your data as follows:

```
data/
├── train.csv              # Training labels (image_id, label)
├── train/                 # Training images
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── test/                  # Test images
    ├── test_001.jpg
    ├── test_002.jpg
    └── ...
```

### Training

```bash
# Train with default configuration (ConvNeXt Base)
python train.py

# Train with specific model
python train.py model=efficientnet_v2_l

# Train with custom settings
python train.py model=swin_transformer_v2 training.epochs=100 dataset.batch_size=16
```

### Inference

```bash
# Generate predictions
python inference.py --checkpoint results/checkpoints/best_model.pt --output predictions.csv

# With test-time augmentation
python inference.py --checkpoint results/checkpoints/best_model.pt --output predictions.csv --tta

# Benchmark inference speed
python inference.py --checkpoint results/checkpoints/best_model.pt --benchmark
```

### Using CLI

```bash
# Show available models
python cli/flower_cli.py models

# Show system information
python cli/flower_cli.py info

# Prepare and validate dataset
python cli/flower_cli.py prepare-data --data-dir ./data

# Train model
python cli/flower_cli.py train --config configs/config.yaml

# Generate predictions
python cli/flower_cli.py predict --checkpoint results/checkpoints/best_model.pt --output predictions.csv
```

## 🎯 Model Selection

### Recommended Models

| Model | Parameters | Accuracy | Speed | Best For |
|-------|-----------|----------|-------|----------|
| ConvNeXt Base | 89M | High | Fast | Balanced performance |
| EfficientNetV2-L | 120M | Very High | Medium | Maximum accuracy |
| Swin Transformer V2 | 88M | Very High | Medium | Latest architecture |
| ConvNeXt Large | 198M | Very High | Medium | Max accuracy with more params |

### Changing Models

Edit `configs/config.yaml` or use command line:

```bash
# Use ConvNeXt Base (default)
python train.py model=convnext_base

# Use EfficientNetV2-L
python train.py model=efficientnet_v2_l

# Use Swin Transformer V2
python train.py model=swin_transformer_v2
```

## 📊 Training Monitoring

View training progress with TensorBoard:

```bash
tensorboard --logdir results/logs
```

This will show:
- Training and validation loss
- Training and validation accuracy
- Learning rate schedule
- Model architecture

## ⚙️ Configuration

### Main Configuration (`configs/config.yaml`)

```yaml
defaults:
  - model: convnext_base
  - dataset: flower100
  - training: default
  - augmentation: strong

project_name: flower_recognition
experiment_name: baseline
seed: 42

data_dir: ./data
output_dir: ./results
device: cuda
num_workers: 4
```

### Model Configuration

Create custom model configs in `configs/model/`:

```yaml
name: my_model
architecture: convnext_base
pretrained: true
num_classes: 100
drop_path_rate: 0.1
input_size: 600
```

### Training Configuration

Customize training in `configs/training/`:

```yaml
optimizer:
  name: adamw
  lr: 1e-4
  weight_decay: 0.05

scheduler:
  name: cosine
  warmup_epochs: 5

epochs: 50
use_amp: true
label_smoothing: 0.1
```

## 🎨 Data Augmentation

The project uses Albumentations for advanced augmentation:

- Random resized crop
- Horizontal and vertical flips
- Rotation and scaling
- Color jittering
- Gaussian blur and noise
- Coarse dropout (Cutout)

Customize augmentation in `configs/augmentation/strong.yaml`.

## 📈 Competition Submission

### Generate Submission File

```bash
python inference.py \
    --checkpoint results/checkpoints/best_model.pt \
    --output submission.csv \
    --tta
```

The output CSV will be in the required format:

```csv
image_id,label
test_001.jpg,42
test_002.jpg,15
...
```

### Verify Requirements

```bash
# Check model size
python -c "import torch; checkpoint = torch.load('results/checkpoints/best_model.pt'); print(f'Model size: {sum(p.numel() * p.element_size() for p in checkpoint[\"model_state_dict\"].values()) / 1024 / 1024:.2f} MB')"

# Benchmark inference speed
python inference.py --checkpoint results/checkpoints/best_model.pt --benchmark
```

## 🔧 Advanced Usage

### Ensemble Predictions

```python
from inference import Predictor
import pandas as pd

# Load multiple models
predictors = [
    Predictor(cfg, 'checkpoint1.pt'),
    Predictor(cfg, 'checkpoint2.pt'),
    Predictor(cfg, 'checkpoint3.pt')
]

# Generate predictions from each model
all_predictions = []
for predictor in predictors:
    _, preds, probs = predictor.predict(test_loader)
    all_predictions.append(probs)

# Ensemble (average probabilities)
ensemble_probs = np.mean(all_predictions, axis=0)
ensemble_preds = np.argmax(ensemble_probs, axis=1)
```

### Custom Data Augmentation

Create a new augmentation config:

```yaml
# configs/augmentation/light.yaml
train:
  - RandomResizedCrop:
      size: 600
      scale: [0.8, 1.0]
  - RandomHorizontalFlip:
      p: 0.5
  - Normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]

val:
  - Resize:
      height: 600
      width: 600
  - Normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
```

Then use it:

```bash
python train.py augmentation=light
```

## 📝 Technical Report

For the competition technical report, include:

1. **Model Architecture**: Describe the chosen model and why
2. **Training Strategy**: Explain hyperparameters, augmentation, optimization
3. **Experimental Results**: Show validation accuracy, confusion matrix, etc.
4. **Innovation**: Highlight any novel approaches or improvements
5. **Code and Reproducibility**: Reference this GitHub repository

See `docs/technical_report_template.md` for a detailed template.

## 🐛 Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
python train.py dataset.batch_size=16
```

### Slow Training

Enable mixed precision (should be on by default):
```bash
python train.py training.use_amp=true
```

Reduce number of workers if CPU is bottleneck:
```bash
python train.py num_workers=2
```

### Model Too Large

Use a smaller model:
```bash
python train.py model=convnext_base  # Instead of convnext_large
```

## 📚 References

- **ConvNeXt**: [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
- **EfficientNetV2**: [Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)
- **Swin Transformer**: [Hierarchical Vision Transformer](https://arxiv.org/abs/2103.14030)
- **PyTorch Image Models (timm)**: [GitHub Repository](https://github.com/huggingface/pytorch-image-models)
- **Albumentations**: [Fast Image Augmentation Library](https://github.com/albumentations-team/albumentations)

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📮 Contact

For questions or issues, please open an issue on GitHub or contact the repository owner.

---

**Good luck with the competition! 🌸🏆**