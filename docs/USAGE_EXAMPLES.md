# Usage Examples for Flower Recognition

This document provides practical examples for using the Flower Recognition system.

## Table of Contents
1. [Basic Training](#basic-training)
2. [Advanced Training](#advanced-training)
3. [Model Evaluation](#model-evaluation)
4. [Inference](#inference)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Ensemble Methods](#ensemble-methods)

---

## Basic Training

### Quick Start with Default Settings
```bash
python train.py
```

### Train with Different Models
```bash
# Fast and efficient (recommended for quick experiments)
python train.py model=convnext_tiny

# Balanced performance (recommended baseline)
python train.py model=convnext_base

# High accuracy
python train.py model=efficientnet_v2_l

# Latest architecture
python train.py model=swin_transformer_v2

# Maximum accuracy (if size permits)
python train.py model=convnext_large
```

### Train with Different Augmentation Strategies
```bash
# Light augmentation (faster training)
python train.py augmentation=light

# Medium augmentation (balanced)
python train.py augmentation=medium

# Strong augmentation (best for small datasets)
python train.py augmentation=strong
```

---

## Advanced Training

### Custom Learning Rate
```bash
python train.py training.optimizer.lr=2e-4
```

### Custom Batch Size
```bash
# Larger batch (requires more GPU memory)
python train.py dataset.batch_size=64

# Smaller batch (less GPU memory)
python train.py dataset.batch_size=16
```

### Longer Training
```bash
python train.py training.epochs=100
```

### Custom Experiment Name
```bash
python train.py experiment_name=my_experiment_v1
```

### Combine Multiple Settings
```bash
python train.py \
    model=convnext_base \
    augmentation=strong \
    training.epochs=80 \
    training.optimizer.lr=1e-4 \
    dataset.batch_size=32 \
    experiment_name=convnext_base_strong_aug
```

### Transfer Learning (Freeze Backbone)
Edit your training script to freeze the backbone:
```python
from models import build_model

model = build_model(cfg)
model.freeze_backbone()  # Freeze all layers except classifier

# Train for a few epochs
# ...

model.unfreeze_all()  # Unfreeze for fine-tuning
# Continue training
```

---

## Model Evaluation

### Evaluate on Validation Set
```bash
python evaluate.py \
    --checkpoint results/checkpoints/best_model.pt \
    --output-dir evaluation_results
```

This generates:
- Confusion matrix
- Classification report
- Per-class accuracy
- Detailed predictions CSV

### Analyze Specific Results
```python
import pandas as pd

# Load detailed results
results = pd.read_csv('evaluation_results/detailed_results.csv')

# Find misclassified samples
errors = results[results['correct'] == False]
print(f"Number of errors: {len(errors)}")

# Most confused classes
confusion = errors.groupby(['true_label', 'predicted_label']).size()
print("Most confused pairs:")
print(confusion.sort_values(ascending=False).head(10))
```

---

## Inference

### Basic Inference
```bash
python inference.py \
    --checkpoint results/checkpoints/best_model.pt \
    --output predictions.csv
```

### Inference with Test-Time Augmentation
```bash
python inference.py \
    --checkpoint results/checkpoints/best_model.pt \
    --output predictions_tta.csv \
    --tta
```

### Benchmark Inference Speed
```bash
python inference.py \
    --checkpoint results/checkpoints/best_model.pt \
    --benchmark
```

### Predict Single Image
```python
from inference import Predictor
from omegaconf import OmegaConf

# Load config
cfg = OmegaConf.load('configs/config.yaml')

# Create predictor
predictor = Predictor(cfg, 'results/checkpoints/best_model.pt')

# Predict single image
pred_class, probabilities = predictor.predict_single_image('path/to/image.jpg')

print(f"Predicted class: {pred_class}")
print(f"Top-5 probabilities: {probabilities.argsort()[-5:][::-1]}")
```

---

## Hyperparameter Tuning

### Grid Search Example
```bash
# Try different learning rates
for lr in 1e-5 5e-5 1e-4 2e-4; do
    python train.py \
        training.optimizer.lr=$lr \
        experiment_name=lr_${lr}
done

# Try different batch sizes
for bs in 16 32 64; do
    python train.py \
        dataset.batch_size=$bs \
        experiment_name=bs_${bs}
done

# Try different models
for model in convnext_tiny convnext_base efficientnet_b3; do
    python train.py \
        model=$model \
        experiment_name=$model
done
```

### Compare Results
```python
import pandas as pd
import os

# Collect results from multiple experiments
results = []
for exp_dir in os.listdir('results/checkpoints'):
    checkpoint_path = f'results/checkpoints/{exp_dir}/best_model.pt'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        results.append({
            'experiment': exp_dir,
            'accuracy': checkpoint.get('best_val_acc', 0),
            'epoch': checkpoint.get('epoch', 0)
        })

df = pd.DataFrame(results)
print(df.sort_values('accuracy', ascending=False))
```

---

## Ensemble Methods

### Train Multiple Models
```bash
# Train different models
python train.py model=convnext_base experiment_name=model1
python train.py model=efficientnet_v2_l experiment_name=model2
python train.py model=swin_transformer_v2 experiment_name=model3

# Train same model with different seeds
python train.py seed=42 experiment_name=seed42
python train.py seed=123 experiment_name=seed123
python train.py seed=456 experiment_name=seed456
```

### Ensemble Predictions
```python
import torch
import numpy as np
import pandas as pd
from inference import Predictor
from omegaconf import OmegaConf

# Load config
cfg = OmegaConf.load('configs/config.yaml')

# Load multiple models
checkpoints = [
    'results/checkpoints/model1/best_model.pt',
    'results/checkpoints/model2/best_model.pt',
    'results/checkpoints/model3/best_model.pt'
]

predictors = [Predictor(cfg, ckpt) for ckpt in checkpoints]

# Get test dataloader
from datasets import FlowerDataset, build_transforms
from torch.utils.data import DataLoader

transform = build_transforms(cfg.augmentation, is_train=False)
test_dataset = FlowerDataset(
    data_dir=cfg.dataset.test_dir,
    transform=transform,
    is_test=True
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Collect predictions from all models
all_predictions = []
for predictor in predictors:
    _, preds, probs = predictor.predict(test_loader)
    all_predictions.append(np.array(probs))

# Ensemble: Average probabilities
ensemble_probs = np.mean(all_predictions, axis=0)
ensemble_preds = np.argmax(ensemble_probs, axis=1)

# Save ensemble predictions
image_ids = [img_id for _, _, img_id in test_dataset]
df = pd.DataFrame({
    'image_id': image_ids,
    'label': ensemble_preds
})
df.to_csv('ensemble_predictions.csv', index=False)
```

### Weighted Ensemble
```python
# If you know validation accuracies
weights = [0.92, 0.94, 0.93]  # Validation accuracies
weights = np.array(weights) / sum(weights)  # Normalize

# Weighted average
ensemble_probs = np.average(all_predictions, axis=0, weights=weights)
ensemble_preds = np.argmax(ensemble_probs, axis=1)
```

---

## Competition Submission

### Complete Workflow
```bash
# 1. Train best model
python train.py \
    model=efficientnet_v2_l \
    augmentation=strong \
    training.epochs=80

# 2. Evaluate on validation set
python evaluate.py \
    --checkpoint results/checkpoints/best_model.pt \
    --output-dir evaluation_results

# 3. Generate predictions with TTA
python inference.py \
    --checkpoint results/checkpoints/best_model.pt \
    --output predictions.csv \
    --tta

# 4. Verify requirements
python inference.py \
    --checkpoint results/checkpoints/best_model.pt \
    --benchmark

# 5. Prepare submission
python prepare_submission.py \
    --checkpoint results/checkpoints/best_model.pt \
    --predictions predictions.csv \
    --output submission.zip
```

---

## Monitoring with TensorBoard

### Start TensorBoard
```bash
tensorboard --logdir results/logs
```

### View in Browser
Open http://localhost:6006

### Monitor Multiple Experiments
```bash
tensorboard --logdir results/logs --reload_interval 5
```

---

## Debugging Tips

### Check Data Loading
```python
from datasets import create_dataloaders
from omegaconf import OmegaConf

cfg = OmegaConf.load('configs/config.yaml')
train_loader, val_loader, _ = create_dataloaders(cfg)

# Check batch
images, labels, image_ids = next(iter(train_loader))
print(f"Batch shape: {images.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Label range: {labels.min()}-{labels.max()}")
```

### Visualize Augmentations
```python
import matplotlib.pyplot as plt
from datasets import FlowerDataset, build_transforms
from PIL import Image

# Load dataset with augmentations
cfg = OmegaConf.load('configs/config.yaml')
transform = build_transforms(cfg.augmentation, is_train=True)

# Load and augment same image multiple times
image_path = 'data/train/image_001.jpg'
image = Image.open(image_path).convert('RGB')

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, ax in enumerate(axes.flat):
    import numpy as np
    img_array = np.array(image)
    augmented = transform(image=img_array)['image']
    # Convert tensor to displayable format
    img_display = augmented.permute(1, 2, 0).numpy()
    img_display = (img_display * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406])
    img_display = np.clip(img_display, 0, 1)
    ax.imshow(img_display)
    ax.axis('off')
plt.tight_layout()
plt.savefig('augmentation_samples.png')
```

### Test Model Forward Pass
```python
import torch
from models import build_model
from omegaconf import OmegaConf

cfg = OmegaConf.load('configs/config.yaml')
model = build_model(cfg)

# Test forward pass
dummy_input = torch.randn(1, 3, 600, 600)
output = model(dummy_input)
print(f"Output shape: {output.shape}")  # Should be [1, 100]
```

---

## Tips for Best Results

1. **Start with a baseline**: Use ConvNeXt Base with default settings
2. **Use strong augmentation**: Helps with overfitting on smaller datasets
3. **Train longer**: More epochs often improve results (50-100 epochs)
4. **Use TTA**: Test-time augmentation can boost accuracy by 1-2%
5. **Ensemble**: Combine multiple models for best results
6. **Monitor validation**: Watch for overfitting
7. **Save best model**: Based on validation accuracy, not training loss
8. **Verify requirements**: Check model size and inference time before submission
