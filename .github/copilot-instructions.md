# Flower Recognition - Copilot Instructions

## Project Context
Competition-focused deep learning system for 100-class flower classification using PyTorch, timm models, and Hydra configuration.

**Competition**: 2025年第七届全国高校计算机能力挑战赛 - 花卉识别AI挑战赛

**Mandatory Constraints**:
- Model size: **≤ 500MB** (validated at `train.py` line ~60: `get_model_size_mb()`)
- Inference speed: **≤ 100ms/image** (benchmark with `--benchmark` flag)
- Input resolution: **600×600** pixels (all model configs enforce this)
- Supported models: ConvNeXt (base/tiny/large), EfficientNet, Swin Transformer V2

## Architecture Overview

### Three-Tier Component Design
1. **`datasets/flower_dataset.py`**: `FlowerDataset` class + `build_transforms()` + `create_dataloaders()`
   - Albumentations pipelines driven by YAML config (see `configs/augmentation/`)
   - Returns tuples: `(image, label, image_id)` for tracking during inference
   - Test mode scans directory automatically when `is_test=True`

2. **`models/flower_model.py`**: `FlowerRecognitionModel` wrapper around timm models
   - Factory: `build_model(cfg)` extracts architecture/pretrained/drop_path_rate from config
   - All models initialize with 100 classes + ImageNet pretrained weights (configurable)
   - Size check: `get_model_size_mb()` calculates parameter + buffer memory

3. **`train.py` / `inference.py`**: `Trainer` and `Predictor` orchestrators
   - Trainer: Mixed precision (AMP) with `GradScaler`, early stopping, TensorBoard logging
   - Predictor: Batch inference with optional TTA (horizontal flip averaging)

**Data Flow**: Hydra config → factory functions → Trainer.train_loop() → checkpoint + logs

## Hydra Configuration System (Critical)

### Config Composition (defaults pattern in `config.yaml`)
All settings cascade through Hydra's config groups—CLI overrides take precedence:

```bash
# Train with specific model and augmentation
python train.py model=convnext_tiny augmentation=medium training.epochs=100

# Check what config will be used
python train.py --cfg job
```

**Config file map**:
- `configs/config.yaml`: Root defaults, data/output paths, device, seed
- `configs/model/*.yaml`: Architecture, pretrained status, drop_path_rate (0.0-0.3 range)
- `configs/training/default.yaml`: AdamW optimizer (lr=1e-4, weight_decay=0.05), cosine scheduler, label_smoothing=0.1, early_stopping patience=10
- `configs/augmentation/{light,medium,strong,ultra_strong}.yaml`: Albumentations stages (list of dicts)
- `configs/dataset/*.yaml`: CSV paths, val_split=0.2, batch_size=32, num_workers

**Design principle**: Never hardcode hyperparameters—add config files or override via CLI.

## Critical Workflows

### Training (with model validation)
```bash
# Step 1: Verify model size before training
python train.py  # First log shows "Model size: XX.XX MB"

# Step 2: Monitor training live
tensorboard --logdir results/logs &

# Step 3: Results saved to
# results/checkpoints/best_model.pt (lowest val_loss)
# results/checkpoints/checkpoint_epoch_N.pt (periodic saves)
```

### Inference & Submission
```bash
# Generate predictions (returns CSV: image_id, label)
python inference.py --checkpoint results/checkpoints/best_model.pt --output predictions.csv

# With test-time augmentation (4x flipped predictions averaged)
python inference.py --checkpoint results/checkpoints/best_model.pt --output predictions.csv --tta

# Verify inference speed meets <100ms requirement
python inference.py --checkpoint results/checkpoints/best_model.pt --benchmark
```

### Checkpoint Structure
```python
torch.load(checkpoint_path) returns {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'epoch': int,
    'best_val_acc': float,
    'config': OmegaConf.to_container(cfg)
}
```

## Common Patterns & Implementation Details

### Loss Functions (in `models/losses.py`)
- **Default**: `nn.CrossEntropyLoss` with label_smoothing from config
- **Focal Loss**: For imbalanced classes, alpha=0.25, gamma=2.0
- Factory: `get_loss_function(cfg)` returns configured loss based on `cfg.training.loss_type`

### Mixed Precision Training (AMP)
```python
# Pattern in Trainer.train_step() at line ~160:
with autocast():
    outputs = self.model(images)
    loss = self.criterion(outputs, labels)
self.scaler.scale(loss).backward()
# Gradient clipping applied before step()
```

### Inference with Image ID Tracking
```python
# FlowerDataset returns (image, label, image_id)
# Predictor.predict() preserves image_ids for CSV output:
# Column 1: image_id (from dataset third return value)
# Column 2: label (argmax of logits)
```

## File References

| File | Purpose |
|------|---------|
| `train.py` (Trainer class) | Main training loop, checkpointing, TensorBoard setup |
| `inference.py` (Predictor class) | Batch prediction, TTA, speed benchmarking |
| `models/__init__.py` | `build_model()`, `get_model_size_mb()`, `get_loss_function()` |
| `datasets/__init__.py` | `create_dataloaders()`, `build_transforms()` |
| `verify_setup.py` | Validate data directory structure before training |
| `prepare_submission.py` | Package predictions into competition submission format |

## Troubleshooting Checklist

1. **Model exceeds 500MB**: Use `convnext_tiny` (146MB) or reduce `drop_path_rate` from 0.1 → 0.05
2. **CUDA OOM during training**: Reduce `batch_size` (config: `dataset.batch_size=16`), decrease `num_workers`
3. **Validation accuracy not improving**: Try `augmentation=light` or increase `training.epochs`
4. **Inference slow**: Enable AMP in config (`training.use_amp=true`) or use smaller model
5. **CSV encoding errors**: Ensure `pd.read_csv()` and `.to_csv()` use `encoding='utf-8'`
