# Flower Recognition AI Challenge - Copilot Instructions

## Project Overview
Competition-focused deep learning system for 100-class flower recognition using PyTorch, timm models, and Hydra configuration. Built for Windows with PowerShell.

**Competition**: 2025年第七届全国高校计算机能力挑战赛 - 花卉识别AI挑战赛

**Critical Constraints**:
- Model size: **≤ 500MB** (checked automatically during training)
- Inference speed: **≤ 100ms/image** (test with `--benchmark`)
- Input size: **600×600** pixels (configured in all models)
- Model parameters: **< 10B** (all supported models comply)
- Encoding: **UTF-8** for all CSV files
- Submission format: **ZIP** with specific structure (use `prepare_submission.py`)

## Architecture & Data Flow

### Three-Module Design
1. **`datasets/flower_dataset.py`** - Handles data loading, Albumentations transforms, and train/val split
2. **`models/flower_model.py`** - timm-based wrapper (`FlowerRecognitionModel`) supporting 6 architectures
3. **`train.py`** / **`inference.py`** - Training loop with AMP, TensorBoard, checkpointing / Batch prediction with optional TTA

**Key Flow**: Hydra config → `build_model()` + `create_dataloaders()` → `Trainer` class → saves to `results/checkpoints/` + logs to `results/logs/`

## Critical Hydra Configuration System

### Config Composition Pattern
```bash
# Configs live in configs/ with defaults pattern in config.yaml
python train.py model=efficientnet_v2_l augmentation=medium training.epochs=100
```

**Key Files**:
- `configs/config.yaml` - defaults list, paths (`data_dir`, `checkpoint_dir`), device settings
- `configs/model/*.yaml` - architecture name, pretrained flag, drop_path_rate, input_size
- `configs/training/default.yaml` - optimizer (AdamW), scheduler (cosine with warmup), AMP, label_smoothing
- `configs/augmentation/*.yaml` - Albumentations pipelines (strong/medium/light) with ImageNet normalization

**Don't**: Hardcode paths or hyperparameters. **Do**: Add overrides via CLI or new YAML files.

## Development Workflows

### Training Workflow
```powershell
# Always check model size first
python train.py  # Logs "Model size: XX.XX MB" - must be <500MB

# Monitor with TensorBoard (runs in background)
tensorboard --logdir results/logs

# Resume from checkpoint (edit train.py to load checkpoint)
# Checkpoint contains: model_state_dict, optimizer_state_dict, epoch, best_val_acc
```

### Inference & Submission
```powershell
# Standard prediction
python inference.py --checkpoint results/checkpoints/best_model.pt --output predictions.csv

# With TTA (horizontal flip averaging)
python inference.py --checkpoint results/checkpoints/best_model.pt --output predictions.csv --tta

# Benchmark speed (required for competition)
python inference.py --checkpoint results/checkpoints/best_model.pt --benchmark
```

**Output format requirement**: CSV with `image_id,label` columns, UTF-8 encoding (see `inference.py` line ~200).

### CLI Alternative
```powershell
python cli/flower_cli.py models      # List all 6 models with stats
python cli/flower_cli.py train       # Wrapper for train.py
python cli/flower_cli.py predict --checkpoint <path> --output <csv> --tta
```

## Project-Specific Conventions

### Trainer Class Pattern (`train.py`)
- Uses **mixed precision (AMP)** by default (`GradScaler` + `autocast` context)
- **Early stopping** based on `val_accuracy` with patience from config
- Saves checkpoints to `cfg.checkpoint_dir` with `best_model.pt` + `checkpoint_epoch_N.pt`
- **TensorBoard logging** of loss, accuracy, LR every epoch

### Dataset Convention (`datasets/flower_dataset.py`)
- Returns `(image, label, image_id)` - third item for tracking during inference
- **Albumentations** transforms are config-driven (list of dicts in YAML)
- `build_transforms()` converts YAML config to `A.Compose` pipeline
- Test set uses `is_test=True` flag (label=-1, scans directory for images)

### Model Registry Pattern (`models/flower_model.py`)
```python
# All models use FlowerRecognitionModel wrapper
model = FlowerRecognitionModel(
    architecture='convnext_base',  # timm model name
    num_classes=100,               # Fixed for competition
    pretrained=True,               # ImageNet weights
    drop_path_rate=0.1             # Stochastic depth
)
# Access underlying timm model: model.model
```

**Supported architectures**: `convnext_base|tiny|large`, `efficientnet_b3`, `efficientnetv2_rw_l`, `swin_base_patch4_window7_224`

### Checkpoint Structure
```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'epoch': epoch,
    'best_val_acc': best_val_acc,
    'config': OmegaConf.to_container(cfg)
}
```

## Common Pitfalls

1. **Model size exceeds 500MB**: Switch to smaller model (`convnext_base` → `convnext_tiny`) or reduce `drop_path_rate`
2. **CUDA OOM**: Reduce `dataset.batch_size` in config (default 32) or `num_workers`
3. **Slow inference**: Ensure `use_amp=true` in training config, consider `torch.compile()` in PyTorch 2.0+
4. **TensorBoard not updating**: Check `log_dir` path in config, ensure SummaryWriter writes every epoch
5. **Augmentation too weak/strong**: Use `augmentation=medium` or `augmentation=light` for large datasets

## Integration Points

- **External**: timm library for models (`timm.create_model()`), Albumentations for augmentation
- **Configuration**: Hydra resolves `${output_dir}` interpolations in config.yaml
- **Logging**: TensorBoard writes to `results/logs/`, accessed via `tensorboard --logdir`
- **Data format**: Expects `data/train.csv` with `image_id,label` columns, `data/train/*.jpg` images
- **Test data**: `data/test/*.jpg` with no CSV (inference scans directory)

## Testing & Validation

```powershell
# Generate sample data for testing
python generate_sample_data.py

# Verify setup before training
python verify_setup.py

# Evaluate model on validation set
python evaluate.py --checkpoint results/checkpoints/best_model.pt

# Prepare competition submission package
python prepare_submission.py --checkpoint <path> --predictions <csv>
```

## Key Files Reference
- **Training entry**: `train.py` (Trainer class, main training loop)
- **Model factory**: `models/__init__.py` (`build_model()`, `get_model_size_mb()`)
- **Dataset factory**: `datasets/__init__.py` (`create_dataloaders()`)
- **Utils**: `utils.py` (plotting, confusion matrix, seed setting)
- **Competition prep**: `prepare_submission.py` (creates submission.zip with required structure)
