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
   - `build_transforms()` receives full config dict (not just augmentation config), uses `cfg.augmentation.train|val` keys

2. **`models/flower_model.py`**: `FlowerRecognitionModel` wrapper around timm models
   - Factory: `build_model(cfg)` extracts `cfg.model.architecture`, `pretrained`, `drop_path_rate` from config
   - All models initialize with 100 classes + ImageNet pretrained weights (configurable via `cfg.model.pretrained`)
   - Size check: `get_model_size_mb()` calculates parameter + buffer memory via `torch.cuda.memory_stats()`
   - **Critical**: timm expects `drop_path_rate` (not `drop_path`), passed as kwarg in `timm.create_model()`

3. **`train.py` / `inference.py`**: `Trainer` and `Predictor` orchestrators
   - Trainer: Mixed precision (AMP) with `GradScaler`, early stopping, TensorBoard logging
   - Predictor: Batch inference with optional TTA (horizontal flip averaging)
   - **Checkpoint format**: `{'model_state_dict': ..., 'optimizer_state_dict': ..., 'epoch': int, 'best_val_acc': float, 'config': OmegaConf dict}`
   - Predictor loads with `weights_only=False` to support full checkpoint restoration

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

**Config hierarchy and structure**:
- `configs/config.yaml`: Root defaults + `defaults: [model: convnext_base, dataset: flower100, training: default, augmentation: strong]`
- `configs/model/*.yaml`: `architecture`, `pretrained`, `drop_path_rate` (0.0-0.3 range), input_size
- `configs/training/default.yaml`: `optimizer.lr=1e-4`, `scheduler.name=cosine`, `label_smoothing=0.1`, `early_stopping.patience=10`, `use_amp=true`
- `configs/augmentation/{light,medium,strong,ultra_strong}.yaml`: Lists of Albumentations transforms under `train:` and `val:` keys (YAML dicts)
- `configs/dataset/*.yaml`: CSV paths, `val_split=0.2`, `batch_size=32`, `num_workers`

**Accessing config values**: Use dot notation (`cfg.model.architecture`, `cfg.training.lr`, `cfg.augmentation.train`)—OmegaConf enables safe deep access with defaults.

**Design principle**: Never hardcode hyperparameters—add config files or override via CLI. All numeric constants (lr, epochs, thresholds) must be in config.

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
- **Default**: `nn.CrossEntropyLoss` with label_smoothing from config (`cfg.training.label_smoothing`)
- **Focal Loss**: For imbalanced classes, `alpha=0.25, gamma=2.0`
- Factory: `get_loss_function(cfg)` returns configured loss based on `cfg.training.loss_type` (checks "focal" or defaults to CrossEntropy)

### Mixed Precision Training (AMP)
```python
# Pattern in Trainer.train_step() (called during training):
with autocast():  # torch.amp.autocast
    outputs = self.model(images)
    loss = self.criterion(outputs, labels)
self.scaler.scale(loss).backward()  # GradScaler('cuda')
self.scaler.unscale_(self.optimizer)
torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.training.clip_grad_norm)
self.scaler.step(self.optimizer)
self.scaler.update()
```
- Only enabled if `cfg.training.use_amp=true` (default); scaler set to `None` if AMP disabled

### Inference with Image ID Tracking
```python
# FlowerDataset returns (image, label, image_id)
# Predictor.predict() preserves image_ids for CSV output:
# Returns: (image_ids, predictions, logits_arrays)
# CSV format: image_id, predicted_label (no logits)
```

### TensorBoard Logging Pattern
- Trainer writes per-epoch metrics: `self.writer.add_scalar('loss/train', train_loss, epoch)`
- Logged to `cfg.log_dir` (default: `results/logs/`)
- Access via: `tensorboard --logdir results/logs --port 6006`

### Augmentation Pipeline Loading
```python
# In build_transforms(), cfg.augmentation.train/val are lists of dicts
# Each dict converted to Albumentations op: A.RandomResizedCrop(**dict_item)
# Always ends with Normalize + ToTensorV2 for model input
```

## Code Organization & Key Interfaces

### Module Factory Functions (importable from `__init__.py`)
- **`models.build_model(cfg)`**: Returns `FlowerRecognitionModel` instance; reads `cfg.model.*` for architecture selection
- **`models.get_model_size_mb(model)`**: Returns model size in MB; used to validate ≤500MB constraint
- **`models.get_loss_function(cfg)`**: Returns loss instance based on `cfg.training.loss_type`
- **`datasets.create_dataloaders(cfg)`**: Returns `(train_loader, val_loader, test_loader)` tuple; handles CSV loading + train/val split
- **`datasets.build_transforms(cfg, is_train, input_size)`**: Returns Albumentations.Compose pipeline

### Training State & Checkpointing
- Trainer attributes: `self.model`, `self.optimizer`, `self.scheduler`, `self.scaler`, `self.current_epoch`, `self.best_val_acc`, `self.writer`
- Checkpoint saved to `cfg.checkpoint_dir/best_model.pt` when validation improves; periodic saves as `checkpoint_epoch_N.pt`
- Early stopping triggered after `cfg.training.early_stopping.patience` epochs without improvement

### Data Flow Convention
1. **CSV files** (train.csv): `image_id,label` columns (integers 0-99)
2. **Image files**: Named by `image_id` in CSV (e.g., `001.jpg`, `image_abc.jpg`)
3. **Test predictions**: CSV with `image_id, predicted_label` (no header by default in submission format)

## File References

| File | Purpose |
|------|---------|
| `train.py` (Trainer class) | Main training loop, checkpointing, TensorBoard setup; entry point via Hydra |
| `inference.py` (Predictor class) | Batch prediction, TTA, speed benchmarking; supports `--benchmark` flag |
| `models/flower_model.py` + `models/__init__.py` | `FlowerRecognitionModel`, `build_model()`, `get_model_size_mb()`, `get_loss_function()` |
| `datasets/flower_dataset.py` + `datasets/__init__.py` | `FlowerDataset`, `create_dataloaders()`, `build_transforms()` |
| `utils.py` | Plotting (training curves, confusion matrix), metrics, seed management |
| `verify_setup.py` | Validate data directory structure before training |
| `prepare_submission.py` | Package predictions into competition submission format |
| `configs/config.yaml` | Hydra defaults + output paths |

## Troubleshooting Checklist

1. **Model exceeds 500MB**: Use `convnext_tiny` (146MB) or reduce `drop_path_rate` from 0.1 → 0.05
2. **CUDA OOM during training**: Reduce `batch_size` (config: `dataset.batch_size=16`), decrease `num_workers`, or use AMP
3. **Validation accuracy not improving**: Try `augmentation=light` or increase `training.epochs`, check label distribution
4. **Inference slow (>100ms)**: Enable AMP in config (`training.use_amp=true`), use smaller model, reduce batch size
5. **CSV encoding errors**: Ensure `pd.read_csv()` and `.to_csv()` use `encoding='utf-8'`; verify image filenames match CSV
6. **Image loading fails**: Check PIL ImageFile truncation handling in `flower_dataset.py` line ~75 is enabled
7. **Config not applying**: Use `python train.py --cfg job` to verify config cascade; CLI overrides take precedence

## Coding Conventions

- **Imports**: PyTorch path added via `sys.path.append(str(Path(__file__).parent))` in executable scripts
- **Type hints**: Used in function signatures (e.g., `cfg: DictConfig`, `model: nn.Module`)
- **Error handling**: OmegaConf safe access via dot notation (returns None if missing); explicit fallbacks in code
- **Random seed**: Set in Trainer.__init__() via `torch.manual_seed()`, `np.random.seed()`, cuDNN determinism flags
- **Image preprocessing**: All images RGB (converted in `FlowerDataset.__getitem__()`); normalized per ImageNet stats in augmentation config
- **Batch dimensions**: Assumed `(B, C, H, W)` for images; labels are `(B,)` integers
