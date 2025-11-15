# Training Guide - Focal Loss & High Performance Models

## 🎯 Current Status Summary

### ✅ Verified Capabilities
- [x] **Hydra Configuration** - All settings managed via YAML composition
- [x] **Pretrained Models** - All 6 models use ImageNet weights (`pretrained: true`)
- [x] **Train/Val Split** - Stratified split with configurable `val_split` ratio
- [x] **Strong Augmentation** - Three levels: light/medium/strong + new ultra_strong
- [x] **Focal Loss** - Now available for handling difficult samples

### 📊 Available Loss Functions
1. **CrossEntropyLoss** (default) - Standard classification loss
2. **LabelSmoothingCrossEntropy** - Prevents overconfidence
3. **FocalLoss** (NEW) - Focuses on hard-to-classify samples
   - `alpha=0.25`: Balance factor between easy/hard samples
   - `gamma=2.0`: Focusing parameter (higher = more focus on hard samples)

### 🏆 Recommended Training Configurations

#### Option 1: High Performance (Best for Competition)
```powershell
python train.py --config-name high_performance training.epochs=100
```
**Uses**: EfficientNetV2-L + Focal Loss + Ultra Strong Augmentation
**Expected**: 88-92% accuracy, ~460MB model size, ~50-80ms inference

#### Option 2: Baseline with Focal Loss
```powershell
python train.py model=convnext_base training=focal augmentation=ultra_strong training.epochs=50
```
**Uses**: ConvNeXt Base + Focal Loss + Ultra Strong Augmentation
**Expected**: 85-88% accuracy, faster training for validation

#### Option 3: Quick Test (Verify Pipeline)
```powershell
python train.py model=convnext_tiny training=focal augmentation=medium training.epochs=5
```
**Uses**: Smallest model for quick iteration
**Expected**: Completes in ~10-15 mins, verifies end-to-end pipeline

## 📁 New Files Created

### Loss Functions (`models/losses.py`)
```python
# Three loss function classes:
1. FocalLoss - For handling difficult samples
   - Downweights easy samples: loss = alpha * (1-p)^gamma * ce_loss
   - Configurable alpha (balance) and gamma (focusing)
   
2. LabelSmoothingCrossEntropy - Prevents overconfidence
   - Smooths one-hot labels with smoothing factor
   
3. get_loss_function(cfg) - Factory function
   - Selects loss based on training config
```

### Training Configs
- `configs/training/focal.yaml` - Focal Loss settings
- `configs/model/efficientnet_v2_l_optimized.yaml` - High-performance model
- `configs/augmentation/ultra_strong.yaml` - Flower-specific augmentations
- `configs/high_performance.yaml` - Combined best practices

## 🚀 Training Workflow

### Step 1: Quick Validation (5-10 mins)
```powershell
# Verify everything works
python train.py model=convnext_tiny training=focal training.epochs=2

# Check model size and inference speed
python check_competition.py --checkpoint results/checkpoints/best_model.pt
```

### Step 2: Baseline Training (2-4 hours)
```powershell
# Train ConvNeXt Base with Focal Loss
python train.py model=convnext_base training=focal augmentation=ultra_strong training.epochs=50

# Monitor with TensorBoard
tensorboard --logdir results/logs
```

### Step 3: High-Performance Training (6-12 hours)
```powershell
# Full competition model - EfficientNetV2-L
python train.py --config-name high_performance training.epochs=100

# With early stopping, may finish earlier if convergence detected
```

### Step 4: Inference & Submission
```powershell
# Standard prediction
python inference.py --checkpoint results/checkpoints/best_model.pt --output predictions.csv

# With TTA (Test Time Augmentation) for +1-2% accuracy
python inference.py --checkpoint results/checkpoints/best_model.pt --output predictions.csv --tta

# Benchmark inference speed (must be <100ms)
python inference.py --checkpoint results/checkpoints/best_model.pt --benchmark

# Prepare submission package
python prepare_submission.py --checkpoint results/checkpoints/best_model.pt --predictions predictions.csv
```

## 🔧 Focal Loss Configuration

### When to Use Focal Loss?
- ✅ **Multi-class classification** (100 flower classes)
- ✅ **Class imbalance** (some flower types may be rare)
- ✅ **Difficult samples** (visually similar flowers)
- ✅ **Fine-grained recognition** (subtle differences between classes)

### Tuning Parameters
Edit `configs/training/focal.yaml`:
```yaml
loss_function: focal
focal_alpha: 0.25    # Balance factor (0.25 = emphasize hard samples)
focal_gamma: 2.0     # Focusing parameter (2.0 = standard, higher = more aggressive)
label_smoothing: 0.1 # Optional smoothing (0.0 = disabled)
```

**Recommendations**:
- `gamma=2.0` is standard, start here
- Increase `gamma` to 3.0-5.0 if model focuses too much on easy samples
- Adjust `alpha` based on class imbalance (lower = more focus on minority classes)

## 📊 Expected Performance

### Model Comparison (100 epochs, Focal Loss, Ultra Strong Augmentation)

| Model | Params | Size | Val Acc | Inference | Notes |
|-------|--------|------|---------|-----------|-------|
| ConvNeXt Tiny | 28M | ~110MB | 85-88% | ~30ms | Fast baseline |
| ConvNeXt Base | 89M | ~340MB | 87-90% | ~50ms | Balanced |
| EfficientNetV2-L | 120M | ~460MB | 88-92% | ~80ms | **Best for competition** |
| ConvNeXt Large | 198M | ~490MB | 88-91% | ~90ms | Near size limit |

**Note**: All models within competition constraints (<500MB, <100ms, <10B params)

## 🎯 Competition Constraints Checklist

Before submission, verify:
- [ ] Model size < 500MB: `python check_competition.py --checkpoint <path>`
- [ ] Inference < 100ms: `python inference.py --checkpoint <path> --benchmark`
- [ ] Input size 600×600: Configured in all model YAML files
- [ ] UTF-8 encoding: CSV files automatically use UTF-8
- [ ] Submission format: Use `prepare_submission.py` to create ZIP

## 🐛 Common Issues & Solutions

### Issue 1: CUDA Out of Memory
```powershell
# Solution: Reduce batch size in config
python train.py dataset.batch_size=16  # Default is 32
```

### Issue 2: Model Size Exceeds 500MB
```powershell
# Solution: Use smaller model or reduce drop_path_rate
python train.py model=convnext_base  # Instead of convnext_large
python train.py model.drop_path_rate=0.1  # Reduce from 0.25
```

### Issue 3: Inference Too Slow (>100ms)
```yaml
# Edit configs/training/*.yaml, ensure AMP is enabled:
use_amp: true  # Mixed precision reduces inference time

# Consider smaller model:
python train.py model=efficientnet_b3  # ~60ms inference
```

### Issue 4: Training Doesn't Improve
```yaml
# Try adjusting Focal Loss parameters in configs/training/focal.yaml:
focal_gamma: 3.0  # Increase focusing on hard samples
learning_rate: 0.0001  # Reduce learning rate
```

## 📈 Next Steps for Improvement

### Priority 1: Model Ensemble (2-5% gain)
Combine predictions from multiple models:
```python
# Create ensemble.py
models = [
    'convnext_base_checkpoint.pt',
    'efficientnet_v2_l_checkpoint.pt',
    'swin_transformer_checkpoint.pt'
]
final_prediction = average(model_predictions)
```

### Priority 2: Enhanced TTA (1-3% gain)
Add more test-time augmentations in `inference.py`:
- Horizontal + Vertical flips
- Rotation ±5°
- Brightness adjustment ±10%

### Priority 3: Advanced Augmentation
Experiment with `ultra_strong.yaml` settings:
- Increase CoarseDropout probability
- Add Cutout/GridMask
- Tune color jitter ranges

## 🎓 Training Tips

1. **Start Small**: Always run quick test with `convnext_tiny` first
2. **Monitor TensorBoard**: Watch val_accuracy curve for overfitting
3. **Early Stopping**: Configured with patience=10, saves best checkpoint
4. **Checkpoint Resume**: Edit `train.py` to load checkpoint if training interrupted
5. **Multiple Runs**: Train with different seeds for ensemble diversity

## 📞 Quick Reference

```powershell
# Train high-performance model
python train.py --config-name high_performance

# List available models
python cli/flower_cli.py models

# Evaluate on validation set
python evaluate.py --checkpoint results/checkpoints/best_model.pt

# Predict with TTA
python inference.py --checkpoint results/checkpoints/best_model.pt --output predictions.csv --tta

# Prepare submission
python prepare_submission.py --checkpoint results/checkpoints/best_model.pt --predictions predictions.csv
```

## 📊 Training Logs Location
- Checkpoints: `results/checkpoints/`
- TensorBoard logs: `results/logs/`
- Predictions: `predictions.csv` (root directory)
- Submission: `submission.zip` (created by prepare_submission.py)

---

**Ready to train?** Start with Option 3 (Quick Test) to verify everything works, then proceed to Option 1 (High Performance) for competition submission.
