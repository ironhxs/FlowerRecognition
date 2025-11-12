"""
Utility functions for the Flower Recognition project.
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Tuple
import pandas as pd


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: str = None
):
    """
    Plot training and validation curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies
    ax2.plot(train_accs, label='Train Acc', linewidth=2)
    ax2.plot(val_accs, label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    save_path: str = None,
    normalize: bool = True
):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save figure
        normalize: Whether to normalize the confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(20, 18))
    sns.heatmap(
        cm,
        annot=False,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names if class_names else range(len(cm)),
        yticklabels=class_names if class_names else range(len(cm)),
        cbar_kws={'label': 'Normalized Count' if normalize else 'Count'}
    )
    
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    save_path: str = None
) -> str:
    """
    Generate detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save report
        
    Returns:
        Classification report string
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Classification report saved to {save_path}")
    
    return report


def analyze_predictions(
    predictions_csv: str,
    ground_truth_csv: str = None
):
    """
    Analyze predictions and compare with ground truth if available.
    
    Args:
        predictions_csv: Path to predictions CSV
        ground_truth_csv: Path to ground truth CSV (optional)
    """
    pred_df = pd.read_csv(predictions_csv)
    
    print("\n" + "="*60)
    print("Prediction Analysis")
    print("="*60)
    print(f"Total predictions: {len(pred_df)}")
    print(f"Unique labels: {pred_df['label'].nunique()}")
    print(f"\nLabel distribution:")
    print(pred_df['label'].value_counts().head(10))
    
    if ground_truth_csv and os.path.exists(ground_truth_csv):
        gt_df = pd.read_csv(ground_truth_csv)
        
        # Merge on image_id
        merged = pred_df.merge(gt_df, on='image_id', suffixes=('_pred', '_true'))
        
        # Calculate accuracy
        accuracy = (merged['label_pred'] == merged['label_true']).mean() * 100
        
        print(f"\nAccuracy: {accuracy:.2f}%")
        print(f"Correct predictions: {(merged['label_pred'] == merged['label_true']).sum()}")
        print(f"Incorrect predictions: {(merged['label_pred'] != merged['label_true']).sum()}")
    
    print("="*60 + "\n")


def calculate_model_flops(model: torch.nn.Module, input_size: Tuple[int, int, int] = (3, 600, 600)):
    """
    Calculate model FLOPs (requires thop library).
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (C, H, W)
    """
    try:
        from thop import profile, clever_format
        
        input_tensor = torch.randn(1, *input_size)
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        
        print(f"FLOPs: {flops}")
        print(f"Parameters: {params}")
        
    except ImportError:
        print("thop library not installed. Install with: pip install thop")


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_size: Tuple[int, int, int] = (3, 600, 600),
    opset_version: int = 11
):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model
        output_path: Path to save ONNX model
        input_size: Input tensor size (C, H, W)
        opset_version: ONNX opset version
    """
    model.eval()
    dummy_input = torch.randn(1, *input_size)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to ONNX: {output_path}")
    
    # Get file size
    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"ONNX model size: {file_size:.2f} MB")


def visualize_predictions(
    image_paths: List[str],
    predictions: List[int],
    class_names: List[str],
    ground_truth: List[int] = None,
    num_images: int = 16,
    save_path: str = None
):
    """
    Visualize predictions on images.
    
    Args:
        image_paths: List of image paths
        predictions: List of predicted labels
        class_names: List of class names
        ground_truth: List of ground truth labels (optional)
        num_images: Number of images to visualize
        save_path: Path to save figure
    """
    from PIL import Image
    
    num_images = min(num_images, len(image_paths))
    rows = int(np.sqrt(num_images))
    cols = int(np.ceil(num_images / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if num_images > 1 else [axes]
    
    for idx, (img_path, pred) in enumerate(zip(image_paths[:num_images], predictions[:num_images])):
        if not os.path.exists(img_path):
            continue
            
        img = Image.open(img_path).convert('RGB')
        axes[idx].imshow(img)
        axes[idx].axis('off')
        
        # Create title
        title = f"Pred: {class_names[pred]}" if class_names else f"Pred: {pred}"
        
        if ground_truth is not None:
            gt = ground_truth[idx]
            gt_name = class_names[gt] if class_names else str(gt)
            title += f"\nGT: {gt_name}"
            
            # Color code: green if correct, red if incorrect
            color = 'green' if pred == gt else 'red'
            axes[idx].set_title(title, color=color, fontsize=10)
        else:
            axes[idx].set_title(title, fontsize=10)
    
    # Hide empty subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
