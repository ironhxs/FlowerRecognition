"""
Evaluation script for flower recognition model.
"""
import os
import argparse
import yaml

import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.model import create_model
from src.dataset import create_dataloaders
from src.utils import AverageMeter, calculate_accuracy


def evaluate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list = None
):
    """
    Evaluate model on a dataset.
    
    Args:
        model: Trained model
        data_loader: Data loader for evaluation
        criterion: Loss function
        device: Device to run evaluation on
        class_names: List of class names
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Calculate accuracy
            acc = calculate_accuracy(outputs, labels)
            
            # Update metrics
            losses.update(loss.item(), images.size(0))
            accuracies.update(acc, images.size(0))
            
            # Store predictions and labels
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{accuracies.avg:.2f}%'
            })
    
    # Generate classification report
    if class_names is None:
        class_names = [f"Class {i}" for i in range(max(all_labels) + 1)]
    
    report = classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        digits=4
    )
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    return {
        'loss': losses.avg,
        'accuracy': accuracies.avg,
        'predictions': all_predictions,
        'labels': all_labels,
        'classification_report': report,
        'confusion_matrix': cm
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    save_path: str = None
):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save plot
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate flower recognition model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to evaluation data (default: test path from config)')
    parser.add_argument('--plot-cm', action='store_true',
                        help='Plot confusion matrix')
    parser.add_argument('--save-cm', type=str, default=None,
                        help='Path to save confusion matrix plot')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Determine data path
    data_path = args.data_path if args.data_path else config['data']['test_path']
    
    # Create data loader
    _, _, test_loader = create_dataloaders(
        train_dir=config['data']['train_path'],
        val_dir=config['data']['val_path'],
        test_dir=data_path,
        batch_size=config['training']['batch_size'],
        image_size=config['data']['image_size'],
        num_workers=config['data']['num_workers']
    )
    
    if test_loader is None:
        print(f"Error: No data found at {data_path}")
        return
    
    print(f"Evaluation samples: {len(test_loader.dataset)}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = create_model(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=False,
        device=device
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate
    print("\nEvaluating model...")
    results = evaluate(
        model,
        test_loader,
        criterion,
        device,
        class_names=test_loader.dataset.get_class_names()
    )
    
    # Print results
    print(f"\nTest Loss: {results['loss']:.4f}")
    print(f"Test Accuracy: {results['accuracy']:.2f}%")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Plot confusion matrix if requested
    if args.plot_cm or args.save_cm:
        plot_confusion_matrix(
            results['confusion_matrix'],
            test_loader.dataset.get_class_names(),
            args.save_cm
        )


if __name__ == '__main__':
    main()
