"""
Evaluation script for Flower Recognition model.

This script evaluates model performance on validation set and generates
detailed metrics and visualizations.
"""

import os
import sys
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from datasets import create_dataloaders, get_class_names
from models import build_model
from utils import (
    plot_confusion_matrix,
    generate_classification_report,
    visualize_predictions
)


class Evaluator:
    """Evaluator class for model evaluation."""
    
    def __init__(self, cfg: DictConfig, checkpoint_path: str):
        """
        Initialize evaluator.
        
        Args:
            cfg: Hydra configuration
            checkpoint_path: Path to model checkpoint
        """
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
        
        # Build model
        self.model = build_model(cfg).to(self.device)
        
        # Load checkpoint
        self.load_checkpoint(checkpoint_path)
        
        # Set to evaluation mode
        self.model.eval()
        
        print(f"Model loaded from: {checkpoint_path}")
        print(f"Using device: {self.device}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'best_val_acc' in checkpoint:
            print(f"Checkpoint validation accuracy: {checkpoint['best_val_acc']:.2f}%")
        
        if 'epoch' in checkpoint:
            print(f"Checkpoint epoch: {checkpoint['epoch']}")
    
    @torch.no_grad()
    def evaluate(self, dataloader) -> dict:
        """
        Evaluate model on dataset.
        
        Args:
            dataloader: DataLoader for evaluation data
            
        Returns:
            Dictionary with evaluation metrics
        """
        all_predictions = []
        all_labels = []
        all_image_ids = []
        
        pbar = tqdm(dataloader, desc='Evaluating')
        
        for images, labels, image_ids in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            _, predictions = outputs.max(1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_image_ids.extend(image_ids)
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        accuracy = (all_predictions == all_labels).mean() * 100
        
        # Per-class accuracy
        num_classes = self.cfg.model.num_classes
        per_class_acc = []
        for class_id in range(num_classes):
            mask = all_labels == class_id
            if mask.sum() > 0:
                class_acc = (all_predictions[mask] == all_labels[mask]).mean() * 100
                per_class_acc.append(class_acc)
            else:
                per_class_acc.append(0.0)
        
        results = {
            'accuracy': accuracy,
            'predictions': all_predictions,
            'labels': all_labels,
            'image_ids': all_image_ids,
            'per_class_accuracy': per_class_acc,
            'mean_per_class_accuracy': np.mean(per_class_acc)
        }
        
        return results
    
    def generate_report(self, results: dict, output_dir: str):
        """
        Generate evaluation report with visualizations.
        
        Args:
            results: Evaluation results dictionary
            output_dir: Directory to save report files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        predictions = results['predictions']
        labels = results['labels']
        
        # Get class names
        class_names = get_class_names(self.cfg.dataset.train_csv)
        
        print("\n" + "="*60)
        print("Evaluation Results")
        print("="*60)
        print(f"Overall Accuracy: {results['accuracy']:.2f}%")
        print(f"Mean Per-Class Accuracy: {results['mean_per_class_accuracy']:.2f}%")
        print(f"Total Samples: {len(predictions)}")
        
        # Find best and worst performing classes
        per_class_acc = results['per_class_accuracy']
        sorted_indices = np.argsort(per_class_acc)
        
        print(f"\nTop 5 Best Performing Classes:")
        for idx in sorted_indices[-5:][::-1]:
            class_name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
            print(f"  {class_name}: {per_class_acc[idx]:.2f}%")
        
        print(f"\nTop 5 Worst Performing Classes:")
        for idx in sorted_indices[:5]:
            class_name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
            print(f"  {class_name}: {per_class_acc[idx]:.2f}%")
        
        print("="*60 + "\n")
        
        # Generate confusion matrix
        print("Generating confusion matrix...")
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plot_confusion_matrix(
            labels,
            predictions,
            class_names=None,  # Too many classes for labels
            save_path=cm_path,
            normalize=True
        )
        
        # Generate classification report
        print("Generating classification report...")
        report_path = os.path.join(output_dir, "classification_report.txt")
        report = generate_classification_report(
            labels,
            predictions,
            class_names=class_names,
            save_path=report_path
        )
        
        # Save detailed results
        results_df = pd.DataFrame({
            'image_id': results['image_ids'],
            'true_label': labels,
            'predicted_label': predictions,
            'correct': labels == predictions
        })
        results_csv_path = os.path.join(output_dir, "detailed_results.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"Detailed results saved to {results_csv_path}")
        
        # Save per-class accuracy
        per_class_df = pd.DataFrame({
            'class_id': range(len(per_class_acc)),
            'class_name': class_names,
            'accuracy': per_class_acc
        })
        per_class_csv_path = os.path.join(output_dir, "per_class_accuracy.csv")
        per_class_df.to_csv(per_class_csv_path, index=False)
        print(f"Per-class accuracy saved to {per_class_csv_path}")
        
        print(f"\nAll evaluation results saved to {output_dir}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Flower Recognition Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Flower Recognition Evaluation")
    print("="*60)
    
    # Create evaluator
    evaluator = Evaluator(cfg, args.checkpoint)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(cfg)
    
    if val_loader is None:
        print("Error: No validation data found")
        return 1
    
    # Evaluate
    print(f"\nEvaluating on {len(val_loader.dataset)} validation samples...")
    results = evaluator.evaluate(val_loader)
    
    # Generate report
    evaluator.generate_report(results, args.output_dir)
    
    print("\nEvaluation completed successfully!")
    
    return 0


if __name__ == '__main__':
    main()
