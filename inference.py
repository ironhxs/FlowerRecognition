"""
Inference script for Flower Recognition model.

This script generates predictions for test images and saves them in the
required CSV format for competition submission.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from datasets import FlowerDataset, build_transforms
from models import build_model


class Predictor:
    """Predictor class for generating test predictions."""
    
    def __init__(self, cfg: DictConfig, checkpoint_path: str):
        """
        Initialize predictor.
        
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
            print(f"Model validation accuracy: {checkpoint['best_val_acc']:.2f}%")
    
    @torch.no_grad()
    def predict(
        self,
        dataloader: DataLoader,
        use_tta: bool = False
    ) -> Tuple[List[str], List[int], List[np.ndarray]]:
        """
        Generate predictions for dataset.
        
        Args:
            dataloader: DataLoader for test data
            use_tta: Whether to use test-time augmentation
            
        Returns:
            Tuple of (image_ids, predictions, probabilities)
        """
        image_ids = []
        predictions = []
        probabilities = []
        
        pbar = tqdm(dataloader, desc='Predicting')
        
        for images, _, img_ids in pbar:
            images = images.to(self.device)
            
            # Forward pass
            if use_tta:
                # Test-time augmentation with horizontal flip
                outputs = self.model(images)
                outputs_flip = self.model(torch.flip(images, dims=[3]))
                outputs = (outputs + outputs_flip) / 2
            else:
                outputs = self.model(images)
            
            # Get probabilities
            probs = F.softmax(outputs, dim=1)
            
            # Get predictions
            _, preds = outputs.max(1)
            
            # Store results
            image_ids.extend(img_ids)
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
        
        return image_ids, predictions, probabilities
    
    def predict_single_image(self, image_path: str) -> Tuple[int, np.ndarray]:
        """
        Predict single image.
        
        Args:
            image_path: Path to image
            
        Returns:
            Tuple of (predicted_class, probabilities)
        """
        from PIL import Image
        
        # Build transform
        transform = build_transforms(self.cfg.augmentation, is_train=False)
        
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        transformed = transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(image_tensor)
            probs = F.softmax(output, dim=1)
            pred = output.argmax(1).item()
        
        return pred, probs.cpu().numpy()[0]
    
    def benchmark_inference_speed(self, num_iterations: int = 100) -> float:
        """
        Benchmark inference speed.
        
        Args:
            num_iterations: Number of iterations to run
            
        Returns:
            Average inference time in milliseconds
        """
        # Create dummy input
        dummy_input = torch.randn(
            1, 3, self.cfg.model.input_size, self.cfg.model.input_size
        ).to(self.device)
        
        # Warmup
        for _ in range(10):
            _ = self.model(dummy_input)
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.time()
            with torch.no_grad():
                _ = self.model(dummy_input)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) * 1000)  # Convert to ms
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"\nInference Speed Benchmark:")
        print(f"Average time: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"Min time: {np.min(times):.2f} ms")
        print(f"Max time: {np.max(times):.2f} ms")
        
        if avg_time > 100:
            print("WARNING: Average inference time exceeds 100ms competition limit!")
        
        return avg_time


def save_predictions(
    image_ids: List[str],
    predictions: List[int],
    output_path: str
):
    """
    Save predictions to CSV file in competition format.
    
    Args:
        image_ids: List of image IDs
        predictions: List of predicted labels
        output_path: Path to save CSV file
    """
    df = pd.DataFrame({
        'image_id': image_ids,
        'label': predictions
    })
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV with UTF-8 encoding
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\nPredictions saved to: {output_path}")
    print(f"Total predictions: {len(df)}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main inference function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Flower Recognition Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Output CSV file path')
    parser.add_argument('--tta', action='store_true',
                       help='Use test-time augmentation')
    parser.add_argument('--benchmark', action='store_true',
                       help='Benchmark inference speed')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Flower Recognition Inference")
    print("="*60)
    
    # Create predictor
    predictor = Predictor(cfg, args.checkpoint)
    
    # Benchmark if requested
    if args.benchmark:
        predictor.benchmark_inference_speed()
    
    # Create test dataloader
    transform = build_transforms(cfg.augmentation, is_train=False)
    test_dataset = FlowerDataset(
        data_dir=cfg.dataset.test_dir,
        transform=transform,
        is_test=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory
    )
    
    # Generate predictions
    print(f"\nGenerating predictions for {len(test_dataset)} images...")
    image_ids, predictions, probabilities = predictor.predict(
        test_loader,
        use_tta=args.tta
    )
    
    # Save predictions
    save_predictions(image_ids, predictions, args.output)
    
    print("\nInference completed successfully!")


if __name__ == '__main__':
    main()
