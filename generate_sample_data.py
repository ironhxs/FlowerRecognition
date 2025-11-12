#!/usr/bin/env python3
"""
Generate sample dataset for testing the flower recognition system.

This script creates a small dummy dataset with the correct structure
for testing purposes.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import argparse


def generate_random_flower_image(size=(600, 600)):
    """Generate a random colorful image simulating a flower."""
    # Create random colorful image
    img = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)
    
    # Add some patterns to make it look more interesting
    center_x, center_y = size[0] // 2, size[1] // 2
    y, x = np.ogrid[-center_x:size[0]-center_x, -center_y:size[1]-center_y]
    mask = x*x + y*y <= (size[0]//3)**2
    
    # Create flower-like pattern
    img[mask] = img[mask] * 0.7 + np.array([255, 200, 150]) * 0.3
    
    return Image.fromarray(img)


def create_sample_dataset(
    output_dir: str = "./data",
    num_classes: int = 100,
    samples_per_class: int = 10,
    test_samples: int = 50
):
    """
    Create sample dataset for testing.
    
    Args:
        output_dir: Output directory for dataset
        num_classes: Number of flower classes
        samples_per_class: Number of training samples per class
        test_samples: Number of test samples
    """
    
    print(f"Creating sample dataset in {output_dir}...")
    
    # Create directories
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Generate training data
    print(f"Generating {num_classes * samples_per_class} training images...")
    train_data = []
    
    for class_id in range(num_classes):
        for sample_id in range(samples_per_class):
            image_id = f"train_{class_id:03d}_{sample_id:03d}.jpg"
            image_path = os.path.join(train_dir, image_id)
            
            # Generate and save image
            img = generate_random_flower_image()
            img.save(image_path, quality=95)
            
            train_data.append({
                'image_id': image_id,
                'label': class_id,
                'class_name': f'flower_{class_id:03d}'
            })
    
    # Save training CSV
    train_csv_path = os.path.join(output_dir, "train.csv")
    train_df = pd.DataFrame(train_data)
    train_df.to_csv(train_csv_path, index=False, encoding='utf-8')
    print(f"✓ Training CSV saved: {train_csv_path}")
    print(f"  Total samples: {len(train_df)}")
    print(f"  Classes: {train_df['label'].nunique()}")
    
    # Generate test data
    print(f"\nGenerating {test_samples} test images...")
    for test_id in range(test_samples):
        image_id = f"test_{test_id:04d}.jpg"
        image_path = os.path.join(test_dir, image_id)
        
        # Generate and save image
        img = generate_random_flower_image()
        img.save(image_path, quality=95)
    
    print(f"✓ Test images saved in {test_dir}")
    
    # Print summary
    print("\n" + "="*60)
    print("Dataset Summary")
    print("="*60)
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {test_samples}")
    print(f"Number of classes: {num_classes}")
    print(f"Samples per class: {samples_per_class}")
    print(f"\nDataset structure:")
    print(f"  {output_dir}/")
    print(f"  ├── train.csv")
    print(f"  ├── train/")
    print(f"  │   └── train_*.jpg")
    print(f"  └── test/")
    print(f"      └── test_*.jpg")
    print("="*60)
    
    return output_dir


def create_predictions_sample(
    output_path: str = "predictions_sample.csv",
    num_samples: int = 50
):
    """Create sample predictions CSV in correct format."""
    
    predictions = []
    for i in range(num_samples):
        predictions.append({
            'image_id': f'test_{i:04d}.jpg',
            'label': np.random.randint(0, 100)
        })
    
    df = pd.DataFrame(predictions)
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\n✓ Sample predictions saved: {output_path}")
    print(f"  Format: image_id, label")
    print(f"  Samples: {len(df)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate sample flower dataset for testing"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data',
        help='Output directory for dataset'
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=100,
        help='Number of flower classes'
    )
    parser.add_argument(
        '--samples-per-class',
        type=int,
        default=10,
        help='Number of training samples per class'
    )
    parser.add_argument(
        '--test-samples',
        type=int,
        default=50,
        help='Number of test samples'
    )
    parser.add_argument(
        '--create-predictions-sample',
        action='store_true',
        help='Also create sample predictions CSV'
    )
    
    args = parser.parse_args()
    
    # Create dataset
    create_sample_dataset(
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        samples_per_class=args.samples_per_class,
        test_samples=args.test_samples
    )
    
    # Optionally create predictions sample
    if args.create_predictions_sample:
        create_predictions_sample(
            output_path=os.path.join(args.output_dir, "predictions_sample.csv"),
            num_samples=args.test_samples
        )


if __name__ == '__main__':
    main()
