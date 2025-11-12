#!/usr/bin/env python3
"""
Competition Submission Preparation Script

This script helps prepare the final submission package for the
Flower Recognition AI Challenge according to competition requirements.
"""

import os
import shutil
import zipfile
from pathlib import Path
import argparse


def create_submission_package(
    model_checkpoint: str,
    predictions_csv: str,
    output_zip: str = "submission.zip"
):
    """
    Create submission package according to competition requirements.
    
    Required structure:
    submission.zip
    ├── code/
    │   ├── train.py
    │   ├── inference.py
    │   ├── datasets/
    │   ├── models/
    │   └── configs/
    ├── model/
    │   └── best_model.pt
    ├── result/
    │   └── predictions.csv
    └── requirements.txt
    """
    
    print("Creating competition submission package...")
    
    # Create temporary directory
    temp_dir = Path("temp_submission")
    temp_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    code_dir = temp_dir / "code"
    model_dir = temp_dir / "model"
    result_dir = temp_dir / "result"
    
    code_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)
    
    # Copy code files
    print("Copying code files...")
    files_to_copy = [
        "train.py",
        "inference.py",
        "requirements.txt"
    ]
    
    for file in files_to_copy:
        if os.path.exists(file):
            shutil.copy(file, code_dir / file)
    
    # Copy directories
    dirs_to_copy = ["datasets", "models", "configs", "cli"]
    for dir_name in dirs_to_copy:
        if os.path.exists(dir_name):
            shutil.copytree(dir_name, code_dir / dir_name, dirs_exist_ok=True)
    
    # Copy model checkpoint
    print("Copying model checkpoint...")
    if os.path.exists(model_checkpoint):
        shutil.copy(model_checkpoint, model_dir / "best_model.pt")
    else:
        print(f"Warning: Model checkpoint not found: {model_checkpoint}")
    
    # Copy predictions
    print("Copying predictions...")
    if os.path.exists(predictions_csv):
        shutil.copy(predictions_csv, result_dir / "predictions.csv")
    else:
        print(f"Warning: Predictions CSV not found: {predictions_csv}")
    
    # Copy requirements.txt to root
    if os.path.exists("requirements.txt"):
        shutil.copy("requirements.txt", temp_dir / "requirements.txt")
    
    # Create README for submission
    readme_content = """# Flower Recognition Submission

## Structure
- code/: Source code for training and inference
- model/: Trained model checkpoint
- result/: Prediction results CSV
- requirements.txt: Python dependencies

## How to Run

### Training
```bash
cd code
python train.py
```

### Inference
```bash
cd code
python inference.py --checkpoint ../model/best_model.pt --output ../result/predictions.csv
```

## Requirements
- Python 3.8+
- PyTorch 2.0+
- See requirements.txt for full list

## Model Information
- Architecture: ConvNeXt Base / EfficientNetV2-L
- Input Size: 600x600
- Number of Classes: 100
- Model Size: < 500MB
- Inference Time: < 100ms per image
"""
    
    with open(temp_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    # Create ZIP file
    print(f"Creating ZIP archive: {output_zip}")
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(temp_dir)
                zipf.write(file_path, arcname)
    
    # Clean up temporary directory
    print("Cleaning up...")
    shutil.rmtree(temp_dir)
    
    print(f"\n✓ Submission package created: {output_zip}")
    print(f"✓ File size: {os.path.getsize(output_zip) / 1024 / 1024:.2f} MB")
    
    # Verify ZIP structure
    print("\nVerifying package structure...")
    with zipfile.ZipFile(output_zip, 'r') as zipf:
        files = zipf.namelist()
        print(f"Total files in package: {len(files)}")
        
        required_files = [
            "code/train.py",
            "code/inference.py",
            "model/best_model.pt",
            "result/predictions.csv",
            "requirements.txt"
        ]
        
        for required_file in required_files:
            if required_file in files:
                print(f"  ✓ {required_file}")
            else:
                print(f"  ✗ {required_file} (missing)")


def verify_predictions(predictions_csv: str):
    """Verify predictions CSV format."""
    import pandas as pd
    
    print("\nVerifying predictions format...")
    
    try:
        df = pd.read_csv(predictions_csv, encoding='utf-8')
        
        # Check columns
        if 'image_id' not in df.columns or 'label' not in df.columns:
            print("✗ Missing required columns (image_id, label)")
            return False
        
        # Check for null values
        if df.isnull().any().any():
            print("✗ Contains null values")
            return False
        
        # Check label range
        if not df['label'].between(0, 99).all():
            print("✗ Labels must be in range [0, 99]")
            return False
        
        print(f"✓ Valid format")
        print(f"✓ Total predictions: {len(df)}")
        print(f"✓ Unique labels: {df['label'].nunique()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error reading CSV: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Prepare submission package for Flower Recognition Challenge"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--predictions',
        type=str,
        required=True,
        help='Path to predictions CSV'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='submission.zip',
        help='Output ZIP file name'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify predictions format without creating package'
    )
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_predictions(args.predictions)
    else:
        # Verify predictions first
        if verify_predictions(args.predictions):
            # Create submission package
            create_submission_package(
                args.checkpoint,
                args.predictions,
                args.output
            )
        else:
            print("\nPlease fix prediction format before creating submission package.")
            return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
