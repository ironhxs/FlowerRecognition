#!/usr/bin/env python3
"""
Project Verification Script

Checks that all components of the Flower Recognition system are properly set up.
"""

import os
import sys
from pathlib import Path


def check_file_exists(filepath, description):
    """Check if a file exists."""
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {filepath}")
    return exists


def check_directory_exists(dirpath, description):
    """Check if a directory exists."""
    exists = os.path.isdir(dirpath)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {dirpath}")
    return exists


def main():
    print("\n" + "="*60)
    print("Flower Recognition System - Verification")
    print("="*60 + "\n")
    
    all_checks = []
    
    # Check directories
    print("Checking directories...")
    dirs = [
        ("configs", "Configuration directory"),
        ("configs/model", "Model configs"),
        ("configs/dataset", "Dataset configs"),
        ("configs/training", "Training configs"),
        ("configs/augmentation", "Augmentation configs"),
        ("datasets", "Dataset module"),
        ("models", "Model module"),
        ("cli", "CLI module"),
        ("docs", "Documentation"),
        ("results", "Results directory"),
    ]
    
    for dirpath, desc in dirs:
        all_checks.append(check_directory_exists(dirpath, desc))
    
    print()
    
    # Check main scripts
    print("Checking main scripts...")
    scripts = [
        ("train.py", "Training script"),
        ("inference.py", "Inference script"),
        ("evaluate.py", "Evaluation script"),
        ("utils.py", "Utility functions"),
        ("generate_sample_data.py", "Sample data generator"),
        ("prepare_submission.py", "Submission preparation"),
    ]
    
    for filepath, desc in scripts:
        all_checks.append(check_file_exists(filepath, desc))
    
    print()
    
    # Check configuration files
    print("Checking configuration files...")
    configs = [
        ("configs/config.yaml", "Main config"),
        ("configs/model/convnext_base.yaml", "ConvNeXt Base"),
        ("configs/model/efficientnet_v2_l.yaml", "EfficientNetV2-L"),
        ("configs/model/swin_transformer_v2.yaml", "Swin Transformer V2"),
        ("configs/augmentation/strong.yaml", "Strong augmentation"),
        ("configs/dataset/flower100.yaml", "Dataset config"),
        ("configs/training/default.yaml", "Training config"),
    ]
    
    for filepath, desc in configs:
        all_checks.append(check_file_exists(filepath, desc))
    
    print()
    
    # Check documentation
    print("Checking documentation...")
    docs = [
        ("README.md", "Main README"),
        ("docs/QUICKSTART.md", "Quick start guide"),
        ("docs/USAGE_EXAMPLES.md", "Usage examples"),
        ("docs/technical_report_template.md", "Technical report template"),
        ("docs/PROJECT_SUMMARY.md", "Project summary"),
    ]
    
    for filepath, desc in docs:
        all_checks.append(check_file_exists(filepath, desc))
    
    print()
    
    # Check Python module structure
    print("Checking Python modules...")
    modules = [
        ("datasets/__init__.py", "Dataset module init"),
        ("models/__init__.py", "Model module init"),
        ("cli/__init__.py", "CLI module init"),
    ]
    
    for filepath, desc in modules:
        all_checks.append(check_file_exists(filepath, desc))
    
    print()
    
    # Check dependencies
    print("Checking dependencies...")
    all_checks.append(check_file_exists("requirements.txt", "Requirements file"))
    all_checks.append(check_file_exists("setup.sh", "Setup script"))
    
    print()
    
    # Try importing key modules
    print("Testing Python imports...")
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        
        import datasets
        print("✅ datasets module imports successfully")
        all_checks.append(True)
        
        import models
        print("✅ models module imports successfully")
        all_checks.append(True)
        
        import cli
        print("✅ cli module imports successfully")
        all_checks.append(True)
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        all_checks.append(False)
    
    print()
    
    # Summary
    print("="*60)
    passed = sum(all_checks)
    total = len(all_checks)
    percentage = (passed / total * 100) if total > 0 else 0
    
    print(f"Verification Results: {passed}/{total} checks passed ({percentage:.1f}%)")
    
    if passed == total:
        print("✅ All checks passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Generate sample data: python generate_sample_data.py")
        print("2. Train a model: python train.py")
        print("3. Generate predictions: python inference.py --checkpoint <path> --output predictions.csv")
        return 0
    else:
        print("❌ Some checks failed. Please review the output above.")
        return 1
    
    print("="*60 + "\n")


if __name__ == "__main__":
    exit(main())
