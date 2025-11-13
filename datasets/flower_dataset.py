"""
Flower Recognition Dataset Module

This module handles data loading, preprocessing, and augmentation for the 
Flower Recognition AI Challenge.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from typing import Optional, Callable, Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


class FlowerDataset(Dataset):
    """
    Dataset class for Flower Recognition Challenge.
    
    Supports both training and test datasets with flexible augmentation.
    """
    
    def __init__(
        self,
        data_dir: str,
        csv_file: Optional[str] = None,
        image_ids: Optional[List[str]] = None,
        labels: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
        is_test: bool = False
    ):
        """
        Args:
            data_dir: Directory containing images
            csv_file: Path to CSV file with image_id and label columns (for training)
            image_ids: List of image IDs (alternative to csv_file)
            labels: List of labels (alternative to csv_file)
            transform: Albumentations transform pipeline
            is_test: Whether this is test dataset (no labels)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.is_test = is_test
        
        if csv_file is not None:
            df = pd.read_csv(csv_file)
            self.image_ids = df['image_id'].tolist()
            self.labels = df['label'].tolist() if 'label' in df.columns else None
        else:
            self.image_ids = image_ids if image_ids is not None else []
            self.labels = labels
            
        # For test set, scan directory if no image_ids provided
        if self.is_test and not self.image_ids:
            self.image_ids = [
                f for f in os.listdir(data_dir) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            self.image_ids.sort()
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get image and label at index.
        
        Returns:
            image: Tensor of shape (C, H, W)
            label: Integer label (or -1 for test set)
        """
        image_id = self.image_ids[idx]
        
        # Load image
        image_path = os.path.join(self.data_dir, image_id)
        # Fix truncated images
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Get label
        if self.is_test or self.labels is None:
            label = -1
        else:
            label = self.labels[idx]
        
        return image, label, image_id


def build_transforms(config: dict, is_train: bool = True, input_size: int = 600) -> A.Compose:
    """
    Build albumentations transform pipeline from config.
    
    Args:
        config: Configuration dictionary with augmentation settings
        is_train: Whether to use training or validation transforms
        input_size: Target image size (overrides size in config)
        
    Returns:
        Albumentations Compose object
    """
    augmentations = []
    
    aug_config = config.get('train' if is_train else 'val', [])
    
    for aug_dict in aug_config:
        for aug_name, aug_params in aug_dict.items():
            if aug_params is None:
                aug_params = {}
            else:
                # Make a copy to avoid modifying the original config
                aug_params = aug_params.copy()
            
            # Replace 'size' parameter with input_size if present
            if 'size' in aug_params:
                # For Albumentations, size should be (height, width) or just size for square
                if aug_name in ['RandomResizedCrop', 'CenterCrop']:
                    aug_params['size'] = (input_size, input_size)
                else:
                    aug_params['size'] = input_size
            # Also handle height/width for Resize
            if aug_name == 'Resize' and 'height' in aug_params:
                aug_params['height'] = input_size
                aug_params['width'] = input_size
            
            # Get augmentation class from albumentations
            if hasattr(A, aug_name):
                aug_class = getattr(A, aug_name)
                augmentations.append(aug_class(**aug_params))
    
    # Always add ToTensorV2 at the end
    augmentations.append(ToTensorV2())
    
    return A.Compose(augmentations)


def create_dataloaders(cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Get input size from model config (default to 600 if not specified)
    input_size = getattr(cfg.model, 'input_size', 600)
    
    # Build transforms with model's input size
    train_transform = build_transforms(cfg.augmentation, is_train=True, input_size=input_size)
    val_transform = build_transforms(cfg.augmentation, is_train=False, input_size=input_size)
    
    # Load training data
    train_csv = cfg.dataset.train_csv
    if os.path.exists(train_csv):
        df = pd.read_csv(train_csv)
        
        # Split into train and validation
        if cfg.dataset.val_split > 0:
            train_ids, val_ids, train_labels, val_labels = train_test_split(
                df['image_id'].tolist(),
                df['label'].tolist(),
                test_size=cfg.dataset.val_split,
                random_state=cfg.seed,
                stratify=df['label'].tolist()
            )
        else:
            train_ids = df['image_id'].tolist()
            train_labels = df['label'].tolist()
            val_ids, val_labels = [], []
        
        # Create datasets
        train_dataset = FlowerDataset(
            data_dir=cfg.dataset.train_dir,
            image_ids=train_ids,
            labels=train_labels,
            transform=train_transform,
            is_test=False
        )
        
        val_dataset = FlowerDataset(
            data_dir=cfg.dataset.train_dir,
            image_ids=val_ids,
            labels=val_labels,
            transform=val_transform,
            is_test=False
        ) if val_ids else None
    else:
        print(f"Warning: Training CSV not found at {train_csv}")
        train_dataset = None
        val_dataset = None
    
    # Create test dataset
    test_dataset = FlowerDataset(
        data_dir=cfg.dataset.test_dir,
        transform=val_transform,
        is_test=True
    ) if os.path.exists(cfg.dataset.test_dir) else None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=cfg.dataset.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True
    ) if train_dataset else None
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory
    ) if val_dataset else None
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory
    ) if test_dataset else None
    
    return train_loader, val_loader, test_loader


def get_class_names(csv_file: str) -> List[str]:
    """
    Get list of class names from training CSV.
    
    Args:
        csv_file: Path to training CSV
        
    Returns:
        List of class names (sorted by label ID)
    """
    if not os.path.exists(csv_file):
        return [f"class_{i}" for i in range(100)]
    
    df = pd.read_csv(csv_file)
    
    # Extract unique classes
    if 'class_name' in df.columns:
        class_mapping = df[['label', 'class_name']].drop_duplicates()
        class_mapping = class_mapping.sort_values('label')
        return class_mapping['class_name'].tolist()
    else:
        # Generate generic class names
        num_classes = df['label'].nunique()
        return [f"class_{i}" for i in range(num_classes)]
