"""
Model architectures for Flower Recognition.

This module provides state-of-the-art CNN and Vision Transformer models
using the timm library (PyTorch Image Models).
"""

import torch
import torch.nn as nn
import timm
from typing import Optional


class FlowerRecognitionModel(nn.Module):
    """
    Wrapper class for flower recognition models.
    
    Supports various architectures from timm library with custom head
    for flower classification.
    """
    
    def __init__(
        self,
        architecture: str = 'convnext_base',
        num_classes: int = 100,
        pretrained: bool = True,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        **kwargs
    ):
        """
        Args:
            architecture: Model architecture name from timm
            num_classes: Number of flower classes
            pretrained: Whether to use pretrained weights
            drop_rate: Dropout rate
            drop_path_rate: Drop path rate for stochastic depth
            **kwargs: Additional arguments passed to timm.create_model
        """
        super().__init__()
        
        self.architecture = architecture
        self.num_classes = num_classes
        
        # Create model using timm
        self.model = timm.create_model(
            architecture,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            **kwargs
        )
        
        # Get model info
        self.model_info = {
            'architecture': architecture,
            'num_classes': num_classes,
            'pretrained': pretrained,
            'num_params': sum(p.numel() for p in self.parameters()),
            'num_trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, num_classes)
        """
        return self.model(x)
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return self.model_info
    
    def freeze_backbone(self):
        """Freeze backbone parameters for transfer learning."""
        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze classifier head
        if hasattr(self.model, 'head'):
            for param in self.model.head.parameters():
                param.requires_grad = True
        elif hasattr(self.model, 'fc'):
            for param in self.model.fc.parameters():
                param.requires_grad = True
        elif hasattr(self.model, 'classifier'):
            for param in self.model.classifier.parameters():
                param.requires_grad = True
    
    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = True


def build_model(cfg) -> FlowerRecognitionModel:
    """
    Build model from configuration.
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        FlowerRecognitionModel instance
    """
    model = FlowerRecognitionModel(
        architecture=cfg.model.architecture,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
        drop_rate=cfg.model.get('drop_rate', 0.0),
        drop_path_rate=cfg.model.get('drop_path_rate', 0.0)
    )
    
    print(f"\n{'='*60}")
    print(f"Model: {cfg.model.architecture}")
    print(f"Total Parameters: {model.model_info['num_params']:,}")
    print(f"Trainable Parameters: {model.model_info['num_trainable_params']:,}")
    print(f"{'='*60}\n")
    
    return model


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    """
    Calculate model size in MB.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


# Recommended models for the competition
RECOMMENDED_MODELS = {
    'convnext_base': {
        'description': 'ConvNeXt Base - Modern CNN with ViT design',
        'params': '89M',
        'accuracy': 'High',
        'speed': 'Fast'
    },
    'tf_efficientnetv2_l': {
        'description': 'EfficientNetV2 Large - Efficient and accurate',
        'params': '120M',
        'accuracy': 'Very High',
        'speed': 'Medium'
    },
    'swinv2_base_window12to16_192to256.ms_in22k_ft_in1k': {
        'description': 'Swin Transformer V2 - State-of-the-art ViT',
        'params': '88M',
        'accuracy': 'Very High',
        'speed': 'Medium'
    },
    'convnext_large': {
        'description': 'ConvNeXt Large - Larger version for max accuracy',
        'params': '198M',
        'accuracy': 'Very High',
        'speed': 'Medium'
    }
}


def list_recommended_models():
    """Print recommended models for the competition."""
    print("\n" + "="*80)
    print("Recommended Models for Flower Recognition Challenge")
    print("="*80)
    
    for model_name, info in RECOMMENDED_MODELS.items():
        print(f"\n{model_name}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
    
    print("\n" + "="*80 + "\n")
