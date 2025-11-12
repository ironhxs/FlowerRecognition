"""
Model definitions for flower recognition.
"""
import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class FlowerClassifier(nn.Module):
    """Flower classification model based on pre-trained architectures."""
    
    def __init__(
        self,
        model_name: str = "resnet50",
        num_classes: int = 102,
        pretrained: bool = True
    ):
        """
        Args:
            model_name: Name of the backbone architecture
            num_classes: Number of flower classes
            pretrained: Whether to use pretrained weights
        """
        super(FlowerClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load backbone model
        if model_name == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
            
        elif model_name == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
            
        elif model_name == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
            
        elif model_name == "resnet101":
            self.backbone = models.resnet101(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
            
        elif model_name == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
            
        elif model_name == "efficientnet_b3":
            self.backbone = models.efficientnet_b3(pretrained=pretrained)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
            
        elif model_name == "vgg16":
            self.backbone = models.vgg16(pretrained=pretrained)
            in_features = self.backbone.classifier[6].in_features
            self.backbone.classifier[6] = nn.Linear(in_features, num_classes)
            
        elif model_name == "densenet121":
            self.backbone = models.densenet121(pretrained=pretrained)
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(in_features, num_classes)
            
        else:
            raise ValueError(f"Model {model_name} not supported")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.backbone(x)


def create_model(
    model_name: str = "resnet50",
    num_classes: int = 102,
    pretrained: bool = True,
    device: Optional[torch.device] = None
) -> FlowerClassifier:
    """
    Create and initialize a flower classification model.
    
    Args:
        model_name: Name of the backbone architecture
        num_classes: Number of flower classes
        pretrained: Whether to use pretrained weights
        device: Device to load model on
        
    Returns:
        Initialized model
    """
    model = FlowerClassifier(model_name, num_classes, pretrained)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    return model
