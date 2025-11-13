"""
Loss functions for Flower Recognition.

Includes standard and advanced loss functions for training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Paper: Focal Loss for Dense Object Detection (https://arxiv.org/abs/1708.02002)
    
    Args:
        alpha: Weighting factor in [0, 1] to balance positive/negative examples
               or a list of weights for each class
        gamma: Focusing parameter for modulating loss (gamma >= 0)
        reduction: Specifies reduction to apply to output ('none', 'mean', 'sum')
        label_smoothing: Label smoothing factor (0.0 = no smoothing)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predictions of shape (N, C) where C = number of classes
            targets: Ground truth labels of shape (N,)
            
        Returns:
            Computed loss
        """
        # Get log probabilities
        ce_loss = F.cross_entropy(
            inputs, 
            targets, 
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        
        # Get probabilities
        p = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross entropy loss with label smoothing.
    
    Args:
        smoothing: Label smoothing factor (default: 0.1)
        reduction: Specifies reduction to apply to output
    """
    
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing cross entropy loss.
        
        Args:
            inputs: Predictions of shape (N, C)
            targets: Ground truth labels of shape (N,)
            
        Returns:
            Computed loss
        """
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Get one-hot encoded targets
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(-1)).float()
        
        # Apply label smoothing
        targets_smooth = (1 - self.smoothing) * targets_one_hot + \
                        self.smoothing / inputs.size(-1)
        
        # Compute loss
        loss = -(targets_smooth * log_probs).sum(dim=-1)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def get_loss_function(cfg):
    """
    Get loss function based on configuration.
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        Loss function instance
    """
    loss_name = cfg.training.get('loss_function', 'ce')
    
    if loss_name == 'focal':
        return FocalLoss(
            alpha=cfg.training.get('focal_alpha', 0.25),
            gamma=cfg.training.get('focal_gamma', 2.0),
            label_smoothing=cfg.training.get('label_smoothing', 0.0)
        )
    elif loss_name == 'lsce':
        return LabelSmoothingCrossEntropy(
            smoothing=cfg.training.get('label_smoothing', 0.1)
        )
    else:  # Default: CrossEntropy
        return nn.CrossEntropyLoss(
            label_smoothing=cfg.training.get('label_smoothing', 0.0)
        )


__all__ = [
    'FocalLoss',
    'LabelSmoothingCrossEntropy',
    'get_loss_function'
]
