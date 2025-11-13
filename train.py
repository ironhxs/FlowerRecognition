"""
Training script for Flower Recognition model.

This script handles the complete training pipeline including:
- Data loading and augmentation
- Model training with mixed precision
- Validation and checkpointing
- TensorBoard logging
"""

import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from datasets import create_dataloaders
from models import build_model, get_model_size_mb, get_loss_function


class Trainer:
    """Trainer class for flower recognition model."""
    
    def __init__(self, cfg: DictConfig):
        """
        Initialize trainer.
        
        Args:
            cfg: Hydra configuration
        """
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
        
        # Set random seed
        self.set_seed(cfg.seed)
        
        # Create output directories
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)
        
        # Initialize TensorBoard
        self.writer = SummaryWriter(log_dir=cfg.log_dir)
        
        # Build model
        self.model = build_model(cfg).to(self.device)
        
        # Check model size
        model_size_mb = get_model_size_mb(self.model)
        print(f"Model size: {model_size_mb:.2f} MB")
        if model_size_mb > 500:
            print("WARNING: Model size exceeds 500MB competition limit!")
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(cfg)
        
        # Loss function (supports CrossEntropy, Focal Loss, etc.)
        self.criterion = get_loss_function(cfg)
        
        # Optimizer
        self.optimizer = self.build_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self.build_scheduler()
        
        # Mixed precision scaler
        self.scaler = GradScaler() if cfg.training.use_amp else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def build_optimizer(self) -> optim.Optimizer:
        """Build optimizer from config."""
        opt_cfg = self.cfg.training.optimizer
        
        if opt_cfg.name.lower() == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=opt_cfg.lr,
                weight_decay=opt_cfg.weight_decay,
                betas=opt_cfg.betas
            )
        elif opt_cfg.name.lower() == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=opt_cfg.lr,
                momentum=opt_cfg.get('momentum', 0.9),
                weight_decay=opt_cfg.weight_decay,
                nesterov=opt_cfg.get('nesterov', True)
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_cfg.name}")
        
        return optimizer
    
    def build_scheduler(self):
        """Build learning rate scheduler."""
        sched_cfg = self.cfg.training.scheduler
        
        if sched_cfg.name.lower() == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.training.epochs - sched_cfg.warmup_epochs,
                eta_min=sched_cfg.min_lr
            )
        elif sched_cfg.name.lower() == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_cfg.get('step_size', 10),
                gamma=sched_cfg.get('gamma', 0.1)
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self) -> dict:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}/{self.cfg.training.epochs}')
        
        for batch_idx, (images, labels, _) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.cfg.training.get('clip_grad_norm', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.training.clip_grad_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                
                if self.cfg.training.get('clip_grad_norm', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.training.clip_grad_norm
                    )
                
                self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def validate(self) -> dict:
        """Validate model."""
        if self.val_loader is None:
            return {'loss': 0.0, 'accuracy': 0.0}
        
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            
            for images, labels, _ in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': total_loss / (len(pbar)),
                    'acc': 100. * correct / total
                })
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': OmegaConf.to_container(self.cfg, resolve=True)
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(
            self.cfg.checkpoint_dir,
            f'checkpoint_epoch_{self.current_epoch}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.cfg.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model with accuracy: {self.best_val_acc:.2f}%")
    
    def train(self):
        """Main training loop."""
        print(f"\nStarting training on {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(self.cfg.training.epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Log metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('Train/Accuracy', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('Val/Accuracy', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Print summary
            print(f"\nEpoch {epoch + 1}/{self.cfg.training.epochs}")
            print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save checkpoint
            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if (epoch + 1) % self.cfg.training.save_every == 0 or is_best:
                self.save_checkpoint(is_best)
            
            # Early stopping
            if self.patience_counter >= self.cfg.training.early_stopping.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        self.writer.close()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""
    print("="*60)
    print("Flower Recognition Training")
    print("="*60)
    print(OmegaConf.to_yaml(cfg))
    print("="*60)
    
    # Create trainer and train
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == '__main__':
    main()
