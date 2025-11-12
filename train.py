"""
Training script for flower recognition model.
"""
import os
import argparse
import yaml
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.model import create_model
from src.dataset import create_dataloaders
from src.utils import (
    save_checkpoint,
    load_checkpoint,
    calculate_accuracy,
    AverageMeter
)


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter = None
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        acc = calculate_accuracy(outputs, labels)
        
        # Update metrics
        losses.update(loss.item(), images.size(0))
        accuracies.update(acc, images.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{accuracies.avg:.2f}%'
        })
        
        # Log to tensorboard
        if writer is not None:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/Accuracy', acc, global_step)
    
    return {
        'loss': losses.avg,
        'accuracy': accuracies.avg
    }


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter = None
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Calculate accuracy
            acc = calculate_accuracy(outputs, labels)
            
            # Update metrics
            losses.update(loss.item(), images.size(0))
            accuracies.update(acc, images.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{accuracies.avg:.2f}%'
            })
    
    # Log to tensorboard
    if writer is not None:
        writer.add_scalar('Val/Loss', losses.avg, epoch)
        writer.add_scalar('Val/Accuracy', accuracies.avg, epoch)
    
    return {
        'loss': losses.avg,
        'accuracy': accuracies.avg
    }


def train(config: Dict[str, Any], args: argparse.Namespace):
    """Main training function."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, _ = create_dataloaders(
        train_dir=config['data']['train_path'],
        val_dir=config['data']['val_path'],
        batch_size=config['training']['batch_size'],
        image_size=config['data']['image_size'],
        num_workers=config['data']['num_workers']
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    model = create_model(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        device=device
    )
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    if config['training']['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs']
        )
    elif config['training']['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.1
        )
    else:
        scheduler = None
    
    # Tensorboard writer
    writer = None
    if config['logging']['tensorboard']:
        os.makedirs(config['logging']['log_dir'], exist_ok=True)
        writer = SummaryWriter(config['logging']['log_dir'])
    
    # Training loop
    best_acc = 0.0
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer)
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        print(f"Resumed from epoch {checkpoint['epoch']}")
    
    # Create checkpoint directory
    os.makedirs(config['checkpoint']['save_dir'], exist_ok=True)
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, epoch, writer
        )
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
            if writer is not None:
                writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Print epoch summary
        print(f"Train Loss: {train_metrics['loss']:.4f}, "
              f"Train Acc: {train_metrics['accuracy']:.2f}%")
        print(f"Val Loss: {val_metrics['loss']:.4f}, "
              f"Val Acc: {val_metrics['accuracy']:.2f}%")
        
        # Save checkpoint
        is_best = val_metrics['accuracy'] > best_acc
        if is_best:
            best_acc = val_metrics['accuracy']
        
        if (epoch + 1) % config['checkpoint']['save_freq'] == 0 or is_best:
            save_checkpoint(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'config': config
                },
                is_best=is_best,
                checkpoint_dir=config['checkpoint']['save_dir'],
                filename=f'checkpoint_epoch_{epoch}.pth'
            )
    
    print(f"\nTraining completed! Best validation accuracy: {best_acc:.2f}%")
    
    if writer is not None:
        writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train flower recognition model')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Train
    train(config, args)


if __name__ == '__main__':
    main()
