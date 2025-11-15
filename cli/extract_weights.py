#!/usr/bin/env python3
"""
Extract model weights from PyTorch checkpoint.
Remove optimizer/scheduler states to reduce file size by ~66%.
"""

import argparse
import os
import torch
from pathlib import Path


def get_file_size_mb(path):
    """Get file size in MB."""
    return os.path.getsize(path) / (1024**2)


def extract_weights(checkpoint_path, output_path=None, keep_config=True):
    """
    Extract model weights from checkpoint.
    
    Args:
        checkpoint_path: Path to full checkpoint
        output_path: Output path (default: adds _weights_only suffix)
        keep_config: If True, keep config/epoch/metrics for inference
    
    Returns:
        Path to saved weights file
    """
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract model weights
    if 'model_state_dict' not in checkpoint:
        raise ValueError("Checkpoint must contain 'model_state_dict' key")
    
    model_weights = checkpoint['model_state_dict']
    
    # Prepare output
    if output_path is None:
        path = Path(checkpoint_path)
        output_path = path.parent / f"{path.stem}_weights_only{path.suffix}"
    
    # Save based on keep_config flag
    if keep_config:
        # Keep minimal info for inference
        save_dict = {
            'model_state_dict': model_weights,
        }
        # Add optional metadata if available
        for key in ['config', 'epoch', 'best_val_acc', 'best_val_loss']:
            if key in checkpoint:
                save_dict[key] = checkpoint[key]
    else:
        # Only weights (for model zoo / transfer learning)
        save_dict = model_weights
    
    # Save
    print(f"Saving weights to: {output_path}")
    torch.save(save_dict, output_path)
    
    # Report size reduction
    original_size = get_file_size_mb(checkpoint_path)
    new_size = get_file_size_mb(output_path)
    saved = original_size - new_size
    saved_pct = (saved / original_size) * 100
    
    print(f"\n{'='*60}")
    print(f"Original checkpoint: {original_size:.2f} MB")
    print(f"Extracted weights:   {new_size:.2f} MB")
    print(f"Space saved:         {saved:.2f} MB ({saved_pct:.1f}%)")
    print(f"{'='*60}\n")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract model weights from PyTorch checkpoint"
    )
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to checkpoint file'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output path (default: adds _weights_only suffix)'
    )
    parser.add_argument(
        '--weights-only',
        action='store_true',
        help='Save only weights without any metadata'
    )
    parser.add_argument(
        '--delete-original',
        action='store_true',
        help='Delete original checkpoint after extraction (use with caution!)'
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return 1
    
    # Extract weights
    try:
        output_path = extract_weights(
            args.checkpoint,
            args.output,
            keep_config=not args.weights_only
        )
        
        # Delete original if requested
        if args.delete_original:
            print(f"⚠️  Deleting original checkpoint: {args.checkpoint}")
            os.remove(args.checkpoint)
            print("✓ Original deleted")
        
        print(f"✓ Success! Weights saved to: {output_path}")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
