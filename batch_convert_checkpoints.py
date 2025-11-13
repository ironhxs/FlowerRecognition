#!/usr/bin/env python3
"""
Batch convert all checkpoints in a directory to inference format.
Saves ~66% storage space.
"""

import argparse
import os
from pathlib import Path
import torch
from tqdm import tqdm


def convert_checkpoint(input_path, output_dir=None, suffix='_inference'):
    """Convert single checkpoint to inference format."""
    # Load checkpoint
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
    
    # Check if already in inference format
    if 'optimizer_state_dict' not in checkpoint and 'scheduler_state_dict' not in checkpoint:
        return None, "already_inference"
    
    # Extract inference data
    inference_checkpoint = {
        'model_state_dict': checkpoint['model_state_dict'],
    }
    
    # Add optional metadata
    for key in ['config', 'epoch', 'best_val_acc', 'best_val_loss']:
        if key in checkpoint:
            inference_checkpoint[key] = checkpoint[key]
    
    # Determine output path
    input_file = Path(input_path)
    if output_dir:
        output_path = Path(output_dir) / f"{input_file.stem}{suffix}{input_file.suffix}"
    else:
        output_path = input_file.parent / f"{input_file.stem}{suffix}{input_file.suffix}"
    
    # Save
    torch.save(inference_checkpoint, output_path)
    
    # Calculate savings
    original_size = os.path.getsize(input_path) / (1024**2)
    new_size = os.path.getsize(output_path) / (1024**2)
    saved = original_size - new_size
    
    return output_path, saved


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert checkpoints to inference format"
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='Directory containing checkpoint files'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: same as input)'
    )
    parser.add_argument(
        '--suffix',
        type=str,
        default='_inference',
        help='Suffix for output files (default: _inference)'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.pt',
        help='File pattern to match (default: *.pt)'
    )
    parser.add_argument(
        '--skip-best',
        action='store_true',
        help='Skip files with "best" in name (already converted)'
    )
    
    args = parser.parse_args()
    
    # Find checkpoint files
    input_dir = Path(args.input_dir)
    checkpoint_files = list(input_dir.glob(args.pattern))
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {input_dir}")
        return 1
    
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
    # Filter if requested
    if args.skip_best:
        checkpoint_files = [f for f in checkpoint_files if 'best' not in f.stem]
        print(f"Skipping 'best' files, processing {len(checkpoint_files)} files")
    
    # Create output directory if needed
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert files
    total_saved = 0
    converted = 0
    skipped = 0
    
    for checkpoint_file in tqdm(checkpoint_files, desc="Converting"):
        try:
            output_path, result = convert_checkpoint(
                checkpoint_file,
                args.output_dir,
                args.suffix
            )
            
            if result == "already_inference":
                skipped += 1
                tqdm.write(f"⊘ Skipped (already inference): {checkpoint_file.name}")
            else:
                total_saved += result
                converted += 1
                tqdm.write(f"✓ Converted: {checkpoint_file.name} (saved {result:.1f} MB)")
                
        except Exception as e:
            tqdm.write(f"✗ Error processing {checkpoint_file.name}: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Conversion complete:")
    print(f"  Converted: {converted} files")
    print(f"  Skipped:   {skipped} files")
    print(f"  Total space saved: {total_saved:.1f} MB")
    print(f"{'='*60}")
    
    return 0


if __name__ == '__main__':
    exit(main())
