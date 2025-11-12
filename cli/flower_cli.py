#!/usr/bin/env python3
"""
Command-line interface for Flower Recognition system.
"""

import click
from rich.console import Console
from rich.table import Table
import sys
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from models import list_recommended_models, RECOMMENDED_MODELS

console = Console()


@click.group()
def cli():
    """Flower Recognition AI Challenge - CLI Tool"""
    pass


@cli.command()
def models():
    """List available models."""
    console.print("\n[bold cyan]Available Models for Flower Recognition[/bold cyan]\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model Name", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Parameters", style="yellow")
    table.add_column("Accuracy", style="green")
    table.add_column("Speed", style="blue")
    
    for model_name, info in RECOMMENDED_MODELS.items():
        table.add_row(
            model_name,
            info['description'],
            info['params'],
            info['accuracy'],
            info['speed']
        )
    
    console.print(table)
    console.print()


@cli.command()
@click.option('--config', '-c', default='configs/config.yaml', help='Config file path')
def train(config):
    """Start training with specified config."""
    import subprocess
    
    console.print(f"\n[bold green]Starting training with config: {config}[/bold green]\n")
    
    cmd = ['python', 'train.py']
    subprocess.run(cmd)


@cli.command()
@click.option('--checkpoint', '-ckpt', required=True, help='Path to model checkpoint')
@click.option('--output', '-o', default='predictions.csv', help='Output CSV file')
@click.option('--tta', is_flag=True, help='Use test-time augmentation')
@click.option('--benchmark', is_flag=True, help='Benchmark inference speed')
def predict(checkpoint, output, tta, benchmark):
    """Generate predictions for test data."""
    import subprocess
    
    console.print(f"\n[bold green]Running inference[/bold green]")
    console.print(f"Checkpoint: {checkpoint}")
    console.print(f"Output: {output}\n")
    
    cmd = ['python', 'inference.py', '--checkpoint', checkpoint, '--output', output]
    
    if tta:
        cmd.append('--tta')
    if benchmark:
        cmd.append('--benchmark')
    
    subprocess.run(cmd)


@cli.command()
@click.option('--data-dir', '-d', default='./data', help='Data directory')
def prepare_data(data_dir):
    """Prepare and validate dataset."""
    import os
    import pandas as pd
    
    console.print(f"\n[bold cyan]Checking dataset at: {data_dir}[/bold cyan]\n")
    
    # Check training data
    train_csv = os.path.join(data_dir, 'train.csv')
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    if os.path.exists(train_csv):
        df = pd.read_csv(train_csv)
        console.print(f"✓ Training CSV found: {len(df)} samples")
        console.print(f"  Classes: {df['label'].nunique()}")
        console.print(f"  Samples per class: {len(df) // df['label'].nunique()}")
    else:
        console.print(f"✗ Training CSV not found at {train_csv}")
    
    if os.path.exists(train_dir):
        num_images = len([f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        console.print(f"✓ Training directory found: {num_images} images")
    else:
        console.print(f"✗ Training directory not found at {train_dir}")
    
    if os.path.exists(test_dir):
        num_images = len([f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        console.print(f"✓ Test directory found: {num_images} images")
    else:
        console.print(f"✗ Test directory not found at {test_dir}")
    
    console.print()


@cli.command()
def info():
    """Show system and competition information."""
    import torch
    
    console.print("\n[bold cyan]System Information[/bold cyan]\n")
    
    table = Table(show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="yellow")
    
    table.add_row("Python Version", sys.version.split()[0])
    table.add_row("PyTorch Version", torch.__version__)
    table.add_row("CUDA Available", str(torch.cuda.is_available()))
    
    if torch.cuda.is_available():
        table.add_row("CUDA Version", torch.version.cuda)
        table.add_row("GPU Name", torch.cuda.get_device_name(0))
        table.add_row("GPU Memory", f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    console.print(table)
    
    console.print("\n[bold cyan]Competition Requirements[/bold cyan]\n")
    
    req_table = Table(show_header=False)
    req_table.add_column("Requirement", style="cyan")
    req_table.add_column("Limit", style="yellow")
    
    req_table.add_row("Model Size", "< 500 MB")
    req_table.add_row("Inference Time", "< 100 ms per image")
    req_table.add_row("Number of Classes", "100")
    req_table.add_row("Image Size", "600 x 600")
    req_table.add_row("Submission Format", "CSV (UTF-8)")
    
    console.print(req_table)
    console.print()


if __name__ == '__main__':
    cli()
