"""
Inference script for flower recognition.
"""
import os
import argparse
import yaml
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from src.model import create_model
from src.dataset import get_val_transforms


def load_model(checkpoint_path: str, config: dict, device: torch.device):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Configuration dictionary
        device: Device to load model on
        
    Returns:
        Loaded model in evaluation mode
    """
    model = create_model(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=False,
        device=device
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def predict_image(
    model: torch.nn.Module,
    image_path: str,
    transform,
    device: torch.device,
    class_names: List[str] = None,
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Predict the class of a single image.
    
    Args:
        model: Trained model
        image_path: Path to image file
        transform: Image transformation pipeline
        device: Device to run inference on
        class_names: List of class names
        top_k: Number of top predictions to return
        
    Returns:
        List of (class_name, probability) tuples
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
    
    # Get top k predictions
    top_probs, top_indices = torch.topk(probabilities, top_k)
    top_probs = top_probs.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]
    
    # Format results
    results = []
    for idx, prob in zip(top_indices, top_probs):
        if class_names:
            class_name = class_names[idx]
        else:
            class_name = f"Class {idx}"
        results.append((class_name, prob))
    
    return results


def visualize_prediction(
    image_path: str,
    predictions: List[Tuple[str, float]],
    save_path: str = None
):
    """
    Visualize image with predictions.
    
    Args:
        image_path: Path to image file
        predictions: List of (class_name, probability) tuples
        save_path: Optional path to save visualization
    """
    # Load image
    image = Image.open(image_path)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display image
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('Input Image')
    
    # Display predictions
    classes = [pred[0] for pred in predictions]
    probs = [pred[1] for pred in predictions]
    
    y_pos = range(len(classes))
    ax2.barh(y_pos, probs)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(classes)
    ax2.invert_yaxis()
    ax2.set_xlabel('Probability')
    ax2.set_title('Top Predictions')
    ax2.set_xlim([0, 1])
    
    # Add probability values on bars
    for i, prob in enumerate(probs):
        ax2.text(prob, i, f' {prob:.3f}', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Inference for flower recognition')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of top predictions to show')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize predictions')
    parser.add_argument('--save-viz', type=str, default=None,
                        help='Path to save visualization')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, config, device)
    
    # Get transforms
    transform = get_val_transforms(config['data']['image_size'])
    
    # Predict
    print(f"Predicting for image: {args.image}")
    predictions = predict_image(
        model,
        args.image,
        transform,
        device,
        top_k=args.top_k
    )
    
    # Print results
    print("\nTop predictions:")
    for i, (class_name, prob) in enumerate(predictions, 1):
        print(f"{i}. {class_name}: {prob:.4f}")
    
    # Visualize if requested
    if args.visualize or args.save_viz:
        visualize_prediction(args.image, predictions, args.save_viz)


if __name__ == '__main__':
    main()
