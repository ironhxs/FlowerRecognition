#!/bin/bash
# Setup script for Flower Recognition project

echo "=========================================="
echo "Flower Recognition Setup"
echo "=========================================="

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed"
    exit 1
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Check PyTorch installation
echo ""
echo "Checking PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p data/train data/test
mkdir -p results/checkpoints results/logs
mkdir -p outputs

echo ""
echo "=========================================="
echo "Setup completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Place your training data in data/train/"
echo "2. Place your training CSV in data/train.csv"
echo "3. Place your test data in data/test/"
echo "4. Run 'python train.py' to start training"
echo ""
echo "Or generate sample data for testing:"
echo "  python generate_sample_data.py"
echo ""
