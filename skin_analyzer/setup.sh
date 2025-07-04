#!/bin/bash

# Skin Analyzer Setup Script
# This script sets up the environment and downloads dependencies

echo "=========================================="
echo "Setting up Skin Type Analyzer"
echo "=========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.7 or higher."
    exit 1
fi

echo "Python 3 found: $(python3 --version)"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv skin_analyzer_env

# Activate virtual environment
echo "Activating virtual environment..."
source skin_analyzer_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "source skin_analyzer_env/bin/activate"
echo ""
echo "Then you can:"
echo "1. Create dataset: python utils/dataset_downloader.py"
echo "2. Train model: python models/train_model.py"
echo "3. Make predictions: python predict.py --image path/to/image.jpg"
echo "4. Run demo: python demo.py"
echo ""
echo "To deactivate the environment, run: deactivate"
