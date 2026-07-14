#!/bin/bash

echo "Creating Python virtual environment..."
python3 -m venv venv

echo "Activating virtual environment and installing dependencies..."
source venv/bin/activate
pip install --upgrade pip

echo "Installing CPU-only PyTorch packages..."
pip uninstall -y torch torchvision torchaudio || true
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

echo "Installing remaining Python requirements..."
pip install -r requirements.txt

echo ""
echo "Python dependencies installed successfully in virtual environment!"
echo ""
echo "To use the virtual environment, run: source venv/bin/activate"
