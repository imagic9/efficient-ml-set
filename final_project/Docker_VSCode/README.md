# MobileNet ONNX Inference Project

A simple project demonstrating PyTorch to ONNX model conversion and C++ inference using ONNX Runtime.

## Project Structure

- `convert_to_onnx.py` - Python script to convert MobileNet V2 to ONNX format
- `main.cpp` - C++ application for loading ONNX model and running inference
- `CMakeLists.txt` - CMake build configuration
- `requirements.txt` - Python dependencies
- `install_dependencies.sh` - Installation script for all dependencies

## Prerequisites

- Python 3.6+
- C++ compiler (g++ or clang)
- CMake 3.10+

## Installation

Run the installation script to install all dependencies:

```bash
chmod +x install_dependencies.sh
./install_dependencies.sh
```

This script will:
- Install Python dependencies (PyTorch, TorchVision, ONNX)
- Install C++ dependencies (CMake, OpenCV, ONNX Runtime)
- Download a test image

## Usage

### Step 1: Convert MobileNet to ONNX

```bash
python3 convert_to_onnx.py
```

This will create `mobilenet_v2.onnx` in the current directory.

### Step 2: Build C++ Application

Option A (recommended): use the build script.

```bash
chmod +x build.sh
./build.sh
```

Option B: run CMake commands directly.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target inference -j"$(nproc)"
```

### Step 3: Run Inference

```bash
./build/inference
```

You can also run it from inside the build directory:

```bash
cd build
./inference
```

The application will:
1. Load the ONNX model
2. Load and preprocess the test image
3. Run inference
4. Display the predicted class index and confidence score

## Model Information

- **Model**: MobileNet V2 (ImageNet pretrained)
- **Input**: 224x224 RGB image
- **Output**: 1000 class probabilities (ImageNet classes)
- **Preprocessing**: 
  - Resize to 224x224
  - Normalize with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

## Notes

- The test image is automatically downloaded during installation
- You can replace `test_image.jpg` with any image you want to classify
- Class indices correspond to ImageNet class labels
