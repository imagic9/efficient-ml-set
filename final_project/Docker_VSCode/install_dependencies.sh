#!/bin/bash

echo "Installing dependencies for ONNX MobileNet project..."

: "${ORT_VERSION:?Set ORT_VERSION to the provisional/P0-approved ONNX Runtime version}"

echo "=== Installing Python and dependencies ==="

if [ -f /etc/debian_version ]; then
    echo "Detected Debian/Ubuntu system"
    sudo apt-get update
    sudo apt-get install -y python3 python3-pip python3-venv python3-full
    sudo apt-get install -y cmake build-essential
    sudo apt-get install -y libopencv-dev
    
    echo "Installing ONNX Runtime..."
    ARCH=$(uname -m)
    if [ "$ARCH" = "x86_64" ]; then
        ONNX_PACKAGE="onnxruntime-linux-x64-${ORT_VERSION}.tgz"
        ONNX_DIR="onnxruntime-linux-x64-${ORT_VERSION}"
    elif [ "$ARCH" = "aarch64" ]; then
        ONNX_PACKAGE="onnxruntime-linux-aarch64-${ORT_VERSION}.tgz"
        ONNX_DIR="onnxruntime-linux-aarch64-${ORT_VERSION}"
    else
        echo "Unsupported architecture: $ARCH"
        exit 1
    fi
    
    wget "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ONNX_PACKAGE}"
    tar -xzf $ONNX_PACKAGE
    sudo cp -r $ONNX_DIR/include/* /usr/local/include/
    sudo cp -r $ONNX_DIR/lib/* /usr/local/lib/
    sudo ldconfig
    rm -rf $ONNX_DIR*
    
elif [ -f /etc/redhat-release ]; then
    echo "Detected RedHat/CentOS/Fedora system"
    sudo yum install -y python3 python3-pip
    sudo yum install -y cmake gcc-c++
    sudo yum install -y opencv-devel
    
    echo "Installing ONNX Runtime..."
    ARCH=$(uname -m)
    if [ "$ARCH" = "x86_64" ]; then
        ONNX_PACKAGE="onnxruntime-linux-x64-${ORT_VERSION}.tgz"
        ONNX_DIR="onnxruntime-linux-x64-${ORT_VERSION}"
    elif [ "$ARCH" = "aarch64" ]; then
        ONNX_PACKAGE="onnxruntime-linux-aarch64-${ORT_VERSION}.tgz"
        ONNX_DIR="onnxruntime-linux-aarch64-${ORT_VERSION}"
    else
        echo "Unsupported architecture: $ARCH"
        exit 1
    fi
    
    wget "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ONNX_PACKAGE}"
    tar -xzf $ONNX_PACKAGE
    sudo cp -r $ONNX_DIR/include/* /usr/local/include/
    sudo cp -r $ONNX_DIR/lib/* /usr/local/lib/
    sudo ldconfig
    rm -rf $ONNX_DIR*
    
else
    echo "Unsupported Linux distribution. Please install manually:"
    echo "  - Python 3 and pip"
    echo "  - CMake"
    echo "  - OpenCV development libraries"
    echo "  - ONNX Runtime"
    exit 1
fi

echo ""
echo "=== Installing Python packages ==="
bash install_python_deps.sh

echo ""
echo "=== Downloading test image ==="
wget -O test_image.jpg https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg

echo ""
echo "Installation complete!"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run: python3 convert_to_onnx.py"
echo "3. Build C++ project: mkdir build && cd build && cmake .. && make"
echo "4. Run inference: ./inference"
