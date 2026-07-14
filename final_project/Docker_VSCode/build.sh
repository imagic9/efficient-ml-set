#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

echo "Configuring CMake project in '${BUILD_DIR}' (type: ${BUILD_TYPE})..."
cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"

echo "Building target 'inference'..."
cmake --build "${BUILD_DIR}" --target inference -j"$(nproc)"

echo "Build complete. Binary: ${BUILD_DIR}/inference"
