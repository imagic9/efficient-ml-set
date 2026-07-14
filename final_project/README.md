# Final Project — Edge AI & On-Device Optimization

## Task

Prepare, port, and deploy a neural network for on-device inference on a Raspberry
Pi (RPi 4 or 5). Establish a baseline, then apply model- and inference-level
optimizations to improve FPS toward real time.

Core requirements:
- Runs natively on the Raspberry Pi (inference over a saved dataset is fine).
- Measure on-device latency, FPS, and resource usage for baseline vs optimized.
- Apply optimizations (pruning, distillation, ONNX backends, bottleneck removal).
- The optimized inference engine is written in **C++**; Python is limited to
  training and model conversion.

Deliverables: codebase + a slide-deck presentation. Graded on the engineering
process and depth of analysis (model/optimization strategy, C++ inference,
benchmarking, results analysis).

## Contents

- [`Docker_VSCode/`](Docker_VSCode/) — reference dev-container and example pipeline
  provided with the course: PyTorch → ONNX → C++ inference (ONNX Runtime + OpenCV),
  with an ARM64 VS Code dev-container so the C++ side can be built and tested
  without physical Raspberry Pi hardware.

## Status

In progress. The channel-pruning work from `hw1/` (real MAC reduction) feeds
directly into the model-optimization stage here.
