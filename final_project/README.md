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

## What we are building

A Raspberry Pi device that **fires a wildlife photographer's camera only when the
animal they actually want walks into frame**. A motion sensor wakes the device, it
classifies the frame, and it triggers the main camera only on the target species —
then goes back to sleep. The photographer downloads a *module* for their animal
(`bobcat-v1`, `elephant-v1`).

This makes latency and energy the product itself rather than vanity metrics: every
millisecond saved is a better photograph, every millijoule is another night in the
field. The course project builds and measures the decision core:

```
is anything in frame?  →  what is it?  →  fire the shutter if it is the target
```

Key decisions in brief — a two-model cascade (a tiny gate rejects the ~70% of
empty triggers before the species model runs), MobileNetV2 on ONNX Runtime, a
crop-trained teacher distilled into a full-frame student, and Caltech Camera
Traps-20 with its cis/trans splits so we can answer *"will it still work at a
location it has never seen?"*.

**See [`DESIGN.md`](DESIGN.md) for the full design**: product framing, scope
boundaries, the quantified reasons behind each choice, metrics, benchmark
protocol, risks, and the roadmap beyond the course.

## Contents

- [`DESIGN.md`](DESIGN.md) — the living design note. Read this first.
- [`PLAN.md`](PLAN.md) — the work plan: dependency graph, phase gates, and what has
  to be true before the 5-day Pi trial is opened.
- [`Docker_VSCode/`](Docker_VSCode/) — reference dev-container and example pipeline
  provided with the course: PyTorch → ONNX → C++ inference (ONNX Runtime + OpenCV),
  with an ARM64 VS Code dev-container so the C++ side can be built and tested
  without physical Raspberry Pi hardware. We use the dev-container as our build
  environment and ONNX Runtime as our runtime; the bundled 145-line MobileNetV2
  demo is a toolchain smoke-test, not a starting template.

## Reuse from the homeworks

| From | Used for |
|---|---|
| `hw3/` — `DistillLoss`, `kd_train` | Distilling a crop-trained teacher into the full-frame student. Load-bearing: this is how we get the accuracy of cropping without running a detector on-device. |
| `hw1/` — `src/structured.py` channel pruning | Real MAC reduction (unstructured pruning would shrink the file without speeding up the runtime — kept as a measured negative result). |
| `hw2`/`hw3` — `src/qat.py` | The QAT loop, which already supports a teacher. The quantizer itself is replaced: HW2's K-Means centroid quantization compresses storage, whereas ONNX Runtime needs affine INT8 for integer arithmetic. |

## Status

Design agreed, implementation not started. Hardware is a rented Raspberry Pi 5
(Hostpro); all training, conversion and C++ development happen off-device, with
the on-device window reserved for final measurements.
</content>
