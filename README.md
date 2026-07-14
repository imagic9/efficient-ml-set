# Efficient ML — SET University

Coursework for the **Efficient ML** course at SET University: deep-learning model
optimization for fast inference and deployment on edge devices (laptops,
smartphones, microcontrollers, neural accelerators).

Topics across the course: pruning, quantization, neural architecture search,
knowledge distillation, TinyML on microcontrollers, and domain-specific
optimization — ending with an on-device deployment project on a Raspberry Pi.

## Structure

| Folder | Contents |
|--------|----------|
| [`hw1/`](hw1/) | HW1 — VGG11 iterative pruning on CIFAR-10 |
| `hw2/` … | later homeworks (one folder each) |
| [`final_project/`](final_project/) | Edge AI / on-device optimization on Raspberry Pi |

Each homework folder has its own `README.md` with the task, approach, results and
run instructions. Notebooks are written in Ukrainian.

## Stack

PyTorch, torchvision, `torch-pruning`. Experiments were run on an NVIDIA GPU;
training scripts are plain Python and reproducible via each folder's runners.

## Notes

Model checkpoints (`*.pt`), datasets, and instructor-provided assignment files are
intentionally excluded from the repository (see `.gitignore`); all results are
captured in the committed notebooks and JSON metrics.
