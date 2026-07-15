"""Wildlife camera-trap shutter trigger.

A full-frame MobileNetV2 classifier decides whether a target animal is present in
a saved camera-trap frame; a C++ ONNX Runtime application runs that decision on a
Raspberry Pi 5 and emits an emulated shutter signal.

This Python package covers training, optimization, export and evaluation only.
The deployed inference path is `cpp/`. See DESIGN.md for the execution contract.
"""

__version__ = "0.1.0"
