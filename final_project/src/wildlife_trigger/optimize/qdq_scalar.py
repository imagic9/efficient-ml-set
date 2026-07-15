#!/usr/bin/env python3
"""Make per-tensor QDQ scales rank-0 scalars, as ORT requires.

## The finding this repairs

`torch.onnx.export` lowers `fake_quantize_per_tensor_affine` to
QuantizeLinear/DequantizeLinear and passes the FakeQuantize module's `scale` and
`zero_point` buffers straight through. Those buffers have shape `[1]` — a rank-1
tensor holding one value — not rank 0. The ONNX spec calls the per-tensor form a
scalar, and ORT enforces it. Measured on gx10, 2026-07-15, ORT 1.27.0:

    FAIL : Node (/features/features.3/Add) Op (QLinearAdd)
    [TypeInferenceError] Scale and Zero-point must be a scalar

The model passes `onnx.checker.check_model` with `full_check=True` and still cannot
be loaded. That is why P0 requires ORT to actually run the graph: a structural
check would have called this artifact valid.

Note *which* node reported it. ORT had already decided to build a QLinearAdd — the
QDQ placement was right — and rejected it only on the scale's rank. A shape-`[1]`
scale is a one-line difference that costs the entire integer execution path.

## Why rank is the only thing changed

`del dims[:]` reinterprets the same single value as rank 0; the payload bytes are
untouched, so no scale, zero-point or weight changes value. Per-channel tensors
(rank 1, many elements, carried by nodes with an explicit `axis`) are left exactly
as they are — squeezing one would silently convert per-channel quantization into
per-tensor and lose accuracy invisibly.

Usage:
    python -m wildlife_trigger.optimize.qdq_scalar --input in.onnx --output out.onnx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx

QDQ_OPS = ("QuantizeLinear", "DequantizeLinear")

# Input positions defined by the ONNX spec for both QuantizeLinear and
# DequantizeLinear: (x, y_scale, y_zero_point). zero_point is optional.
SCALE_INPUT, ZERO_POINT_INPUT = 1, 2


class PerChannelAmbiguity(RuntimeError):
    """A single-element tensor on a node that also declares a per-channel axis."""


def _scalarize(tensor: onnx.TensorProto) -> bool:
    """Turn a shape-[1] tensor into a rank-0 scalar. Returns True if changed."""
    if list(tensor.dims) != [1]:
        return False
    del tensor.dims[:]
    return True


def scalarize_per_tensor_qdq(input_model: Path, output_model: Path) -> dict:
    """Rewrite every per-tensor QDQ scale/zero-point to rank 0.

    Both initializers and Constant nodes are handled: which one an exporter emits
    is its own implementation detail, and depending on that would make this fix
    quietly stop working after a torch upgrade.
    """
    model = onnx.load(str(input_model))
    graph = model.graph

    initializers = {init.name: init for init in graph.initializer}
    constants = {
        node.output[0]: attr.t
        for node in graph.node
        if node.op_type == "Constant"
        for attr in node.attribute
        if attr.name == "value"
    }

    changed: set[str] = set()
    per_channel_left: set[str] = set()

    for node in graph.node:
        if node.op_type not in QDQ_OPS:
            continue
        declares_axis = any(attr.name == "axis" for attr in node.attribute)

        for position in (SCALE_INPUT, ZERO_POINT_INPUT):
            if position >= len(node.input):
                continue
            name = node.input[position]
            if not name or name in changed or name in per_channel_left:
                continue

            tensor = initializers.get(name) or constants.get(name)
            if tensor is None:
                # A computed scale, which this project never produces. Left alone
                # rather than guessed at.
                continue

            element_count = 1
            for dim in tensor.dims:
                element_count *= dim

            if element_count > 1:
                per_channel_left.add(name)
                continue

            if declares_axis and element_count == 1 and list(tensor.dims) == [1]:
                # Per-channel over exactly one channel is indistinguishable from
                # per-tensor by shape alone. MobileNetV2 has no single-channel
                # convolution, so this never fires — but if it ever does, guessing
                # would silently change the quantization scheme.
                raise PerChannelAmbiguity(
                    f"{node.op_type} node {node.name!r} declares a per-channel "
                    f"axis, yet its input {name!r} has shape [1]. Cannot tell "
                    "per-tensor from single-channel per-channel by shape; refusing "
                    "to guess."
                )

            if _scalarize(tensor):
                changed.add(name)

    output_model.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(output_model))

    return {
        "input": str(input_model),
        "output": str(output_model),
        "scalarized_tensors": len(changed),
        "per_channel_tensors_left_alone": len(per_channel_left),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    print(json.dumps(scalarize_per_tensor_qdq(args.input, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
