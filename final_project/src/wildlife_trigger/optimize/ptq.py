#!/usr/bin/env python3
"""Static post-training quantization to INT8 QDQ — the M1 machinery.

DESIGN §8.1 fixes the representation: ONNX Runtime static quantization, S8S8 QDQ
primary, per-channel signed INT8 weights. S8S8 rather than U8S8 because the target
is ARM64, whose dot-product kernels (`asimddp` SDOT) are signed; U8S8 is the x86
VNNI recommendation and would pick a different kernel path on the Pi.

**Calibration data decides what this produces.** DESIGN §8.1 calibrates M1 on 1,024
stratified CCT-20 training images. This module does not choose the data — the
caller passes a reader — precisely so that the P0 spike can drive it with
synthetic tensors without that shortcut leaking into M1. A synthetically
calibrated model is a toolchain artifact whose accuracy is meaningless; see
`--synthetic-calibration` below, which says so at the point of use.

Usage (spike only):
    python -m wildlife_trigger.optimize.ptq --input m0.onnx --output m1.onnx \
        --synthetic-calibration 32
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from wildlife_trigger.models.export import describe

# DESIGN §8.1 tests all three on validation. The spike only proves they run.
CALIBRATION_METHODS = {
    "minmax": CalibrationMethod.MinMax,
    "entropy": CalibrationMethod.Entropy,
    "percentile": CalibrationMethod.Percentile,
}


class ArrayCalibrationReader(CalibrationDataReader):
    """Feed pre-built batches to the ORT calibrator.

    ORT consumes the reader exactly once and never rewinds it. Holding the
    sequence and rebuilding the iterator in `rewind` keeps the reader reusable
    across calibration methods, which DESIGN §8.1 compares on the same data — a
    one-shot generator would silently calibrate the second method on nothing.
    """

    def __init__(self, batches: Sequence[dict[str, np.ndarray]]):
        self._batches = list(batches)
        self._iterator: Iterator[dict[str, np.ndarray]] = iter(self._batches)

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._iterator, None)

    def rewind(self) -> None:
        self._iterator = iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


def synthetic_batches(
    count: int,
    input_name: str = "input",
    shape: tuple[int, ...] = (1, 3, 224, 224),
    seed: int = 0,
) -> list[dict[str, np.ndarray]]:
    """Random calibration batches — for toolchain proof only, never for M1.

    Deliberately normal(0,1): that is roughly the range ImageNet-normalised pixels
    occupy, so the activation ranges the calibrator observes are plausible enough
    to exercise every code path. They are still not this dataset's ranges, so the
    resulting scales are not M1's scales and the resulting accuracy is not a
    result.
    """
    rng = np.random.default_rng(seed)
    return [
        {input_name: rng.standard_normal(shape, dtype=np.float32)}
        for _ in range(count)
    ]


def model_input_name(path: Path) -> str:
    """Read the graph's input name rather than assuming the export used ours."""
    model = onnx.load(str(path), load_external_data=False)
    initializers = {init.name for init in model.graph.initializer}
    inputs = [i.name for i in model.graph.input if i.name not in initializers]
    if len(inputs) != 1:
        raise ValueError(f"expected exactly one graph input in {path}, got {inputs}")
    return inputs[0]


def quantize(
    input_model: Path,
    output_model: Path,
    reader: CalibrationDataReader,
    calibration_method: str = "minmax",
    per_channel: bool = True,
) -> dict:
    """Quantize `input_model` to S8S8 QDQ INT8 and describe the result.

    `quant_pre_process` first: ORT's quantizer needs inferred shapes to decide what
    it may quantize, and skipping it leaves nodes silently in FP32 — which would
    then look like a hardware or opset finding instead of a missing preprocessing
    step.
    """
    if calibration_method not in CALIBRATION_METHODS:
        raise ValueError(
            f"unknown calibration method {calibration_method!r}; "
            f"expected one of {sorted(CALIBRATION_METHODS)}"
        )

    output_model.parent.mkdir(parents=True, exist_ok=True)
    prepared = output_model.with_suffix(".prepared.onnx")
    quant_pre_process(str(input_model), str(prepared), skip_symbolic_shape=False)

    quantize_static(
        model_input=str(prepared),
        model_output=str(output_model),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        # S8S8: signed activations and signed weights, matching ARM64 SDOT.
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        calibrate_method=CALIBRATION_METHODS[calibration_method],
        # reduce_range is an x86 VNNI saturation workaround that costs a bit of
        # precision. On ARM64 it buys nothing, so it stays off.
        reduce_range=False,
    )

    description = describe(output_model)
    description["quantization"] = {
        "format": "QDQ",
        "activation_type": "QInt8",
        "weight_type": "QInt8",
        "scheme": "S8S8",
        "per_channel": per_channel,
        "calibration_method": calibration_method,
        "calibration_batches": len(reader) if hasattr(reader, "__len__") else None,
    }
    return description


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--synthetic-calibration",
        type=int,
        metavar="N",
        required=True,
        help="Calibrate on N random batches. Toolchain proof only: the resulting "
        "model's accuracy is meaningless and it is not M1. M1 calibrates on 1,024 "
        "stratified CCT-20 images (DESIGN §8.1).",
    )
    parser.add_argument(
        "--calibration-method", default="minmax", choices=sorted(CALIBRATION_METHODS)
    )
    parser.add_argument("--describe-json", type=Path)
    args = parser.parse_args()

    reader = ArrayCalibrationReader(
        synthetic_batches(args.synthetic_calibration, model_input_name(args.input))
    )
    description = quantize(
        args.input, args.output, reader, calibration_method=args.calibration_method
    )
    description["calibration_source"] = "SYNTHETIC — toolchain proof, not M1"

    print(json.dumps(description, indent=2))
    if args.describe_json:
        args.describe_json.parent.mkdir(parents=True, exist_ok=True)
        args.describe_json.write_text(json.dumps(description, indent=2) + "\n")
        print(f"wrote {args.describe_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
