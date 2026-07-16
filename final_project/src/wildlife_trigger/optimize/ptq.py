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

The real M1 path is `--config`: it reads the frozen calibration manifest (built
once by `optimize.calibration_manifest`, hash-pinned in the config), refuses a
source ONNX or manifest whose bytes moved since the config was written, and
produces one candidate directory per calibration method — quantized graph,
description, and the `ort_coverage` evidence (optimized graph, ORT profile,
operator/data-type verdict) that DESIGN §8.1 requires saved per candidate.

Usage:
    python -m wildlife_trigger.optimize.ptq --config configs/optimize/m1_ptq.yaml

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
import onnxruntime
import yaml
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from wildlife_trigger.models.export import describe
from wildlife_trigger.runs import atomic_write_json, sha256_file

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


class ManifestCalibrationReader(CalibrationDataReader):
    """The frozen calibration manifest as batch-1 tensors, in manifest order.

    Batch 1 because the exported graph's batch dimension is static 1 (DESIGN
    §9.1 infers one frame at a time, and `models.export` pins the shape); the
    calibrators are histogram/range collectors, so batch size changes memory,
    never the resulting scales.

    Decoding is lazy — the dataset falls back from cache to JPEG per record —
    so three methods over 1,024 images cost three passes, not one giant pinned
    array. `rewind` restarts the same order; DESIGN §8.1 compares the methods
    on identical data, and identical includes the order they saw it in.
    """

    def __init__(self, dataset, input_name: str):
        self._dataset = dataset
        self._input_name = input_name
        self._index = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._index >= len(self._dataset):
            return None
        image = self._dataset[self._index]["image"].numpy()[None, ...]
        self._index += 1
        return {self._input_name: image}

    def rewind(self) -> None:
        self._index = 0

    def __len__(self) -> int:
        return len(self._dataset)


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


REQUIRED_CONFIG_KEYS = (
    "source_run_id",
    "source_onnx",
    "source_onnx_sha256",
    "calibration_manifest",
    "calibration_manifest_sha256",
    "calibration_images",
    "images_dir",
    "supplement_dir",
    "cache_dir",
    "classes_config",
    "width",
    "height",
    "methods",
    "output_root",
)


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"{path} lacks required keys: {missing}")
    unknown = sorted(set(config["methods"]) - set(CALIBRATION_METHODS))
    if unknown:
        raise ValueError(
            f"unknown calibration methods {unknown}; expected a subset of "
            f"{sorted(CALIBRATION_METHODS)}"
        )
    return config


def calibration_dataset(config: dict):
    """The frozen manifest as a deterministic eval-mode dataset, hash-checked.

    Torch-side imports are local: this function only runs where the training
    environment lives, while the rest of the module stays importable next to a
    bare onnxruntime.
    """
    from ..data.dataset import WildlifeDataset, load_class_names
    from ..data.preprocess import PreprocessConfig

    manifest = Path(config["calibration_manifest"])
    measured = sha256_file(manifest)
    if measured != config["calibration_manifest_sha256"]:
        raise RuntimeError(
            f"{manifest} hashes to {measured[:12]}… but the config pins "
            f"{config['calibration_manifest_sha256'][:12]}…; the calibration data "
            "is frozen (DESIGN §8.1) and this file is not it"
        )
    dataset = WildlifeDataset(
        manifest,
        load_class_names(Path(config["classes_config"])),
        PreprocessConfig(width=config["width"], height=config["height"]),
        Path(config["images_dir"]),
        cache_root=Path(config["cache_dir"]),
        train=False,  # deterministic: no augmentation may touch calibration
        image_root_overrides={"empty_supplement": Path(config["supplement_dir"])},
    )
    if len(dataset) != config["calibration_images"]:
        raise RuntimeError(
            f"{manifest} holds {len(dataset)} images, the config promises "
            f"{config['calibration_images']}; refusing to calibrate on a "
            "different corpus than the one registered"
        )
    return dataset


def generate_candidates(config: dict) -> dict:
    """One candidate directory per calibration method, with its evidence.

    Every candidate starts from the identical hash-verified M0 ONNX and reads
    the identical manifest in the identical order — so the only degree of
    freedom between the directories is the calibration method, which is the
    comparison DESIGN §8.1 asks for.
    """
    from ..validate import ort_coverage

    source = Path(config["source_onnx"])
    measured = sha256_file(source)
    if measured != config["source_onnx_sha256"]:
        raise RuntimeError(
            f"{source} hashes to {measured[:12]}… but the config pins "
            f"{config['source_onnx_sha256'][:12]}…; whatever this file is, it is "
            "not the M0 every candidate must start from"
        )

    dataset = calibration_dataset(config)
    input_name = model_input_name(source)
    output_root = Path(config["output_root"])
    summary: dict[str, dict] = {}

    for method in config["methods"]:
        candidate_dir = output_root / method
        model_path = candidate_dir / "model.onnx"
        reader = ManifestCalibrationReader(dataset, input_name)
        description = quantize(
            source,
            model_path,
            reader,
            calibration_method=method,
            per_channel=bool(config.get("per_channel", True)),
        )

        label = f"m1_ptq_{method}"
        coverage = ort_coverage.analyse(model_path, candidate_dir, label)
        atomic_write_json(candidate_dir / "coverage.json", coverage)

        candidate = {
            "tool": "wildlife_trigger.optimize.ptq",
            "design": "8.1",
            "candidate_id": f"d1_{label}",
            "model_id": "M1-candidate",
            "kind": "int8_ptq",
            "method": method,
            "source_run_id": config["source_run_id"],
            "source_onnx": {"path": str(source), "sha256": measured},
            "calibration": {
                "manifest": str(config["calibration_manifest"]),
                "sha256": config["calibration_manifest_sha256"],
                "images": len(dataset),
                "order": "manifest order, batch 1",
            },
            "input": {"width": config["width"], "height": config["height"]},
            "onnxruntime_version": onnxruntime.__version__,
            "model": description,
            "integer_execution": coverage["verdict"]["integer_execution"],
        }
        atomic_write_json(candidate_dir / "candidate.json", candidate)

        summary[method] = {
            "candidate_dir": str(candidate_dir),
            "model_sha256": description["sha256"],
            "size_bytes": description["size_bytes"],
            "integer_execution": candidate["integer_execution"],
        }
        print(
            f"{method}: {description['size_bytes']:,} bytes, integer_execution="
            f"{candidate['integer_execution']} -> {candidate_dir}"
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, help="The real M1 path: configs/optimize/m1_ptq.yaml."
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--synthetic-calibration",
        type=int,
        metavar="N",
        help="Calibrate on N random batches. Toolchain proof only: the resulting "
        "model's accuracy is meaningless and it is not M1. M1 calibrates on 1,024 "
        "stratified CCT-20 images (DESIGN §8.1).",
    )
    parser.add_argument(
        "--calibration-method", default="minmax", choices=sorted(CALIBRATION_METHODS)
    )
    parser.add_argument("--describe-json", type=Path)
    args = parser.parse_args()

    if args.config:
        generate_candidates(load_config(args.config))
        return 0

    if not (args.input and args.output and args.synthetic_calibration):
        parser.error(
            "either --config (the M1 path) or all of --input/--output/"
            "--synthetic-calibration (the P0 spike path) is required"
        )

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
