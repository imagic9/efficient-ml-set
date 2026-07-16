#!/usr/bin/env python3
"""The A4 smoke artifacts: a deterministic 16-output model, class map and policy.

PLAN A4 runs the whole vertical slice — JPEG -> C++ decode/preprocess -> ORT ->
policy -> SHUTTER_TRIGGER JSON — before any data exists and before any training. It
therefore needs a model with the *shape* of the real one and none of its meaning.

What this produces is deliberately honest about being fake:

  - the architecture is the real one (MobileNetV2, 16 outputs), so the slice
    exercises the kernels, tensor shapes and latencies M0 will actually have;
  - the backbone carries ImageNet weights and the head is randomly initialised from
    a fixed seed, so predictions are arbitrary but *reproducible*;
  - every artifact says so in a `provisional` field, because the one thing that
    must never happen is an A4 number reaching a results table.

The class map is provisional in a second, sharper sense: DESIGN §4 fixes the 16
class *names*, but their integer order comes from the CCT-20 annotations, which B1
freezes and which cannot be downloaded before Gate A. The order here is alphabetical
with `car`/`empty` appended — a placeholder chosen so that a mismatch with the real
order is loud (the names would disagree) rather than silent.

Usage:
    python -m wildlife_trigger.models.smoke --output-dir artifacts/smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from wildlife_trigger.models.export import P0_OPSET, export_onnx
from wildlife_trigger.models.mobilenet import (
    INPUT_SHAPE_CORE,
    build_mobilenet_v2,
    example_input,
)

# DESIGN §4: 14 animals + car + empty = the 16-way single-label task. `car` and
# `empty` are model classes but never selectable wildlife targets.
ANIMAL_CLASSES = (
    "badger",
    "bird",
    "bobcat",
    "cat",
    "coyote",
    "deer",
    "dog",
    "fox",
    "opossum",
    "rabbit",
    "raccoon",
    "rodent",
    "skunk",
    "squirrel",
)
NON_ANIMAL_CLASSES = ("car", "empty")
SMOKE_CLASSES = ANIMAL_CLASSES + NON_ANIMAL_CLASSES

# DESIGN §4 catalog: these three have no defensible operating point (badger has one
# validation image; deer and fox have none). The policy loader must reject them
# rather than invent a threshold. A4 carries the null so the C++ loader's rejection
# path has something real to reject.
NO_THRESHOLD_CLASSES = ("badger", "deer", "fox")

# Not calibrated — calibration is D-phase work on validation data that does not
# exist yet (DESIGN §6.3). 0.5 is the arbitrary midpoint, marked as such everywhere
# it appears.
PROVISIONAL_THRESHOLD = 0.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, payload: dict) -> str:
    """Write canonical JSON and return its SHA-256.

    `sort_keys` and a fixed separator: the class map's hash is bound into the policy,
    so the same content must serialise to the same bytes every time. Without that,
    re-running the exporter would invalidate a policy that had not changed.
    """
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return sha256_bytes(text.encode())


def build_class_map() -> dict:
    return {
        "schema_version": 1,
        "provisional": (
            "A4 smoke placeholder. The 16 class NAMES are fixed by DESIGN §4, but "
            "their integer order comes from the CCT-20 annotations and is frozen by "
            "B1, which cannot run before Gate A. Do not train against this order."
        ),
        "classes": list(SMOKE_CLASSES),
        "animal_classes": list(ANIMAL_CLASSES),
        "non_selectable_classes": list(NON_ANIMAL_CLASSES),
    }


def build_policy(model_hash: str, class_map_hash: str, targets: list[str]) -> dict:
    for name in targets:
        if name in NO_THRESHOLD_CLASSES:
            raise ValueError(
                f"{name!r} has no calibrated threshold in the DESIGN §4 catalog "
                "(insufficient validation support). A policy naming it must be "
                "rejected, not generated."
            )
        if name not in ANIMAL_CLASSES:
            raise ValueError(f"{name!r} is not a selectable animal class")

    return {
        "schema_version": 1,
        "policy_id": "smoke_" + "_".join(targets) + "_v0",
        "provisional": (
            f"Threshold {PROVISIONAL_THRESHOLD} is an arbitrary midpoint, NOT a "
            "calibrated operating point. DESIGN §6.3 calibrates the real threshold "
            "on cis-val-clean + trans-val, which do not exist yet. No A4 number is "
            "a result."
        ),
        "model_sha256": model_hash,
        "class_map_sha256": class_map_hash,
        "mode": "any",
        "targets": [
            {"class": name, "threshold": PROVISIONAL_THRESHOLD} for name in targets
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opset", type=int, default=P0_OPSET)
    parser.add_argument(
        "--width", type=int, default=INPUT_SHAPE_CORE[3]
    )
    parser.add_argument(
        "--height", type=int, default=INPUT_SHAPE_CORE[2]
    )
    args = parser.parse_args()

    # The head is randomly initialised; without this the "deterministic" in PLAN A4
    # is a wish. A3 already learned this the hard way with nn.Dropout.
    torch.manual_seed(args.seed)

    # The Core input, NOT ImageNet's 224x224. The application defaults to 256x192, and
    # A4's first run failed on exactly this mismatch: the model contract check refused a
    # 224x224 model against a 256x192 preprocessor rather than letting the geometry
    # disagree silently. Defaulting to the same source of truth is the fix. C1a has since
    # frozen that source of truth, so the two can no longer drift apart.
    shape = (1, 3, args.height, args.width)

    model = build_mobilenet_v2(num_classes=len(SMOKE_CLASSES), pretrained=True)
    model_path = args.output_dir / "smoke_mobilenetv2_16.onnx"
    description = export_onnx(
        model, model_path, example_input(shape), opset=args.opset, dynamo=False
    )

    model_hash = description["sha256"]
    class_map_path = args.output_dir / "class_map.json"
    class_map_hash = write_json(class_map_path, build_class_map())

    # Two policies: the Core single-target case, and the multi-target case DESIGN §4
    # requires the same model to accept without a reload.
    policies = {
        "bobcat_v0.json": ["bobcat"],
        "bobcat_coyote_v0.json": ["bobcat", "coyote"],
    }
    written = {}
    for filename, targets in policies.items():
        path = args.output_dir / filename
        write_json(path, build_policy(model_hash, class_map_hash, targets))
        written[filename] = str(path)

    manifest = {
        "provisional": "A4 smoke artifacts. Not M0. No number here is a result.",
        "seed": args.seed,
        "input_shape": {
            "nchw": list(shape),
            "note": "The Core input, frozen at 256x192 by C1a on 2026-07-16 "
            "(results/ablations/data_input_decision.md). These smoke artifacts predate "
            "that decision but happen to agree with it.",
        },
        "model": {
            "path": str(model_path),
            "sha256": model_hash,
            "opset": description["opset_import"],
            "inputs": description["inputs"],
            "outputs": description["outputs"],
            "size_bytes": description["size_bytes"],
        },
        "class_map": {"path": str(class_map_path), "sha256": class_map_hash},
        "policies": written,
    }
    manifest_path = args.output_dir / "smoke_manifest.json"
    write_json(manifest_path, manifest)

    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
