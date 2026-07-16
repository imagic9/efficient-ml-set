#!/usr/bin/env python3
"""P1 — preprocessing parity: Python canonical vs C++ reference vs C++ fused.

The comparison DESIGN §10 demands, run where both tensors exist. The golden JSON
deliberately froze hashes and stats, not tensors, because bit-exactness across
OpenCV builds is not a promise anyone made (`golden_tensors.py`'s `exactness`
block); so this tool recomputes the Python tensor live and reads the C++ tensors
from `dump-tensor`'s `.bin` files.

Three-way, not two-way, so a disagreement localises:

- all three agree            -> the contract holds;
- fused alone differs        -> a fusion bug in the hot path;
- both C++ differ from Python-> the OpenCV 4.6-vs-4.13 INTER_LINEAR gap, the
                                named P1 risk, now measured instead of feared.

## The pre-registered gates (DESIGN §10, C4 amendment, 2026-07-16)

Registered before the first real comparison ran, so the numbers cannot drift
toward whatever the measurement happened to produce:

- **geometry, exactly**: resized/pad integers are pure arithmetic on two ints
  and must be identical everywhere, forever;
- **pad region within 1e-6** of the normalised grey constant: pads never pass
  through the resize, so no OpenCV version excuse applies (1e-6 rather than
  bit-exact because `convertTo(alpha=1/255)` multiplies by a reciprocal where
  the others divide — a 1-ulp difference with no meaning);
- **reference vs fused (same OpenCV, same host): max abs <= 1e-6** — anything
  more is a fusion bug, never a version gap;
- **Python vs either C++ path: max abs <= 0.035, mean abs <= 2e-3.** One uint8
  LSB after /255 and the largest ImageNet std (0.225) is (1/255)/0.225 = 0.0174;
  INTER_LINEAR across 4.6/4.13 may plausibly land 1 LSB apart on interior
  pixels, so the gate admits 2 LSB peak and a mean an order under 1 LSB. If the
  measured gap exceeds this, the registered answer is to build matching OpenCV
  in the container and bundle its .so to the Pi (pins.env) — not to widen the
  tolerance.

Usage (normally via scripts/run_c4_parity.sh):
    python -m wildlife_trigger.validate.p1_preprocess \
        --cpp-dir results/parity/<run_id>/p1 \
        --supplement tests/fixtures/p1_supplement/manifest.json \
        --golden tests/fixtures/golden_raw.json \
        --images-dir data/raw/extracted/eccv_18_all_images_sm \
        --output results/parity/<run_id>/p1_preprocess.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from ..data.preprocess import PreprocessConfig, preprocess_file
from ..runs import atomic_write_json

# The pre-registered gates. Changing any of these is a DESIGN §10 amendment, not
# an edit.
GEOMETRY_EXACT = ("resized_width", "resized_height", "pad_left", "pad_top")
PAD_REGION_MAX_ABS = 1e-6
SAME_HOST_MAX_ABS = 1e-6  # C++ reference vs C++ fused
CROSS_VERSION_MAX_ABS = 0.035  # Python (OpenCV 4.13) vs C++ (OpenCV 4.6), 2 uint8 LSB
CROSS_VERSION_MEAN_ABS = 2e-3


def load_cpp(cpp_dir: Path, name: str, mode: str, expected_elements: int) -> tuple[np.ndarray, dict]:
    """One dump-tensor output pair, size-checked before it is trusted."""
    bin_path = cpp_dir / f"{name}.{mode}.bin"
    json_path = cpp_dir / f"{name}.{mode}.json"
    meta = json.loads(json_path.read_text())
    tensor = np.fromfile(bin_path, dtype=np.float32)
    if tensor.size != expected_elements:
        raise RuntimeError(
            f"{bin_path} holds {tensor.size} floats, expected {expected_elements}; "
            "the C++ dump was made at a different geometry than this comparison"
        )
    measured = hashlib.sha256(tensor.tobytes()).hexdigest()
    if measured != meta["tensor_sha256"]:
        raise RuntimeError(
            f"{bin_path} hashes to {measured[:16]}... but its own JSON records "
            f"{meta['tensor_sha256'][:16]}...; the pair is not from one run"
        )
    return tensor, meta


def error_stats(a: np.ndarray, b: np.ndarray) -> dict:
    difference = np.abs(a - b)
    return {"max_abs": float(difference.max()), "mean_abs": float(difference.mean())}


def pad_mask(letterbox: dict, height: int, width: int) -> np.ndarray:
    """True where the tensor is letterbox padding rather than image content."""
    mask = np.ones((height, width), dtype=bool)
    top = letterbox["pad_top"]
    left = letterbox["pad_left"]
    mask[top : top + letterbox["resized_height"], left : left + letterbox["resized_width"]] = False
    return mask


def compare_fixture(
    name: str,
    image_path: Path,
    source: str,
    cpp_dir: Path,
    config: PreprocessConfig,
) -> dict:
    python_tensor, info = preprocess_file(image_path, config)
    python_tensor = python_tensor.reshape(3, config.height, config.width)
    elements = python_tensor.size

    fused, fused_meta = load_cpp(cpp_dir, name, "fused", elements)
    reference, reference_meta = load_cpp(cpp_dir, name, "reference", elements)
    fused = fused.reshape(3, config.height, config.width)
    reference = reference.reshape(3, config.height, config.width)

    failures = []

    # Geometry: pure integer arithmetic; identical or someone's contract moved.
    python_letterbox = asdict(info)
    for meta, mode in ((fused_meta, "fused"), (reference_meta, "reference")):
        cpp_letterbox = meta["letterbox"]
        cpp_values = {
            "resized_width": cpp_letterbox["resized"][0],
            "resized_height": cpp_letterbox["resized"][1],
            "pad_left": cpp_letterbox["pad_left"],
            "pad_top": cpp_letterbox["pad_top"],
        }
        for key in GEOMETRY_EXACT:
            if cpp_values[key] != python_letterbox[key]:
                failures.append(
                    f"{mode} geometry {key}: cpp={cpp_values[key]} python={python_letterbox[key]}"
                )

    # The pad region: never touched by the resize, so no version excuse applies.
    mask = pad_mask(
        {k: python_letterbox[k] for k in ("pad_top", "pad_left", "resized_height", "resized_width")},
        config.height,
        config.width,
    )
    pad_errors = {}
    if mask.any():
        expected = (
            np.float32(config.pad_value) / np.float32(255.0)
            - np.asarray(config.mean, dtype=np.float32)
        ) / np.asarray(config.std, dtype=np.float32)
        for tensor, who in ((python_tensor, "python"), (fused, "fused"), (reference, "reference")):
            worst = max(
                float(np.abs(tensor[c][mask] - expected[c]).max()) for c in range(3)
            )
            pad_errors[who] = worst
            if worst > PAD_REGION_MAX_ABS:
                failures.append(f"{who} pad region off the grey constant by {worst:.2e}")

    errors = {
        "python_vs_fused": error_stats(python_tensor, fused),
        "python_vs_reference": error_stats(python_tensor, reference),
        "reference_vs_fused": error_stats(reference, fused),
    }
    if errors["reference_vs_fused"]["max_abs"] > SAME_HOST_MAX_ABS:
        failures.append(
            f"reference vs fused max abs {errors['reference_vs_fused']['max_abs']:.2e} "
            "exceeds the same-host gate: a fusion bug, not a version gap"
        )
    for pair in ("python_vs_fused", "python_vs_reference"):
        if errors[pair]["max_abs"] > CROSS_VERSION_MAX_ABS:
            failures.append(f"{pair} max abs {errors[pair]['max_abs']:.2e} exceeds the gate")
        if errors[pair]["mean_abs"] > CROSS_VERSION_MEAN_ABS:
            failures.append(f"{pair} mean abs {errors[pair]['mean_abs']:.2e} exceeds the gate")

    return {
        "fixture": name,
        "source": source,
        "image": str(image_path),
        "letterbox": python_letterbox,
        "pixel_utilisation": info.pixel_utilisation(),
        "pad_region_max_abs": pad_errors,
        "errors": errors,
        "passed": not failures,
        "failures": failures,
    }


def gather_fixtures(args) -> list[tuple[str, Path, str]]:
    """(name, image path, source) triples from the two fixture pools."""
    fixtures: list[tuple[str, Path, str]] = []
    if args.supplement:
        manifest = json.loads(args.supplement.read_text())
        base = args.supplement.parent
        for name, entry in sorted(manifest["fixtures"].items()):
            path = base / Path(entry["path"]).name
            measured = hashlib.sha256(path.read_bytes()).hexdigest()
            if measured != entry["sha256"]:
                raise RuntimeError(
                    f"{path} does not match its manifest hash; the committed "
                    "fixture drifted from what the suite registered"
                )
            fixtures.append((name, path, "supplement"))
    if args.golden:
        golden = json.loads(args.golden.read_text())
        if not args.images_dir:
            raise RuntimeError("--golden requires --images-dir for the raw frames")
        for entry in golden["fixtures"]:
            path = args.images_dir / entry["file_name"]
            measured = hashlib.sha256(path.read_bytes()).hexdigest()
            if measured != entry["sha256"]:
                raise RuntimeError(
                    f"{path} does not hash to golden_raw.json's record; this is "
                    "not the frozen fixture"
                )
            fixtures.append((entry["image_id"], path, "golden"))
    if not fixtures:
        raise RuntimeError("no fixtures: pass --supplement and/or --golden")
    return fixtures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpp-dir", required=True, type=Path,
                        help="directory of dump-tensor outputs (<name>.<mode>.{bin,json})")
    parser.add_argument("--supplement", type=Path,
                        help="tests/fixtures/p1_supplement/manifest.json")
    parser.add_argument("--golden", type=Path, help="tests/fixtures/golden_raw.json")
    parser.add_argument("--images-dir", type=Path,
                        help="raw image root for the golden fixtures (gx10)")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config = PreprocessConfig()
    results = [
        compare_fixture(name, path, source, args.cpp_dir, config)
        for name, path, source in gather_fixtures(args)
    ]

    cpp_versions = {
        json.loads(p.read_text())["opencv_version"]
        for p in args.cpp_dir.glob("*.json")
    }
    worst = {
        pair: {
            "max_abs": max(r["errors"][pair]["max_abs"] for r in results),
            "mean_abs": max(r["errors"][pair]["mean_abs"] for r in results),
        }
        for pair in ("python_vs_fused", "python_vs_reference", "reference_vs_fused")
    }
    report = {
        "gate": "P1 preprocessing parity (DESIGN 10, tolerances registered 2026-07-16)",
        "tolerances": {
            "geometry": "exact",
            "pad_region_max_abs": PAD_REGION_MAX_ABS,
            "reference_vs_fused_max_abs": SAME_HOST_MAX_ABS,
            "python_vs_cpp_max_abs": CROSS_VERSION_MAX_ABS,
            "python_vs_cpp_mean_abs": CROSS_VERSION_MEAN_ABS,
        },
        "opencv": {"python": cv2.__version__, "cpp": sorted(cpp_versions)},
        "fixtures": len(results),
        "by_source": {
            source: sum(1 for r in results if r["source"] == source)
            for source in {r["source"] for r in results}
        },
        "worst_case": worst,
        "verdict": {
            "passed": all(r["passed"] for r in results),
            "failed_fixtures": [r["fixture"] for r in results if not r["passed"]],
        },
        "results": results,
    }
    atomic_write_json(args.output, report)

    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        e = r["errors"]
        print(
            f"  {mark}  {r['fixture']:<28} py-fused max {e['python_vs_fused']['max_abs']:.2e} "
            f"py-ref max {e['python_vs_reference']['max_abs']:.2e} "
            f"ref-fused max {e['reference_vs_fused']['max_abs']:.2e}"
        )
    print(f"P1 {'PASSED' if report['verdict']['passed'] else 'FAILED'} "
          f"({report['fixtures']} fixtures) -> {args.output}")
    return 0 if report["verdict"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
