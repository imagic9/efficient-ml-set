#!/usr/bin/env python3
"""E6 pre-rental ISA parity — native gx10 vs `qemu-aarch64 -cpu cortex-a76`.

DESIGN §12.2 / PLAN E6: before the rental clock starts, run the C++ dataset runner
under QEMU with the Pi's CPU model and compare it to the native gx10 run of the SAME
binary on the SAME frames. QEMU withholds the build host's i8mm/sve2, so ORT dispatches
the kernels the Pi would, and any divergence surfaces here — in minutes — instead of on
Day 4 with the rental clock running.

The registered expectation (PLAN E6): **the FP32 arm moves, the INT8 arms do not.** An
FP32 convolution takes a different kernel path without i8mm/sve2, so M0's scores shift
slightly; the INT8 graphs run the same asimddp dot-product kernels QEMU's cortex-a76
exposes, so M2/M4 stay near-bitwise. This turns that expectation into a check:

  - **decisions** must agree except where the native score sits within the per-model
    score wobble of the threshold (a flip there is the wobble, not a bug);
  - **INT8 models** (M2, M4) must stay within 1e-3 — the near-bitwise ISA claim;
  - **the FP32 model** (M0) has its score delta reported, expected nonzero, and only
    sanity-bounded (a gross dispatch bug would blow past 0.05 AND flip decisions).

Only CORRECTNESS is judged here. Emulated latency is never a result (DESIGN §12.4): the
runner's timings under QEMU model no caches or memory bandwidth and must not reach a
table.

Usage:
    python -m wildlife_trigger.validate.qemu_parity \\
        --pairs M0:results/e6/native_M0.jsonl:results/e6/qemu_M0.jsonl:0.5381 \\
                M2:...:...:0.6504  M4:...:...:0.3730 \\
        --int8 M2 M4 --output results/e6/qemu_parity.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

INT8_MAX_ABS = 1e-3       # near-bitwise: the asimddp dot-product kernels are shared
FP32_SANITY_ABS = 0.05    # a gross FP dispatch bug, not the expected small ISA shift


def read_predictions(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    out: dict[str, dict] = {}
    for r in rows:
        if r.get("kind") in ("run_dataset_header", "run_dataset_footer"):
            continue
        if "image_id" in r and "target_scores" in r:
            out[r["image_id"]] = r
    return out


def compare_model(model_id: str, native_path: Path, qemu_path: Path,
                  threshold: float, is_int8: bool, target: str = "bobcat") -> dict:
    native = read_predictions(native_path)
    qemu = read_predictions(qemu_path)
    failures: list[str] = []

    common = [i for i in native if i in qemu]
    if len(common) != len(native) or len(native) != len(qemu):
        failures.append(
            f"frame sets differ (native {len(native)}, qemu {len(qemu)}, common {len(common)})"
        )

    max_abs = 0.0
    sum_abs = 0.0
    decisions_differ = 0
    hard_disagree = []
    for image_id in common:
        n, q = native[image_id], qemu[image_id]
        ns = n["target_scores"][target]
        qs = q["target_scores"][target]
        gap = abs(ns - qs)
        max_abs = max(max_abs, gap)
        sum_abs += gap
        if n["shutter_trigger"] != q["shutter_trigger"]:
            decisions_differ += 1
            # A flip is explained if the native score sits within the observed wobble
            # of the threshold; otherwise it is a real divergence.
            if abs(ns - threshold) > gap:
                hard_disagree.append(
                    {"image_id": image_id, "native_score": ns, "qemu_score": qs}
                )
    mean_abs = (sum_abs / len(common)) if common else 0.0

    if hard_disagree:
        failures.append(
            f"{len(hard_disagree)} decision flips not explained by the score wobble"
        )
    if is_int8 and max_abs > INT8_MAX_ABS:
        failures.append(
            f"INT8 max score delta {max_abs:.2e} exceeds the near-bitwise gate "
            f"{INT8_MAX_ABS:.0e}; the ISA dispatch is not identical"
        )
    if not is_int8 and max_abs > FP32_SANITY_ABS:
        failures.append(
            f"FP32 max score delta {max_abs:.2e} exceeds the {FP32_SANITY_ABS} sanity "
            "bound; expected a small shift, not this"
        )

    return {
        "model": model_id,
        "kind": "int8" if is_int8 else "fp32",
        "frames": len(common),
        "max_score_delta": max_abs,
        "mean_score_delta": mean_abs,
        "decisions_differing": decisions_differ,
        "hard_disagreements": hard_disagree[:5],
        "expectation": "near-bitwise (INT8 shares asimddp kernels)" if is_int8
        else "small shift expected (FP32 kernels differ without i8mm/sve2)",
        "passed": not failures,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs", nargs="+", required=True,
        help="model_id:native.jsonl:qemu.jsonl:threshold, one per model",
    )
    parser.add_argument("--int8", nargs="*", default=[], help="model ids that are INT8")
    parser.add_argument("--target", default="bobcat")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    int8 = set(args.int8)
    results = []
    for spec in args.pairs:
        model_id, native_path, qemu_path, threshold = spec.split(":")
        results.append(
            compare_model(
                model_id, Path(native_path), Path(qemu_path),
                float(threshold), model_id in int8, args.target,
            )
        )

    passed = all(r["passed"] for r in results)
    report = {
        "gate": "E6 pre-rental ISA parity (native gx10 vs qemu -cpu cortex-a76)",
        "design": "12.2 / PLAN E6",
        "note": "correctness only; emulated latency is never a result (DESIGN §12.4)",
        "tolerances": {"int8_max_abs": INT8_MAX_ABS, "fp32_sanity_abs": FP32_SANITY_ABS},
        "models": results,
        "verdict": {"passed": passed, "failed": [r["model"] for r in results if not r["passed"]]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    for r in results:
        print(
            f"{r['model']} ({r['kind']}): {'PASS' if r['passed'] else 'FAIL'} "
            f"— {r['frames']} frames, max Δ {r['max_score_delta']:.2e}, "
            f"mean Δ {r['mean_score_delta']:.2e}, {r['decisions_differing']} decisions differ"
        )
        for f in r["failures"]:
            print(f"    FAIL: {f}")
    print(f"E6 QEMU parity {'PASSED' if passed else 'FAILED'}; wrote {args.output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
