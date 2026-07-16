#!/usr/bin/env python3
"""Re-bind a calibrated policy from its checkpoint to the parity-proven ONNX.

C3 bound `bobcat_v1.json` to the PyTorch checkpoint deliberately — the only
model artifact that existed, and a binding that fails loudly against any ONNX
until the proof exists. This tool is the *only* sanctioned way that binding
moves, and it refuses to move it without the proof:

1. the P2 report must exist and its verdict must be `passed` — an unproven
   graph keeps the old binding and the loud failure;
2. the report's checkpoint hash must equal the policy's current
   `model_sha256` — the proof must be about the weights this policy was
   calibrated on, not any weights;
3. the ONNX file is re-hashed now and must equal the report's recorded hash —
   the file being bound is the file that passed, not whatever sits at that
   path today.

The calibration block is left untouched: it describes the calibration event
(datasets, verdict, thresholds), which remains true. The old binding is kept
inside the new `model` block, so the artifact's history reads forward from the
file itself. The rewritten policy is re-validated against the class map with
both actual hashes before a byte is written.

Usage:
    python -m wildlife_trigger.rebind_policy \
        --policy artifacts/policies/bobcat_v1.json \
        --class-map artifacts/class_map.json \
        --onnx results/training/c2/<run_id>/m0_fp32_seed42.onnx \
        --p2-report results/parity/<run_id>/p2_fp32.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .policy import validate_policy, write_canonical_json
from .runs import sha256_file


def rebind(policy_path: Path, class_map_path: Path, onnx_path: Path, report_path: Path) -> dict:
    if not report_path.exists():
        raise RuntimeError(
            f"{report_path} does not exist. No P2 report, no re-bind: the policy "
            "keeps failing loudly against the ONNX until the proof exists."
        )
    report = json.loads(report_path.read_text())
    if not report.get("verdict", {}).get("passed"):
        raise RuntimeError(
            f"{report_path} records verdict.passed="
            f"{report.get('verdict', {}).get('passed')!r}. A failed or absent "
            "verdict re-binds nothing."
        )

    policy = json.loads(policy_path.read_text())
    current = policy.get("model_sha256", "")
    if report.get("checkpoint_sha256") != current:
        raise RuntimeError(
            f"the P2 report proves checkpoint {report.get('checkpoint_sha256', '')[:16]}... "
            f"but this policy is bound to {current[:16]}.... The proof is about "
            "different weights than the calibration."
        )

    onnx_sha256 = sha256_file(onnx_path)
    if report.get("onnx", {}).get("sha256") != onnx_sha256:
        raise RuntimeError(
            f"{onnx_path} hashes to {onnx_sha256[:16]}... but the P2 report passed "
            f"{report.get('onnx', {}).get('sha256', '')[:16]}.... The file being "
            "bound is not the file that was proven."
        )

    previous = policy.get("model", {})
    policy["model_sha256"] = onnx_sha256
    policy["model"] = {
        "kind": "onnx",
        "artifact": str(onnx_path),
        "parity": str(report_path),
        "binding": (
            "model_sha256 names the FP32 ONNX, re-bound after P2 parity "
            "(logits, top-1 and decisions match the calibrated checkpoint). "
            "The C++ loader now accepts this policy against exactly that file."
        ),
        "previous": {
            "kind": previous.get("kind", "pytorch_checkpoint"),
            "artifact": previous.get("artifact"),
            "model_sha256": current,
        },
    }

    class_map = json.loads(class_map_path.read_text())
    validate_policy(
        policy,
        class_map,
        model_sha256=onnx_sha256,
        class_map_sha256=sha256_file(class_map_path),
    )
    write_canonical_json(policy_path, policy)
    return policy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--class-map", required=True, type=Path)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--p2-report", required=True, type=Path)
    args = parser.parse_args()

    policy = rebind(args.policy, args.class_map, args.onnx, args.p2_report)
    print(f"re-bound {args.policy}")
    print(f"  model_sha256: {policy['model_sha256']}")
    print(f"  previous:     {policy['model']['previous']['model_sha256']}")
    print(f"  parity proof: {policy['model']['parity']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
