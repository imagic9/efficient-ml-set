#!/usr/bin/env python3
"""Prove the deployed CLI refuses invalid policies — not just the loader in ctest.

`cpp/tests/test_policy.cpp` already exercises `Policy::load` directly. This drives the
*real binary*, because between the loader and the operator sits the CLI's wiring, and
that wiring is where a rejection gets swallowed: a caught-and-logged exception, a
`return 0` on the error path, a check that runs after the first inference. The unit
test cannot see any of that.

Each case below is a policy that must be refused. A trigger that fires on the wrong
species is the product's worst failure, and every one of these is a way to reach it
while looking correctly configured.

Usage (driven by scripts/run_a4_slice.sh):
    python -m wildlife_trigger.validate.policy_rejections --artifacts A --image I ...
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

# name -> the policy body that must be rejected. `model_sha256`/`class_map_sha256` are
# filled in where the case needs a *valid* binding to isolate what is under test.
REJECTION_CASES: dict[str, dict] = {
    "empty_targets": {
        "schema_version": 1,
        "policy_id": "bad",
        "mode": "any",
        "targets": [],
    },
    "unsupported_mode": {
        "schema_version": 1,
        "policy_id": "bad",
        "mode": "all",
        "targets": [{"class": "bobcat", "threshold": 0.5}],
    },
    "unknown_class": {
        "schema_version": 1,
        "policy_id": "bad",
        "mode": "any",
        "targets": [{"class": "unicorn", "threshold": 0.5}],
    },
    "non_selectable_class": {
        "schema_version": 1,
        "policy_id": "bad",
        "mode": "any",
        "targets": [{"class": "empty", "threshold": 0.5}],
    },
    "duplicate_target": {
        "schema_version": 1,
        "policy_id": "bad",
        "mode": "any",
        "targets": [
            {"class": "bobcat", "threshold": 0.4},
            {"class": "bobcat", "threshold": 0.6},
        ],
    },
    "threshold_above_one": {
        "schema_version": 1,
        "policy_id": "bad",
        "mode": "any",
        "targets": [{"class": "bobcat", "threshold": 1.4}],
    },
    "threshold_below_zero": {
        "schema_version": 1,
        "policy_id": "bad",
        "mode": "any",
        "targets": [{"class": "bobcat", "threshold": -0.2}],
    },
    # DESIGN §4: badger/deer/fox carry threshold null for want of validation support.
    "null_threshold_class": {
        "schema_version": 1,
        "policy_id": "bad",
        "mode": "any",
        "targets": [{"class": "badger", "threshold": None}],
    },
    "unsupported_schema_version": {
        "schema_version": 99,
        "policy_id": "bad",
        "mode": "any",
        "targets": [{"class": "bobcat", "threshold": 0.5}],
    },
    # The binding that stops a policy calibrated for one model being applied to
    # another, where the same class index may denote a different animal.
    "wrong_model_hash": {
        "schema_version": 1,
        "policy_id": "bad",
        "mode": "any",
        "model_sha256": "0" * 64,
        "targets": [{"class": "bobcat", "threshold": 0.5}],
    },
    "wrong_class_map_hash": {
        "schema_version": 1,
        "policy_id": "bad",
        "mode": "any",
        "class_map_sha256": "0" * 64,
        "targets": [{"class": "bobcat", "threshold": 0.5}],
    },
    "malformed_json": {},  # written as raw text below
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--image-rel", required=True)
    parser.add_argument("--artifacts-rel", required=True)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--image-tag", required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    bad_dir = args.artifacts / "invalid_policies"
    bad_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for name, body in REJECTION_CASES.items():
        path = bad_dir / f"{name}.json"
        if name == "malformed_json":
            path.write_text('{"schema_version": 1, "mode": "any", targets: [}\n')
        else:
            path.write_text(json.dumps(body, indent=2) + "\n")

        completed = subprocess.run(
            [
                "docker", "run", "--rm",
                "-v", f"{args.project_root}:/work",
                "-w", "/work",
                args.image_tag,
                args.binary,
                "infer",
                "--model", f"{args.artifacts_rel}/smoke_mobilenetv2_16.onnx",
                "--class-map", f"{args.artifacts_rel}/class_map.json",
                "--policy", f"{args.artifacts_rel}/invalid_policies/{name}.json",
                "--image", args.image_rel,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        # A non-zero exit AND an explanation. An exit code alone could come from a
        # segfault, which is not a rejection — it is a different bug.
        rejected = completed.returncode != 0 and "error:" in completed.stderr
        results[name] = {
            "rejected": rejected,
            "exit_code": completed.returncode,
            "error": completed.stderr.strip().splitlines()[0]
            if completed.stderr.strip()
            else "",
        }
        mark = "PASS" if rejected else "FAIL"
        print(f"    {mark}  {name}: {results[name]['error'][:80]}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(results, indent=2) + "\n")

    return 0 if all(r["rejected"] for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
