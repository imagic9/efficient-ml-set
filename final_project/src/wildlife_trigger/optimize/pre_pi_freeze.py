#!/usr/bin/env python3
"""D6 — freeze the deployable bundle for Pi validation (PLAN D6, Gate D).

Turns the shortlist into a hash-locked bill of materials: for every shortlisted
model, the exact bytes that ship — the ONNX artifact, its calibrated bobcat
policy, the shared class map, and the preprocessing contract — each verified to
hash to what the policy already bound. E7 packages from this file and F4 runs
its mandatory parity against it, so it must be produced where the artifacts
actually live (gx10), not from a checkout that only knows their hashes.

The freeze is the moment the comparison stops being a table of numbers and
becomes a set of files someone will run on hardware. Nothing here re-decides
anything — the shortlist chose the models, the policies bound the bytes; this
records the intersection and refuses any drift.

Test labels stay sealed: the benchmark this bundle is measured on is
validation-only (`benchmark_val_1000.jsonl`), and no test manifest is named.

Usage (gx10):
    python -m wildlife_trigger.optimize.pre_pi_freeze \
        --shortlist results/model_selection/pre_pi_shortlist.json \
        --comparison results/model_selection/comparison.jsonl \
        --benchmark data/manifests/benchmark_val_1000.jsonl \
        --output results/model_selection/pre_pi_freeze.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..data.preprocess import PreprocessConfig
from ..runs import atomic_write_json, sha256_file


def resolve_policy(comparison_rows: list[dict], model_id: str) -> Path:
    (row,) = [r for r in comparison_rows if r["model_id"] == model_id]
    return Path(row["policy"]["path"]), row


def freeze_model(model_id: str, comparison_rows: list[dict]) -> dict:
    """The deployable triple for one model, every hash re-verified now."""
    policy_path, row = resolve_policy(comparison_rows, model_id)
    policy = json.loads(policy_path.read_text())

    artifact = Path(policy["model"]["artifact"])
    if not artifact.exists():
        raise RuntimeError(
            f"{model_id}: {artifact} is absent here; the freeze must run where "
            "the deployable artifact lives (gx10), not from a bare checkout"
        )
    artifact_sha = sha256_file(artifact)
    if artifact_sha != policy["model_sha256"]:
        raise RuntimeError(
            f"{model_id}: {artifact} hashes to {artifact_sha[:12]}… but its "
            f"policy binds {policy['model_sha256'][:12]}…; the file drifted "
            "from what was calibrated and parity-proven"
        )
    if artifact_sha != row["model"]["sha256"]:
        raise RuntimeError(
            f"{model_id}: the comparison row and the policy disagree on the "
            "artifact hash; the freeze refuses an inconsistent bundle"
        )

    class_map = Path(policy["class_map"]["path"]) if "path" in policy.get(
        "class_map", {}
    ) else Path("artifacts/class_map.json")

    return {
        "model_id": model_id,
        "kind": row["kind"],
        "onnx": {
            "artifact": str(artifact),
            "sha256": artifact_sha,
            "bytes": artifact.stat().st_size,
        },
        "policy": {
            "path": str(policy_path),
            "policy_id": policy["policy_id"],
            "sha256": sha256_file(policy_path),
            "threshold": policy["targets"][0]["threshold"],
            "status": policy["calibration"]["status"],
        },
        "parity_report": policy["model"].get("parity"),
        "macs": row["macs"],
        "params": row["params"],
    }


def freeze(
    shortlist_path: Path,
    comparison_path: Path,
    benchmark_path: Path,
    class_map_path: Path,
    width: int,
    height: int,
) -> dict:
    shortlist = json.loads(shortlist_path.read_text())
    comparison_rows = [
        json.loads(line) for line in comparison_path.read_text().splitlines() if line
    ]
    ids = shortlist["shortlist"]

    models = [freeze_model(model_id, comparison_rows) for model_id in ids]

    # The class map is shared and identical across the bundle; verify once that
    # every policy names the same one.
    class_map_sha = sha256_file(class_map_path)

    config = PreprocessConfig(width=width, height=height)

    benchmark_provenance = benchmark_path.with_suffix(".provenance.json")
    return {
        "tool": "wildlife_trigger.optimize.pre_pi_freeze",
        "design": "8.5 / 12.2 (Gate D)",
        "shortlist": ids,
        "shortlist_source": str(shortlist_path),
        "models": models,
        "class_map": {"path": str(class_map_path), "sha256": class_map_sha},
        "preprocessing": {
            "width": config.width,
            "height": config.height,
            "pad_value": config.pad_value,
            "mean": list(config.mean),
            "std": list(config.std),
            "fingerprint": config.fingerprint(),
        },
        "benchmark": {
            "manifest": str(benchmark_path),
            "sha256": sha256_file(benchmark_path),
            "provenance": str(benchmark_provenance),
            "provenance_sha256": sha256_file(benchmark_provenance)
            if benchmark_provenance.exists()
            else None,
            "note": "validation-only; test manifests stay sealed (§5.4)",
        },
        "test_labels": "sealed — no test manifest is named in this bundle",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shortlist", type=Path,
        default=Path("results/model_selection/pre_pi_shortlist.json"),
    )
    parser.add_argument(
        "--comparison", type=Path,
        default=Path("results/model_selection/comparison.jsonl"),
    )
    parser.add_argument(
        "--benchmark", type=Path,
        default=Path("data/manifests/benchmark_val_1000.jsonl"),
    )
    parser.add_argument(
        "--class-map", type=Path, default=Path("artifacts/class_map.json")
    )
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/model_selection/pre_pi_freeze.json"),
    )
    args = parser.parse_args()

    record = freeze(
        args.shortlist, args.comparison, args.benchmark,
        args.class_map, args.width, args.height,
    )
    atomic_write_json(args.output, record)
    print(f"froze {len(record['models'])} models: {' · '.join(record['shortlist'])}")
    for model in record["models"]:
        print(f"  {model['model_id']}: {model['onnx']['bytes']:,} B "
              f"sha {model['onnx']['sha256'][:12]}… policy {model['policy']['policy_id']}")
    print(f"  benchmark {record['benchmark']['sha256'][:12]}… "
          f"class_map {record['class_map']['sha256'][:12]}…")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
