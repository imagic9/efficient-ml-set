#!/usr/bin/env python3
"""Maintain the single machine-readable model comparison table (DESIGN §16 phase D).

One row per candidate (M0-M4), one file for the whole ladder:
`results/model_selection/comparison.jsonl`. Every phase that produces a candidate
appends or replaces its row; D6's shortlist and the report's headline tables read
this file rather than re-deriving numbers from scattered artifacts.

A row is written only from evidence, never typed in:

- validation metrics come from the run's own `history.json` at its selected epoch;
- the operating point comes from the calibrated policy, which must name this run;
- the deployable artifact must hash to the policy's `model_sha256` — the same
  binding `rebind_policy` enforced, re-checked here so a table row cannot outlive
  a re-export;
- the parity report must name that same artifact and must have passed. A candidate
  that failed its gates has no business in the comparison table: DESIGN D6 rejects
  it before selection, so admitting its row would stage exactly the mistake D6
  exists to prevent.

Pi columns are absent by design until F-phase measures them; a null field is a
promise nobody has kept yet, and this table only holds kept ones.

Usage:
    python -m wildlife_trigger.comparison \
        --run results/training/c2/c2_m0_fp32_seed42_20260716T061203Z \
        --policy artifacts/policies/bobcat_v1.json \
        --model-id M0 --kind fp32_baseline
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch

from .runs import BEST_CHECKPOINT, resolve_run_id, sha256_file

TABLE_PATH = Path("results/model_selection/comparison.jsonl")


def load_row_inputs(run_dir: Path, policy_path: Path) -> tuple[dict, dict]:
    history = json.loads((run_dir / "history.json").read_text())
    policy = json.loads(policy_path.read_text())

    run_id = resolve_run_id(run_dir, history["run_name"])
    calibrated_for = policy["calibration"]["run_id"]
    if calibrated_for != run_id:
        raise RuntimeError(
            f"policy {policy_path} was calibrated for {calibrated_for} but this run "
            f"is {run_id}; a row that mixes one model's metrics with another's "
            "operating point describes a device that does not exist"
        )
    return history, policy


def verify_artifact(policy: dict) -> tuple[Path, str]:
    """The deployable artifact, proven to be the one the policy is bound to.

    Policy paths are relative to `final_project/`, where every tool in this
    package is documented to run from.
    """
    artifact = Path(policy["model"]["artifact"])
    if not artifact.exists():
        raise RuntimeError(
            f"{artifact} does not exist here; the comparison row must be produced "
            "where the deployable artifact lives (gx10), not from a checkout that "
            "only knows its hash"
        )
    measured = sha256_file(artifact)
    if measured != policy["model_sha256"]:
        raise RuntimeError(
            f"{artifact} hashes to {measured[:12]}… but the policy binds "
            f"{policy['model_sha256'][:12]}…; whatever this file is now, it is not "
            "the model the policy was calibrated against"
        )
    return artifact, measured


def verify_parity(parity_path: Path, onnx_sha256: str) -> dict:
    report = json.loads(parity_path.read_text())
    if report["onnx"]["sha256"] != onnx_sha256:
        raise RuntimeError(
            f"parity report {parity_path} proves a different artifact "
            f"({report['onnx']['sha256'][:12]}… vs {onnx_sha256[:12]}…)"
        )
    if not report["verdict"]["passed"]:
        raise RuntimeError(
            f"parity report {parity_path} did not pass; DESIGN D6 rejects gate "
            "failures before selection, so they do not enter the comparison table"
        )
    return report


def count_parameters(run_dir: Path, history: dict) -> int:
    """Learnable parameters of the selected checkpoint, hash-verified first.

    The state dict is loaded into a freshly built architecture and the count taken
    from `model.parameters()`: summing state-dict tensors directly would silently
    include BatchNorm running statistics, which are buffers, not parameters.
    Pruned candidates (M3/M4) change the architecture and will need their own
    loading path here — that is D-phase's task, named in PLAN D4.
    """
    from .models.mobilenet import build_mobilenet_v2

    hashes = json.loads((run_dir / "hashes.json").read_text())
    checkpoint_path = run_dir / BEST_CHECKPOINT
    measured = sha256_file(checkpoint_path)
    if measured != hashes["checkpoint:best"]["sha256"]:
        raise RuntimeError(
            f"{checkpoint_path} does not hash to the run's record; its parameter "
            "count would describe an unknown file"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "epoch" in checkpoint and checkpoint["epoch"] != history["best_epoch"]:
        raise RuntimeError(
            f"best.pt holds epoch {checkpoint['epoch']} but the history selected "
            f"{history['best_epoch']}; this checkpoint is not this run's model"
        )
    model = build_mobilenet_v2(num_classes=len(history["class_names"]), pretrained=False)
    model.load_state_dict(checkpoint["model"])
    return sum(parameter.numel() for parameter in model.parameters())


def build_row(
    model_id: str,
    kind: str,
    run_dir: Path,
    history: dict,
    policy: dict,
    policy_path: Path,
    artifact: Path,
    onnx_sha256: str,
    parity_path: Path,
    params: int,
    macs: int,
) -> dict:
    config = history["config"]
    # By the epoch field, not by list index: an index would silently read the wrong
    # entry the day a resumed run writes a partial history.
    (best,) = [e for e in history["history"] if e["epoch"] == history["best_epoch"]]
    calibration = policy["calibration"]

    def domain_f2(name: str) -> float:
        return best[name]["frame_f2"]

    validation = {
        "cis_f2": domain_f2("cis_val_clean"),
        "trans_f2": domain_f2("trans_val"),
        "selection_score": best["selection_score"]["primary"],
    }
    # AP per epoch arrived with issue #19's verdict; the baseline run predates it.
    for domain, key in (("cis_val_clean", "cis_ap"), ("trans_val", "trans_ap")):
        if "bobcat_ap" in best[domain]:
            validation[key] = best[domain]["bobcat_ap"]

    return {
        "model_id": model_id,
        "kind": kind,
        "run_id": resolve_run_id(run_dir, history["run_name"]),
        "seed": config["seed"],
        "best_epoch": history["best_epoch"],
        "input": f"{config['width']}x{config['height']}",
        "params": params,
        "macs": macs,
        "model": {
            "artifact": policy["model"]["artifact"],
            "sha256": onnx_sha256,
            "bytes": artifact.stat().st_size,
        },
        "validation_at_0p5": validation,
        "operating_point": {
            "threshold": policy["targets"][0]["threshold"],
            "status": calibration["status"],
            "primary_rule_met": calibration["primary_rule_met"],
            "per_domain": calibration["per_domain"],
        },
        "policy": {
            "policy_id": policy["policy_id"],
            "path": str(policy_path),
            "sha256": sha256_file(policy_path),
        },
        "parity": {"report": str(parity_path), "passed": True},
    }


def update_table(table_path: Path, row: dict) -> list[dict]:
    """Replace this model's row, keep everyone else's, keep the table ordered."""
    rows: list[dict] = []
    if table_path.exists():
        rows = [json.loads(line) for line in table_path.read_text().splitlines() if line]
    rows = [r for r in rows if r["model_id"] != row["model_id"]]
    rows.append(row)
    rows.sort(key=lambda r: r["model_id"])

    table_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    fd, temp = tempfile.mkstemp(dir=table_path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(payload)
        os.replace(temp, table_path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--model-id", required=True, help="M0…M4")
    parser.add_argument("--kind", required=True,
                        help="fp32_baseline | int8_ptq | int8_qat | pruned_fp32 | pruned_qat")
    parser.add_argument("--parity", type=Path,
                        help="default: the report the policy itself names")
    parser.add_argument("--table", type=Path, default=TABLE_PATH)
    args = parser.parse_args()

    history, policy = load_row_inputs(args.run, args.policy)
    artifact, onnx_sha256 = verify_artifact(policy)
    parity_path = args.parity or Path(policy["model"]["parity"])
    verify_parity(parity_path, onnx_sha256)
    params = count_parameters(args.run, history)

    from .validate.input_cost import macs_at

    config = history["config"]
    macs = macs_at(config["width"], config["height"], len(history["class_names"]))

    row = build_row(
        args.model_id, args.kind, args.run, history, policy, args.policy,
        artifact, onnx_sha256, parity_path, params, macs,
    )
    rows = update_table(args.table, row)
    print(f"{args.table}: {len(rows)} row(s); wrote {args.model_id} "
          f"(params {params:,}, macs {macs:,}, "
          f"cis F2 {row['validation_at_0p5']['cis_f2']:.4f}, "
          f"trans F2 {row['validation_at_0p5']['trans_f2']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
