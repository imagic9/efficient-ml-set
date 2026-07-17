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

Optimized candidates (M1/M2) enter through `--candidate`: their validation
metrics come from the candidate's own ORT evaluation record
(`optimize.evaluate_onnx`), never from the source run's history — a quantized
model's numbers belong to its arithmetic, not to the checkpoint it started
from. Params/MACs are copied from the base row (`--base-model-id`, default M0)
because quantization changes neither: the architecture is byte-for-byte the
same graph shape, only the storage and kernels differ. Pruned candidates
(M3/M4) change the architecture and will need their own loading path — PLAN D4.

Usage:
    python -m wildlife_trigger.comparison \
        --run results/training/c2/c2_m0_fp32_seed42_20260716T061203Z \
        --policy artifacts/policies/bobcat_v1.json \
        --model-id M0 --kind fp32_baseline

    python -m wildlife_trigger.comparison \
        --candidate results/optimize/m1_ptq/minmax \
        --policy artifacts/policies/bobcat_m1_ptq_minmax_v1.json \
        --parity results/optimize/m1_ptq/minmax/p3_quantized.json \
        --model-id M1 --kind int8_ptq
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
        if "average_precision" in best[domain]:
            validation[key] = best[domain]["average_precision"]

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


def load_candidate_row_inputs(
    candidate_dir: Path, policy_path: Path
) -> tuple[dict, dict, dict]:
    candidate = json.loads((candidate_dir / "candidate.json").read_text())
    evaluation = json.loads((candidate_dir / "evaluation.json").read_text())
    policy = json.loads(policy_path.read_text())

    calibrated_for = policy["calibration"]["run_id"]
    if calibrated_for != candidate["candidate_id"]:
        raise RuntimeError(
            f"policy {policy_path} was calibrated for {calibrated_for} but this "
            f"candidate is {candidate['candidate_id']}; a row that mixes one "
            "model's metrics with another's operating point describes a device "
            "that does not exist"
        )
    if evaluation["model"]["sha256"] != candidate["model"]["sha256"]:
        raise RuntimeError(
            f"{candidate_dir} is inconsistent: candidate.json and evaluation.json "
            "describe different artifacts"
        )
    return candidate, evaluation, policy


def base_row(table_path: Path, base_model_id: str) -> dict:
    """The base candidate's committed row — the source of params/MACs.

    Copied from evidence rather than re-derived: the base row already proved its
    params against the hash-verified checkpoint, and PTQ/QAT change neither the
    parameter count nor the MAC count, only their representation.
    """
    if not table_path.exists():
        raise RuntimeError(
            f"{table_path} does not exist; the base row {base_model_id} must be "
            "written before a derived candidate can copy its params/MACs"
        )
    rows = [json.loads(line) for line in table_path.read_text().splitlines() if line]
    matches = [r for r in rows if r["model_id"] == base_model_id]
    if not matches:
        raise RuntimeError(
            f"{table_path} has no {base_model_id} row; a derived candidate cannot "
            "invent its params/MACs"
        )
    (row,) = matches
    return row


def build_candidate_row(
    model_id: str,
    kind: str,
    candidate: dict,
    evaluation: dict,
    policy: dict,
    policy_path: Path,
    artifact: Path,
    onnx_sha256: str,
    parity_path: Path,
    base: dict,
) -> dict:
    calibration = policy["calibration"]
    domains = evaluation["domains"]

    validation = {
        "cis_f2": domains["cis_val_clean"]["target"]["frame_f2"],
        "trans_f2": domains["trans_val"]["target"]["frame_f2"],
        "cis_ap": domains["cis_val_clean"]["target"]["average_precision"],
        "trans_ap": domains["trans_val"]["target"]["average_precision"],
        "selection_score": evaluation["selection_score"]["primary"],
    }

    return {
        "model_id": model_id,
        "kind": kind,
        "run_id": candidate["candidate_id"],
        "source_run_id": candidate["source_run_id"],
        "seed": base["seed"],
        "input": evaluation["regime"]["input"],
        "params": base["params"],
        "macs": base["macs"],
        "quantization": {
            "method": candidate["method"],
            "scheme": candidate["model"]["quantization"]["scheme"],
            "format": candidate["model"]["quantization"]["format"],
            "per_channel": candidate["model"]["quantization"]["per_channel"],
            "calibration_manifest_sha256": candidate["calibration"]["sha256"],
            "calibration_images": candidate["calibration"]["images"],
        },
        "model": {
            "artifact": policy["model"]["artifact"],
            "sha256": onnx_sha256,
            "bytes": artifact.stat().st_size,
        },
        "evaluation_regime": evaluation["regime"],
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


def pruned_params_and_macs(candidate: dict) -> tuple[int, int]:
    """A pruned candidate's own params/MACs, re-derived from its checkpoint.

    The M0-row copy is wrong for M3/M4 — pruning is precisely a change of
    params and MACs — so the numbers are measured the way M0's were: load the
    hash-verified best checkpoint into the candidate's recorded architecture
    and count. The ladder MAC convention (`macs_of_model`) keeps the column
    comparable with every other row.
    """
    import torch

    from .models.mobilenet import build_mobilenet_v2
    from .optimize.prune import apply_widths
    from .validate.input_cost import macs_of_model

    run_dir = Path(candidate["finetune_run_dir"])
    checkpoint_path = run_dir / BEST_CHECKPOINT
    measured = sha256_file(checkpoint_path)
    if measured != candidate["best_checkpoint_sha256"]:
        raise RuntimeError(
            f"{checkpoint_path} does not hash to the candidate's record; its "
            "parameter count would describe an unknown file"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("epoch") != candidate["best_epoch"]:
        raise RuntimeError(
            f"best.pt holds epoch {checkpoint.get('epoch')} but the candidate "
            f"selected {candidate['best_epoch']}; not this candidate's model"
        )
    model = build_mobilenet_v2(
        num_classes=len(checkpoint["class_names"]), pretrained=False
    )
    apply_widths(model, candidate["pruning"]["widths"])
    model.load_state_dict(checkpoint["model"])
    params = sum(parameter.numel() for parameter in model.parameters())
    width = candidate["input"]["width"]
    height = candidate["input"]["height"]
    return params, macs_of_model(model, width, height)


def build_pruned_row(
    model_id: str,
    kind: str,
    candidate: dict,
    evaluation: dict,
    policy: dict,
    policy_path: Path,
    artifact: Path,
    onnx_sha256: str,
    parity_path: Path,
    params: int,
    macs: int,
) -> dict:
    calibration = policy["calibration"]
    domains = evaluation["domains"]
    pruning = candidate["pruning"]

    return {
        "model_id": model_id,
        "kind": kind,
        "run_id": candidate["candidate_id"],
        "source_run_id": candidate["source_run_id"],
        "seed": candidate["seed"],
        "input": evaluation["regime"]["input"],
        "params": params,
        "macs": macs,
        "pruning": {
            "method": candidate["method"],
            "target_fraction": pruning["target_fraction"],
            "realized_mac_reduction_tp": pruning["realized_mac_reduction_tp"],
            "param_reduction": pruning["param_reduction"],
            "widths": pruning["widths"],
            "pre_finetune_primary": pruning["pre_finetune_primary"],
            "finetune_run_id": candidate["finetune_run_id"],
            "best_epoch": candidate["best_epoch"],
        },
        "model": {
            "artifact": policy["model"]["artifact"],
            "sha256": onnx_sha256,
            "bytes": artifact.stat().st_size,
        },
        "evaluation_regime": evaluation["regime"],
        "validation_at_0p5": {
            "cis_f2": domains["cis_val_clean"]["target"]["frame_f2"],
            "trans_f2": domains["trans_val"]["target"]["frame_f2"],
            "cis_ap": domains["cis_val_clean"]["target"]["average_precision"],
            "trans_ap": domains["trans_val"]["target"]["average_precision"],
            "selection_score": evaluation["selection_score"]["primary"],
        },
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
    parser.add_argument("--run", type=Path, help="a training run (M0)")
    parser.add_argument("--candidate", type=Path,
                        help="an optimize candidate directory (M1/M2)")
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--model-id", required=True, help="M0…M4")
    parser.add_argument("--kind", required=True,
                        help="fp32_baseline | int8_ptq | int8_qat | pruned_fp32 | pruned_qat")
    parser.add_argument("--parity", type=Path,
                        help="default: the report the policy itself names")
    parser.add_argument("--base-model-id", default="M0",
                        help="whose committed row supplies params/MACs (--candidate mode)")
    parser.add_argument("--table", type=Path, default=TABLE_PATH)
    args = parser.parse_args()

    if bool(args.run) == bool(args.candidate):
        parser.error("exactly one of --run or --candidate is required")

    if args.candidate:
        candidate, evaluation, policy = load_candidate_row_inputs(
            args.candidate, args.policy
        )
        artifact, onnx_sha256 = verify_artifact(policy)
        parity_path = args.parity or Path(policy["model"]["parity"])
        verify_parity(parity_path, onnx_sha256)
        if candidate.get("kind") in ("pruned_fp32", "pruned_qat"):
            # Pruning changes params/MACs — the base-row copy would be a lie.
            params, macs = pruned_params_and_macs(candidate)
            row = build_pruned_row(
                args.model_id, args.kind, candidate, evaluation, policy,
                args.policy, artifact, onnx_sha256, parity_path, params, macs,
            )
        else:
            base = base_row(args.table, args.base_model_id)
            row = build_candidate_row(
                args.model_id, args.kind, candidate, evaluation, policy,
                args.policy, artifact, onnx_sha256, parity_path, base,
            )
    else:
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
          f"(params {row['params']:,}, macs {row['macs']:,}, "
          f"model {row['model']['bytes']:,} B, "
          f"cis F2 {row['validation_at_0p5']['cis_f2']:.4f}, "
          f"trans F2 {row['validation_at_0p5']['trans_f2']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
