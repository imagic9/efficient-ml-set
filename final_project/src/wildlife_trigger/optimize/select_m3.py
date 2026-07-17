#!/usr/bin/env python3
"""D4: select M3 from the fine-tuned candidates — the registered rule as code.

`results/optimize/m3_prune/m3_registration.md` §4, applied mechanically:

1. a candidate participates only with a complete evidence chain — candidate
   record, ORT evaluation whose hash matches the artifact on disk, and a
   calibrated policy (a calibration that refused to write one is a gate
   failure, not a formatting problem);
2. dominated candidates drop: A dominates B when A's primary >= B's and A's
   ladder MACs <= B's, at least one strict;
3. among the non-dominated, the largest realized MAC reduction whose primary
   >= 0.95 x the M0 ORT reference wins;
4. if none reaches the line, the highest-primary non-dominated candidate wins
   and the failure of the line is recorded in the verdict — "pruning hurts
   more than 5%" is a result, not an emergency.

Usage (gx10):
    python -m wildlife_trigger.optimize.select_m3 \
        --root results/optimize/m3_prune --targets c15 c30 c45 \
        --reference results/optimize/m1_ptq/m0_fp32_reference/evaluation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..runs import atomic_write_json, sha256_file

# The D1 material-drop line, reused verbatim by the registration.
RECOVERY_LINE = 0.95


def load_candidate(root: Path, label: str, artifacts_root: Path) -> dict:
    candidate_dir = root / label
    candidate = json.loads((candidate_dir / "candidate.json").read_text())
    evaluation = json.loads((candidate_dir / "evaluation.json").read_text())
    calibration = json.loads((candidate_dir / "calibration.json").read_text())

    if evaluation["model"]["sha256"] != candidate["model"]["sha256"]:
        raise RuntimeError(f"{label}: evaluation and candidate describe different artifacts")
    measured = sha256_file(Path(evaluation["model"]["path"]))
    if measured != candidate["model"]["sha256"]:
        raise RuntimeError(
            f"{label}: {evaluation['model']['path']} hashes to {measured[:12]}…, "
            "not the recorded artifact; selection refuses moved bytes"
        )

    policy_path = artifacts_root / "policies" / f"bobcat_{root.name}_{label}_v1.json"
    if not policy_path.exists():
        raise RuntimeError(
            f"{label}: no calibrated policy at {policy_path}; a candidate "
            "without an operating point failed its gate and cannot be selected"
        )
    policy = json.loads(policy_path.read_text())
    if policy["model_sha256"] != candidate["model"]["sha256"]:
        raise RuntimeError(f"{label}: the policy is bound to a different artifact")

    return {
        "label": label,
        "candidate_id": candidate["candidate_id"],
        "primary": evaluation["selection_score"]["primary"],
        "cis_f2": evaluation["domains"]["cis_val_clean"]["target"]["frame_f2"],
        "trans_f2": evaluation["domains"]["trans_val"]["target"]["frame_f2"],
        "macs_ladder": candidate["pruning"]["macs_ladder_convention"],
        "realized_mac_reduction_tp": candidate["pruning"]["realized_mac_reduction_tp"],
        "params": candidate["pruning"]["params"],
        "bytes": candidate["model"]["size_bytes"],
        "pre_finetune_primary": candidate["pruning"]["pre_finetune_primary"],
        "threshold": policy["targets"][0]["threshold"],
        "calibration_status": calibration["selection"]["status"],
        "policy_path": str(policy_path),
    }


def dominated(a: dict, b: dict) -> bool:
    """Is `b` dominated by `a` on (primary up, ladder MACs down)?"""
    at_least_as_good = a["primary"] >= b["primary"] and a["macs_ladder"] <= b["macs_ladder"]
    strictly_better = a["primary"] > b["primary"] or a["macs_ladder"] < b["macs_ladder"]
    return at_least_as_good and strictly_better


def select(rows: list[dict], reference_primary: float) -> dict:
    line = RECOVERY_LINE * reference_primary
    non_dominated = [
        row
        for row in rows
        if not any(dominated(other, row) for other in rows if other is not row)
    ]
    above_line = [row for row in non_dominated if row["primary"] >= line]

    if above_line:
        chosen = max(above_line, key=lambda r: r["realized_mac_reduction_tp"])
        rule = (
            f"largest realized MAC reduction among non-dominated candidates with "
            f"primary >= {RECOVERY_LINE} x reference ({line:.4f})"
        )
        line_met = True
    else:
        chosen = max(non_dominated, key=lambda r: r["primary"])
        rule = (
            f"no non-dominated candidate reached {RECOVERY_LINE} x reference "
            f"({line:.4f}); highest primary selected and the miss is the finding"
        )
        line_met = False

    return {
        "selected": chosen["label"],
        "selected_candidate_id": chosen["candidate_id"],
        "rule": rule,
        "recovery_line": line,
        "recovery_line_met": line_met,
        "reference_primary": reference_primary,
        "non_dominated": [r["label"] for r in non_dominated],
        "dominated": [r["label"] for r in rows if r["label"] not in {n["label"] for n in non_dominated}],
    }


def render_markdown(verdict: dict, rows: list[dict], reference: dict) -> str:
    lines = [
        "# D4 — M3 candidate selection",
        "",
        "Rule: `results/optimize/m3_prune/m3_registration.md` §4 — applied "
        "mechanically by `wildlife_trigger.optimize.select_m3`.",
        "",
        f"Reference (M0 FP32 through deployment ORT): primary "
        f"{verdict['reference_primary']:.4f}; the recovery line is "
        f"{RECOVERY_LINE} x that = {verdict['recovery_line']:.4f}.",
        "",
        "| candidate | primary | cis F2 | trans F2 | pre-FT primary | ladder MACs | tp MAC cut | params | bytes | threshold | status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(rows, key=lambda r: -r["primary"]):
        lines.append(
            f"| {row['label']} | {row['primary']:.4f} | {row['cis_f2']:.4f} | "
            f"{row['trans_f2']:.4f} | {row['pre_finetune_primary']:.4f} | "
            f"{row['macs_ladder']:,} | {row['realized_mac_reduction_tp']:.2%} | "
            f"{row['params']:,} | {row['bytes']:,} | {row['threshold']:.6f} | "
            f"{row['calibration_status']} |"
        )
    lines += [
        "",
        f"Non-dominated: {', '.join(verdict['non_dominated'])}"
        + (f"; dominated: {', '.join(verdict['dominated'])}" if verdict["dominated"] else "; none dominated"),
        "",
        f"**Selected: {verdict['selected']}** (`{verdict['selected_candidate_id']}`) — "
        + verdict["rule"] + ".",
        "",
        f"Recovery line met: **{verdict['recovery_line_met']}**.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/optimize/m3_prune"))
    parser.add_argument("--targets", nargs="+", default=["c15", "c30", "c45"])
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("results/optimize/m1_ptq/m0_fp32_reference/evaluation.json"),
        help="the M0 deployment-ORT evaluation the recovery line anchors to",
    )
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    reference = json.loads(args.reference.read_text())
    rows = [load_candidate(args.root, label, args.artifacts_root) for label in args.targets]
    verdict = select(rows, reference["selection_score"]["primary"])

    record = {
        "tool": "wildlife_trigger.optimize.select_m3",
        "registration": "results/optimize/m3_prune/m3_registration.md",
        "reference": {
            "path": str(args.reference),
            "primary": reference["selection_score"]["primary"],
            "model_sha256": reference["model"]["sha256"],
        },
        "candidates": rows,
        "verdict": verdict,
    }
    atomic_write_json(args.root / "selection.json", record)
    (args.root / "selection.md").write_text(render_markdown(verdict, rows, reference))
    print(f"selected: {verdict['selected']} ({verdict['rule']})")
    print(f"wrote {args.root / 'selection.json'}")
    print(f"wrote {args.root / 'selection.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
