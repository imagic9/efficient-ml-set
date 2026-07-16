#!/usr/bin/env python3
"""Calibrate a candidate's bobcat policy from its own ORT validation scores.

`wildlife_trigger.calibrate` is C3's tool: it reads a *training run's*
`predictions.npz` (torch, deployment regime) and binds the policy to the
checkpoint, to be re-bound to the ONNX after P2. A D-phase candidate needs none
of that indirection — `optimize.evaluate_onnx` already scored the deployable
artifact itself, through deployment ORT, so the policy binds **directly** to the
ONNX whose outputs produced the calibration scores. There is no re-bind step
because there is no gap to bridge: the artifact hash in the policy and the
arithmetic behind the threshold name the same bytes from the start.

The rule is the frozen §6.3 rule, verbatim — same `metrics.select_threshold`,
same fast-path cross-check, same bootstrap, same verbatim status recording.
`recall_floor_infeasible` ships an operating point and is NOT a pass; the
policy artifact carries the status so it cannot be quoted without it (the C3
contract, unchanged).

The frozen class map is *read and verified*, never rewritten: candidates do not
get to move `artifacts/class_map.json`, they get refused if they disagree with
it.

Usage (gx10):
    python -m wildlife_trigger.optimize.calibrate_candidate \
        --candidate results/optimize/m1_ptq/minmax \
        --policy-id bobcat_m1_ptq_minmax_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .. import metrics
from ..calibrate import DOMAINS, metric_intervals, score_histograms
from ..policy import (
    ANIMAL_CLASSES,
    NO_THRESHOLD_CLASSES,
    build_class_map,
    build_policy,
    write_canonical_json,
)
from ..runs import atomic_write_json, sha256_file


def load_candidate(candidate_dir: Path, target: str) -> tuple[dict, dict]:
    """The candidate's scores, cross-checked against its own evidence chain.

    Three hashes must agree before a threshold is computed: the npz's recorded
    model hash, the evaluation record's, and the bytes on disk right now. A
    mismatch anywhere means the scores describe a different artifact than the
    one the policy would bind, which is exactly the mistake this tool exists
    to make impossible.
    """
    evaluation = json.loads((candidate_dir / "evaluation.json").read_text())
    candidate = json.loads((candidate_dir / "candidate.json").read_text())
    npz_path = candidate_dir / "predictions.npz"
    data = np.load(npz_path, allow_pickle=False)

    model_path = Path(evaluation["model"]["path"])
    recorded = evaluation["model"]["sha256"]
    if str(data["model_sha256"]) != recorded:
        raise RuntimeError(
            "predictions.npz and evaluation.json disagree on the model hash; "
            "these files are from different scoring runs"
        )
    if candidate["model"]["sha256"] != recorded:
        raise RuntimeError(
            "candidate.json describes a different artifact than evaluation.json "
            "scored; the candidate directory is inconsistent"
        )
    measured = sha256_file(model_path)
    if measured != recorded:
        raise RuntimeError(
            f"{model_path} hashes to {measured[:12]}… but the scores were "
            f"produced by {recorded[:12]}…; whatever this file is now, the "
            "calibration would not describe it"
        )

    class_names = [str(name) for name in data["class_names"]]
    if class_names != evaluation["class_names"]:
        raise RuntimeError(
            "predictions.npz and evaluation.json disagree on the class order; "
            "a threshold calibrated on the wrong column fires on the wrong animal"
        )
    if target not in class_names:
        raise RuntimeError(f"{target!r} is not one of this candidate's classes")
    column = class_names.index(target)

    scores_by_domain = {}
    for domain in DOMAINS:
        scores_by_domain[domain] = (
            data[f"{domain}/probabilities"][:, column].astype(float),
            data[f"{domain}/present"][:, column].astype(float),
            [str(s) for s in data[f"{domain}/seq_ids"]],
        )

    context = {
        "candidate_id": candidate["candidate_id"],
        "label": evaluation["label"],
        "method": candidate.get("method"),
        "model_path": str(model_path),
        "model_sha256": recorded,
        "class_names": class_names,
        "target_column": column,
        "predictions_sha256": sha256_file(npz_path),
        "regime": evaluation["regime"],
        "manifests": evaluation["manifests"],
        "source_run_id": candidate.get("source_run_id"),
    }
    return scores_by_domain, context


def frozen_class_map(artifacts_root: Path, class_names: list[str]) -> tuple[dict, str]:
    """artifacts/class_map.json, verified against this candidate's order."""
    path = artifacts_root / "class_map.json"
    class_map = json.loads(path.read_text())
    expected = build_class_map(class_names)
    if class_map != expected:
        raise RuntimeError(
            f"{path} does not match this candidate's class order; the frozen "
            "class map does not move for a candidate — the candidate is wrong"
        )
    return class_map, sha256_file(path)


def calibrate_candidate(
    candidate_dir: Path,
    policy_id: str,
    target: str = "bobcat",
    replicates: int = 1000,
    seed: int = 42,
    artifacts_root: Path = Path("artifacts"),
) -> dict:
    # Catalog eligibility first, same order as C3: a §4 fact, not a score shape.
    if target in NO_THRESHOLD_CLASSES:
        raise ValueError(
            f"{target!r} has no calibrated threshold in the DESIGN §4 catalog "
            "(insufficient validation support); calibrating one would invent it"
        )
    if target not in ANIMAL_CLASSES:
        raise ValueError(f"{target!r} is not a selectable animal class")

    scores_by_domain, context = load_candidate(candidate_dir, target)

    selection = metrics.select_threshold(scores_by_domain)
    fast_threshold, fast_status = metrics.select_threshold_point(scores_by_domain)
    if (fast_threshold, fast_status) != (selection["threshold"], selection["status"]):
        raise RuntimeError(
            f"select_threshold_point returned ({fast_threshold}, {fast_status!r}) "
            f"but select_threshold returned ({selection['threshold']}, "
            f"{selection['status']!r}). The fast path has drifted from the "
            "registered rule; fix that before calibrating anything."
        )

    bootstrap = metrics.bootstrap_threshold_selection(
        scores_by_domain, replicates=replicates, seed=seed
    )
    threshold = selection["threshold"]
    strata = {
        domain: metrics.positive_sequence_length_report(scores, present, seq_ids, threshold)
        for domain, (scores, present, seq_ids) in scores_by_domain.items()
    }

    record = {
        "tool": "wildlife_trigger.optimize.calibrate_candidate",
        "design": "6.3 (deployment-regime amendment, applied via ORT scores)",
        "candidate_id": context["candidate_id"],
        "candidate_dir": str(candidate_dir),
        "method": context["method"],
        "source_run_id": context["source_run_id"],
        "target": target,
        "target_column": context["target_column"],
        "regime": context["regime"],
        "inputs": {
            "predictions_npz_sha256": context["predictions_sha256"],
            "model": {"path": context["model_path"], "sha256": context["model_sha256"]},
            "manifests": context["manifests"],
        },
        "selection": selection,
        "threshold_bootstrap": bootstrap,
        "length_strata": strata,
        "metric_intervals": metric_intervals(scores_by_domain, threshold, seed),
        "score_histograms": score_histograms(scores_by_domain),
    }
    calibration_path = candidate_dir / "calibration.json"
    atomic_write_json(calibration_path, record)
    written = {"calibration": str(calibration_path)}

    if selection["status"] == "fire_budget_infeasible":
        print(
            f"status: {selection['status']} — no admissible threshold, no policy "
            "written. This candidate cannot ship a trigger."
        )
        return {**record, "written": written}

    class_map, class_map_sha256 = frozen_class_map(artifacts_root, context["class_names"])

    policy = build_policy(
        policy_id=policy_id,
        targets=[{"class": target, "threshold": threshold}],
        class_map=class_map,
        class_map_sha256=class_map_sha256,
        model_sha256=context["model_sha256"],
        metadata={
            "model": {
                "artifact": context["model_path"],
                "kind": "onnx",
                "binding": (
                    "model_sha256 names the quantized ONNX directly: the "
                    "calibration scores are that artifact's own ORT outputs "
                    "(optimize.evaluate_onnx), so there is no checkpoint→ONNX "
                    "re-bind step. P3 attaches its parity report after it passes."
                ),
            },
            "calibration": {
                "status": selection["status"],
                "primary_rule_met": selection["primary_rule_met"],
                "rule": selection["rule"],
                "min_sequence_recall_required": selection["min_sequence_recall_required"],
                "max_false_fire_rate_allowed": selection["max_false_fire_rate_allowed"],
                "unmet_constraint": selection["unmet_constraint"],
                "per_domain": {
                    domain: {
                        key: measured[key]
                        for key in (
                            "sequence_balanced_recall",
                            "frame_recall",
                            "frame_precision",
                            "frame_f2",
                            "false_fire_rate",
                            "fire_rate",
                            "event_capture_rate",
                        )
                    }
                    for domain, measured in selection["per_domain"].items()
                },
                "threshold_ci95": [bootstrap["ci95_low"], bootstrap["ci95_high"]],
                "replicates": replicates,
                "seed": seed,
                "datasets": context["manifests"],
                "run_id": context["candidate_id"],
                "record": str(calibration_path),
            },
        },
    )
    policy_path = artifacts_root / "policies" / f"{policy_id}.json"
    write_canonical_json(policy_path, policy)
    written["policy"] = str(policy_path)

    return {**record, "written": written}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--target", default="bobcat")
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    result = calibrate_candidate(
        args.candidate,
        policy_id=args.policy_id,
        target=args.target,
        replicates=args.replicates,
        seed=args.seed,
        artifacts_root=args.artifacts_root,
    )

    selection = result["selection"]
    bootstrap = result["threshold_bootstrap"]
    print(f"candidate: {result['candidate_id']} ({result['method']})")
    print(f"status: {selection['status']} | primary_rule_met: {selection['primary_rule_met']}")
    print(
        f"threshold: {selection['threshold']:.6f} "
        f"(admissible {selection['admissible_candidates']}/{selection['candidates_searched']}; "
        f"bootstrap 95% [{bootstrap['ci95_low']:.4f}, {bootstrap['ci95_high']:.4f}])"
    )
    for domain, measured in selection["per_domain"].items():
        print(
            f"  {domain}: seq_recall={measured['sequence_balanced_recall']:.4f} "
            f"ff={measured['false_fire_rate']:.4f} fire={measured['fire_rate']:.4f}"
        )
    for name, path in result["written"].items():
        print(f"wrote {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
