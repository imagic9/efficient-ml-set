#!/usr/bin/env python3
"""C3: calibrate the deployed operating point from a run's saved validation predictions.

DESIGN §6.3's rule, applied end to end: search the observed bobcat scores on
cis-val-clean + trans-val, keep the thresholds inside the 5% per-domain false-fire
budget, take the largest one meeting the 90% sequence-balanced recall floor on both
domains — and *record the status verbatim* when that threshold does not exist.
`recall_floor_infeasible` ships an operating point and is NOT a pass; no table,
slide or README line downstream of this tool may describe it as one, and the policy
artifact itself carries the failure so it cannot be quoted without it.

Reads `predictions.npz` (written by `validate.dump_predictions` from the selected
checkpoint), never the checkpoint itself: C3 must calibrate from the exact frames
the run validated on, not from a re-inference through a data pipeline that may have
moved since. Only the two validation splits exist in that file — DESIGN §5.4 keeps
the test sets sealed, and this tool cannot be pointed at them.

Writes three artifacts:

- `results/evaluation/<run_id>/calibration.json` — the full record: the rule's
  verdict, the recall/false-fire curve (step 5), the threshold bootstrap (step 7),
  length strata, per-metric CIs, score histograms, and the hashes of everything
  the numbers depend on;
- `artifacts/class_map.json` — the real class map, in the run's frozen B1 order;
- `artifacts/policies/<target>_v1.json` — the deployable policy, bound by hash to
  the class map and to the calibrated checkpoint. `fire_budget_infeasible` writes
  no policy: a device that cannot meet its own fire budget has no operating point
  to ship (§6.3 step 4), and an artifact would be that verdict with a number on it.

The policy's `model_sha256` names the *PyTorch checkpoint* the scores came from —
the only model artifact that exists before C4. The C++ loader will therefore refuse
this policy against the FP32 ONNX until C4 re-binds it after P2 proves PyTorch→ORT
parity. That is deliberate: a hash that fails loudly until the parity proof exists
beats an empty binding that never fails at all.

Usage:
    python -m wildlife_trigger.calibrate \
        --run results/training/c2/c2_m0_fp32_seed42_20260716T061203Z
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import metrics
from .policy import (
    ANIMAL_CLASSES,
    NO_THRESHOLD_CLASSES,
    build_class_map,
    build_policy,
    write_canonical_json,
)
from .runs import atomic_write_json, resolve_run_id, sha256_file

# The same two splits dump_predictions can reach, under the domain names DESIGN
# §6.3 uses. Nothing else is loadable here, by construction of the npz.
DOMAINS = ("cis_val_clean", "trans_val")

HISTOGRAM_BINS = 50


def load_predictions(run_dir: Path, target: str) -> tuple[dict, dict]:
    """The per-frame validation scores, cross-checked against the run's history.

    Returns `(scores_by_domain, context)` where each domain maps to the
    `(scores, present, seq_ids)` triple every metrics function takes.
    """
    history = json.loads((run_dir / "history.json").read_text())
    npz_path = run_dir / "predictions.npz"
    data = np.load(npz_path, allow_pickle=False)

    class_names = [str(name) for name in data["class_names"]]
    if class_names != history["class_names"]:
        raise RuntimeError(
            "predictions.npz and history.json disagree on the class order. These "
            "files are from different runs, and a threshold calibrated on the wrong "
            "column fires on the wrong animal."
        )
    if int(data["best_epoch"]) != history["best_epoch"]:
        raise RuntimeError(
            f"predictions.npz holds epoch {int(data['best_epoch'])} but the history "
            f"selected epoch {history['best_epoch']}. The predictions are not this "
            "run's selected model."
        )
    if target not in class_names:
        raise RuntimeError(f"{target!r} is not one of this run's classes")
    column = class_names.index(target)

    scores_by_domain = {}
    for domain in DOMAINS:
        probabilities = data[f"{domain}/probabilities"]
        present = data[f"{domain}/present"]
        seq_ids = [str(s) for s in data[f"{domain}/seq_ids"]]
        scores_by_domain[domain] = (
            probabilities[:, column].astype(float),
            present[:, column].astype(float),
            seq_ids,
        )

    run_id = resolve_run_id(run_dir, history["run_name"])

    context = {
        "run_id": run_id,
        "run_name": history["run_name"],
        "best_epoch": history["best_epoch"],
        "class_names": class_names,
        "target": target,
        "target_column": column,
        "predictions_sha256": sha256_file(npz_path),
    }
    return scores_by_domain, context


def score_histograms(scores_by_domain: dict) -> dict:
    """Positive/negative score histograms per domain (§6.3 step 6's distribution).

    Fixed uniform bins over [0, 1] so histograms from different models and
    quantization stages remain comparable bin for bin.
    """
    edges = np.linspace(0.0, 1.0, HISTOGRAM_BINS + 1)
    result = {"bin_edges": edges.tolist(), "bins": HISTOGRAM_BINS}
    for domain, (scores, present, _) in scores_by_domain.items():
        positives = present > 0
        result[domain] = {
            "positive": np.histogram(scores[positives], bins=edges)[0].tolist(),
            "negative": np.histogram(scores[~positives], bins=edges)[0].tolist(),
        }
    return result


def metric_intervals(scores_by_domain: dict, threshold: float, seed: int) -> dict:
    """seq_id-cluster bootstrap CIs for the reported metrics at the chosen point."""
    watched = ("frame_recall", "frame_precision", "frame_f2", "false_fire_rate")
    intervals: dict[str, dict] = {}
    for domain, (scores, present, seq_ids) in scores_by_domain.items():
        intervals[domain] = {
            metric: metrics.bootstrap_sequence_clusters(
                scores, present, seq_ids, threshold, metric=metric, seed=seed
            )
            for metric in watched
        }
    return intervals


def run_hashes(run_dir: Path) -> dict:
    """What the calibration depends on, lifted from the run's own fingerprints."""
    recorded = json.loads((run_dir / "hashes.json").read_text())
    kept = {
        key: {"path": value["path"], "sha256": value["sha256"]}
        for key, value in recorded.items()
        if key in ("manifest:cis_val_clean", "manifest:trans_val", "checkpoint:best")
    }
    missing = {"manifest:cis_val_clean", "manifest:trans_val", "checkpoint:best"} - set(kept)
    if missing:
        raise RuntimeError(
            f"{run_dir / 'hashes.json'} lacks {sorted(missing)}; without them the "
            "policy cannot be bound to what it was calibrated on"
        )
    return kept


def calibrate(
    run_dir: Path,
    target: str = "bobcat",
    replicates: int = 1000,
    seed: int = 42,
    output_root: Path = Path("results/evaluation"),
    artifacts_root: Path = Path("artifacts"),
) -> dict:
    # Catalog eligibility first, before any data is read: whether badger may carry
    # a threshold is a DESIGN §4 fact, not a property of this run's score shapes.
    if target in NO_THRESHOLD_CLASSES:
        raise ValueError(
            f"{target!r} has no calibrated threshold in the DESIGN §4 catalog "
            "(insufficient validation support); calibrating one would invent it"
        )
    if target not in ANIMAL_CLASSES:
        raise ValueError(f"{target!r} is not a selectable animal class")

    scores_by_domain, context = load_predictions(run_dir, target)
    hashes = run_hashes(run_dir)

    # The registered rule, on the full cleaned validation data. This is the record.
    selection = metrics.select_threshold(scores_by_domain)

    # Cross-check the bootstrap's fast path against the registered implementation
    # before trusting a thousand of its re-runs. A mismatch means the two have
    # drifted; neither number ships until a human decides which one is the rule.
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
        "tool": "wildlife_trigger.calibrate",
        "design": "6.3",
        "run_id": context["run_id"],
        "run_name": context["run_name"],
        "run_dir": str(run_dir),
        "best_epoch": context["best_epoch"],
        "target": target,
        "target_column": context["target_column"],
        "inputs": {
            "predictions_npz_sha256": context["predictions_sha256"],
            **hashes,
        },
        "selection": selection,
        "threshold_bootstrap": bootstrap,
        "length_strata": strata,
        "metric_intervals": metric_intervals(scores_by_domain, threshold, seed),
        "score_histograms": score_histograms(scores_by_domain),
    }

    output_dir = output_root / context["run_id"]
    atomic_write_json(output_dir / "calibration.json", record)
    written = {"calibration": str(output_dir / "calibration.json")}

    if selection["status"] == "fire_budget_infeasible":
        # §6.3 step 4: no operating point exists; naming one would be inventing it.
        print(
            f"status: {selection['status']} — no admissible threshold, no policy "
            "written. This model cannot ship a trigger."
        )
        return {**record, "written": written}

    class_map = build_class_map(context["class_names"])
    class_map_path = artifacts_root / "class_map.json"
    class_map_sha256 = write_canonical_json(class_map_path, class_map)
    written["class_map"] = str(class_map_path)

    policy = build_policy(
        policy_id=f"{target}_v1",
        targets=[{"class": target, "threshold": threshold}],
        class_map=class_map,
        class_map_sha256=class_map_sha256,
        model_sha256=hashes["checkpoint:best"]["sha256"],
        metadata={
            "model": {
                "artifact": hashes["checkpoint:best"]["path"],
                "kind": "pytorch_checkpoint",
                "binding": (
                    "model_sha256 names the calibrated checkpoint. C4 re-binds it "
                    "to the FP32 ONNX after P2 parity; until then the C++ loader "
                    "refuses this policy against any ONNX, which is the point."
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
                "datasets": {
                    key: value["sha256"]
                    for key, value in hashes.items()
                    if key.startswith("manifest:")
                },
                "run_id": context["run_id"],
                "best_epoch": context["best_epoch"],
                "record": str(output_dir / "calibration.json"),
            },
        },
    )
    policy_path = artifacts_root / "policies" / f"{target}_v1.json"
    write_canonical_json(policy_path, policy)
    written["policy"] = str(policy_path)

    return {**record, "written": written}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="a training run directory")
    parser.add_argument("--target", default="bobcat")
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=Path("results/evaluation"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    result = calibrate(
        args.run,
        target=args.target,
        replicates=args.replicates,
        seed=args.seed,
        output_root=args.output_root,
        artifacts_root=args.artifacts_root,
    )

    selection = result["selection"]
    bootstrap = result["threshold_bootstrap"]
    print(f"run: {result['run_id']} (best epoch {result['best_epoch']})")
    print(f"status: {selection['status']} | primary_rule_met: {selection['primary_rule_met']}")
    print(
        f"threshold: {selection['threshold']:.6f} "
        f"(admissible {selection['admissible_candidates']}/{selection['candidates_searched']} "
        f"candidates; bootstrap 95% [{bootstrap['ci95_low']:.4f}, {bootstrap['ci95_high']:.4f}])"
    )
    for domain, measured in selection["per_domain"].items():
        print(
            f"  {domain}: seq_recall={measured['sequence_balanced_recall']:.4f} "
            f"ff={measured['false_fire_rate']:.4f} fire={measured['fire_rate']:.4f} "
            f"event_capture={measured['event_capture_rate']:.4f}"
        )
    for name, path in result["written"].items():
        print(f"wrote {name}: {path}")
    # All three registered statuses are valid, recorded outcomes of a calibration
    # that ran to completion — the exit code reports tool failure, not model quality.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
