#!/usr/bin/env python3
"""C5 — baseline training variability across the confirmation seeds (DESIGN §7.2).

Answers issue #18's question with measurements instead of one seed's anecdote:
seed 42's trans-val F2 never left [0.037, 0.109] while a shorter-budget C1a arm
touched 0.268 — is that the recipe or the seed? Three runs of the frozen recipe,
differing in nothing but the seed, are the experiment that separates those. This
tool reads their committed histories and aggregates; it computes nothing that is
not already in the runs' own records.

The runs must actually be the same experiment. The §7.2 recipe fields and the
dataset manifests are compared before any number is aggregated, and a mismatch is
a refusal, not a footnote: averaging runs that differ in anything but the seed
would report variability of an experiment nobody ran.

Std is the n-1 sample estimate over three seeds — reported with its n, because a
three-point std is a scale indicator, not a distribution claim. The bootstrap CIs
of §6.3 remain the uncertainty measure for any single model's metrics.

Usage:
    python -m wildlife_trigger.validate.seed_variability \
        --runs results/training/c2/c2_m0_fp32_seed42_20260716T061203Z \
               results/training/c5/c5_m0_fp32_seed17_<ts>Z \
               results/training/c5/c5_m0_fp32_seed73_<ts>Z \
        --output results/training/c5/seed_variability.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from ..runs import atomic_write_json, resolve_run_id

# DESIGN §7.2's frozen recipe, the fields a confirmation seed must not move.
# run_name/seed/phase are the run's identity; paths and workers are box layout.
RECIPE_FIELDS = (
    "width", "height", "head_epochs", "max_epochs", "batch_size", "head_lr",
    "full_lr", "weight_decay", "early_stopping_patience", "amp",
    "exclude_empty_class",
)

HEADLINE = ("frame_f2", "sequence_balanced_recall", "event_capture_rate")


def load_runs(run_dirs: list[Path]) -> list[dict]:
    runs = []
    for run_dir in run_dirs:
        history = json.loads((run_dir / "history.json").read_text())
        hashes = json.loads((run_dir / "hashes.json").read_text())
        history["_run_dir"] = str(run_dir)
        history["_run_id"] = resolve_run_id(run_dir, history["run_name"])
        history["_manifests"] = {
            key: value["sha256"] if isinstance(value, dict) else value
            for key, value in hashes.items()
            if key.startswith("manifest:")
        }
        runs.append(history)
    return runs


def verify_same_experiment(runs: list[dict]) -> None:
    if len(runs) < 2:
        raise RuntimeError("variability needs at least two runs to compare")

    seeds = [run["config"]["seed"] for run in runs]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError(
            f"duplicate seeds {seeds}: the same run twice measures nothing"
        )

    reference = runs[0]
    for run in runs[1:]:
        for field in RECIPE_FIELDS:
            ours, theirs = reference["config"][field], run["config"][field]
            if ours != theirs:
                raise RuntimeError(
                    f"{run['_run_id']} trained with {field}={theirs} but "
                    f"{reference['_run_id']} used {ours}; these runs are different "
                    "experiments and their spread is not seed variability"
                )
        if run["_manifests"] != reference["_manifests"]:
            raise RuntimeError(
                f"{run['_run_id']} saw different dataset manifests than "
                f"{reference['_run_id']}; seed variability must hold the data fixed"
            )
        if run["class_names"] != reference["class_names"]:
            raise RuntimeError(
                f"{run['_run_id']} used a different class map than "
                f"{reference['_run_id']}"
            )


def best_entry(run: dict) -> dict:
    (entry,) = [e for e in run["history"] if e["epoch"] == run["best_epoch"]]
    return entry


def summarise_run(run: dict) -> dict:
    best = best_entry(run)
    phase_b = [e for e in run["history"] if e["phase"] == "B"]
    trans_trajectory = [e["trans_val"]["frame_f2"] for e in phase_b]

    domains = {}
    for domain in ("cis_val_clean", "trans_val"):
        domains[domain] = {key: best[domain][key] for key in HEADLINE}
        if "average_precision" in best[domain]:
            domains[domain]["average_precision"] = best[domain]["average_precision"]

    return {
        "run_id": run["_run_id"],
        "seed": run["config"]["seed"],
        "best_epoch": run["best_epoch"],
        "epochs_trained": len(run["history"]),
        "selection_score": best["selection_score"]["primary"],
        "at_best": domains,
        "trans_f2_trajectory": {
            "phase_b_epochs": len(trans_trajectory),
            "min": min(trans_trajectory),
            "max": max(trans_trajectory),
            "values": trans_trajectory,
        },
    }


def aggregate(summaries: list[dict]) -> dict:
    def spread(values: list[float]) -> dict:
        return {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values),  # n-1; n is named in the payload
            "min": min(values),
            "max": max(values),
            "values": values,
        }

    result = {
        "n_seeds": len(summaries),
        "selection_score": spread([s["selection_score"] for s in summaries]),
        "trans_f2_max_over_epochs": spread(
            [s["trans_f2_trajectory"]["max"] for s in summaries]
        ),
    }
    for domain in ("cis_val_clean", "trans_val"):
        result[domain] = {
            key: spread([s["at_best"][domain][key] for s in summaries])
            for key in HEADLINE
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    runs = load_runs(args.runs)
    verify_same_experiment(runs)
    summaries = sorted((summarise_run(run) for run in runs), key=lambda s: s["seed"])

    payload = {
        "question": "issue #18: is M0's trans-val gap the recipe or seed 42?",
        "recipe": {
            field: runs[0]["config"][field] for field in RECIPE_FIELDS
        },
        "std_note": "std is the n-1 sample estimate over n_seeds runs; "
        "a three-point std indicates scale, not a distribution",
        "per_seed": summaries,
        "aggregate": aggregate(summaries),
    }
    atomic_write_json(args.output, payload)

    for s in summaries:
        trajectory = s["trans_f2_trajectory"]
        print(
            f"seed {s['seed']:>2}  best epoch {s['best_epoch']:>2}  "
            f"score {s['selection_score']:.4f}  "
            f"cis F2 {s['at_best']['cis_val_clean']['frame_f2']:.4f}  "
            f"trans F2 {s['at_best']['trans_val']['frame_f2']:.4f}  "
            f"trans F2 range over B [{trajectory['min']:.4f}, {trajectory['max']:.4f}]"
        )
    agg = payload["aggregate"]
    print(
        f"mean±std (n={agg['n_seeds']}):  "
        f"cis F2 {agg['cis_val_clean']['frame_f2']['mean']:.4f}"
        f"±{agg['cis_val_clean']['frame_f2']['std']:.4f}  "
        f"trans F2 {agg['trans_val']['frame_f2']['mean']:.4f}"
        f"±{agg['trans_val']['frame_f2']['std']:.4f}"
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
