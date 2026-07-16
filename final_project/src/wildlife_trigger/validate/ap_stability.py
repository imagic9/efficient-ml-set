#!/usr/bin/env python3
"""The §7.2 amendment's pre-registered acceptance test (issue #19).

"AP is more stable than F2@0.5" was the argument for changing the checkpoint rule, and
an argument for a rule change is exactly the kind of claim that must be measured before
it is believed. Both criteria were registered in DESIGN §7.2 *before* the AP-selected
runs were trained; this tool computes them and returns a verdict, not a narrative.

Criterion 1 — trajectory noise. On the re-run M0's phase-B epochs (which record both
metrics side by side, so the comparison is on identical weights), the mean absolute
epoch-to-epoch change of the AP selection score, relative to its phase-B mean, must be
at most HALF that of the F2@0.5 score.

Criterion 2 — sampling noise at the chosen point. The seq_id-cluster bootstrap 95% CI
half-width of trans-val AP at the selected checkpoint, relative to its point estimate,
must be no wider than that of trans-val F2@0.5 at the same checkpoint. AP removes the
arbitrary-threshold noise; it does not remove sequence-sampling noise, and this
criterion is what keeps that distinction honest.

Failing either reverts the rule (DESIGN §7.2 amendment) and closes #19 as checked and
not adopted.

Usage:
    python -m wildlife_trigger.validate.ap_stability \
        --run results/training/c2/<run_id> --output <run_id>/ap_stability.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .. import metrics as M

SPLITS = ("cis_val_clean", "trans_val")


def score_series(history: list[dict]) -> dict[str, list[float]]:
    """Both selection scores per phase-B epoch, from one trajectory.

    The F2 series is reconstructed the way the old rule computed it — mean of the two
    domains' frame F2 — from the same entries that carry AP, so nothing here depends on
    comparing two different training runs.
    """
    phase_b = [e for e in history if e["phase"] == "B"]
    return {
        "ap": [
            (e["cis_val_clean"]["average_precision"] + e["trans_val"]["average_precision"]) / 2
            for e in phase_b
        ],
        "f2_at_half": [
            (e["cis_val_clean"]["frame_f2"] + e["trans_val"]["frame_f2"]) / 2
            for e in phase_b
        ],
        "epochs": [e["epoch"] for e in phase_b],
    }


def relative_step_change(series: list[float]) -> float:
    """mean |x_t - x_{t-1}| / mean(x): scale-free epoch-to-epoch jitter."""
    values = np.asarray(series, dtype=float)
    if len(values) < 2 or values.mean() == 0:
        return float("nan")
    return float(np.abs(np.diff(values)).mean() / values.mean())


def relative_ci_half_width(predictions: Path, metric: str, replicates: int, seed: int) -> dict:
    """seq_id-cluster bootstrap CI of trans-val `metric` at the selected checkpoint."""
    data = np.load(predictions, allow_pickle=False)
    class_names = [str(n) for n in data["class_names"]]
    target = class_names.index("bobcat")

    result = M.bootstrap_sequence_clusters(
        data["trans_val/probabilities"][:, target],
        data["trans_val/present"][:, target],
        [str(s) for s in data["trans_val/seq_ids"]],
        threshold=0.5,
        metric=metric,
        replicates=replicates,
    )
    half_width = (result["ci95_high"] - result["ci95_low"]) / 2
    point = result["point_estimate"]
    return {
        "point_estimate": point,
        "ci95": [result["ci95_low"], result["ci95_high"]],
        "half_width": half_width,
        "relative_half_width": half_width / point if point else float("inf"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="an AP-selected run dir")
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = json.loads((args.run / "history.json").read_text())
    series = score_series(summary["history"])

    jitter_ap = relative_step_change(series["ap"])
    jitter_f2 = relative_step_change(series["f2_at_half"])

    bootstrap = {
        metric: relative_ci_half_width(
            args.run / "predictions.npz", metric, args.replicates, args.seed
        )
        for metric in ("average_precision", "frame_f2")
    }

    criterion_1 = bool(jitter_ap <= 0.5 * jitter_f2)
    criterion_2 = bool(
        bootstrap["average_precision"]["relative_half_width"]
        <= bootstrap["frame_f2"]["relative_half_width"]
    )

    report = {
        "run": str(args.run),
        "phase_b_epochs": len(series["epochs"]),
        "criterion_1_trajectory": {
            "requirement": "AP relative epoch-to-epoch change <= 0.5 x F2@0.5's",
            "ap_series": [round(v, 4) for v in series["ap"]],
            "f2_series": [round(v, 4) for v in series["f2_at_half"]],
            "ap_relative_step_change": round(jitter_ap, 4),
            "f2_relative_step_change": round(jitter_f2, 4),
            "ratio_ap_over_f2": round(jitter_ap / jitter_f2, 4) if jitter_f2 else None,
            "passes": criterion_1,
        },
        "criterion_2_bootstrap_trans": {
            "requirement": (
                "relative CI half-width of trans AP <= that of trans F2@0.5, "
                "seq_id clusters, at the selected checkpoint"
            ),
            "average_precision": {
                k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v])
                for k, v in bootstrap["average_precision"].items()
            },
            "frame_f2_at_half": {
                k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v])
                for k, v in bootstrap["frame_f2"].items()
            },
            "passes": criterion_2,
        },
        "verdict": "adopted" if (criterion_1 and criterion_2) else "revert_to_f2",
        "registered_in": "DESIGN §7.2 amendment 2026-07-16, before these runs were trained",
    }

    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    return 0 if report["verdict"] == "adopted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
