#!/usr/bin/env python3
"""Re-derive a run's best checkpoint from its own history and check it (PLAN C2).

C2's fourth bullet is "verify the selected checkpoint follows the configured rule".
The engine applying the rule cannot be the evidence that it applied the rule — that is
the shape of issue #12, where `train.run()` declared DESIGN §7.2's tie-breaks, computed
them every epoch, stored them, and compared only the primary. Everything looked right
from the outside.

So this replays the recorded history through `metrics.is_better_checkpoint` and checks
three things agree: the epoch the rule picks, the epoch the summary claims, and the epoch
inside `best.pt`. It reuses the comparator rather than restating it — a second
implementation would drift and would test itself.

Usage:
    python -m wildlife_trigger.validate.selection_audit \
        --run results/training/c2/c2_m0_fp32_seed42_20260716T061203Z
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import metrics as M


def replay(history: list[dict]) -> dict:
    """The selection, re-derived from the epoch records alone.

    Phase B only, in ascending epoch order — which is what makes "an exact tie keeps the
    earlier epoch" a rule rather than an accident of iteration order.
    """
    best: dict = {"score": None, "epoch": -1}
    for entry in sorted(history, key=lambda e: e["epoch"]):
        if entry["phase"] != "B":
            continue
        if M.is_better_checkpoint(entry["selection_score"], best["score"]):
            best = {"score": entry["selection_score"], "epoch": entry["epoch"]}
    return best


def audit(run_dir: Path, checkpoint: str = "best.pt") -> dict:
    summary = json.loads((run_dir / "history.json").read_text())
    derived = replay(summary["history"])

    problems = []
    if derived["epoch"] != summary["best_epoch"]:
        problems.append(
            f"the rule selects epoch {derived['epoch']}, the summary claims "
            f"{summary['best_epoch']}"
        )
    if summary.get("best_selection_score") and derived["score"] != summary["best_selection_score"]:
        problems.append(
            "the summary's winning vector is not the vector the rule selected: "
            f"{summary['best_selection_score']} vs {derived['score']}"
        )

    checkpoint_epoch = None
    path = run_dir / checkpoint
    if path.exists():
        import torch

        # weights_only=False: our own artifact from our own box, and it deliberately
        # carries the run's score dict beside the tensors.
        state = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint_epoch = state.get("epoch")
        if checkpoint_epoch != summary["best_epoch"]:
            problems.append(
                f"{checkpoint} holds epoch {checkpoint_epoch}, the summary claims "
                f"{summary['best_epoch']} — the file is not this run's selected model"
            )
        if state.get("phase") not in (None, "B"):
            problems.append(
                f"{checkpoint} was written in phase {state['phase']}: phase A "
                "checkpoints are never selectable (DESIGN §7.2)"
            )

    return {
        "run": str(run_dir),
        "rule": summary.get("selection_rule"),
        "selected_epoch": summary["best_epoch"],
        "rule_selects_epoch": derived["epoch"],
        "checkpoint_epoch": checkpoint_epoch,
        "winning_score": derived["score"],
        "phase_b_epochs": sum(1 for e in summary["history"] if e["phase"] == "B"),
        "problems": problems,
        "agrees": not problems,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--output", type=Path, help="write the report as JSON")
    args = parser.parse_args()

    report = audit(args.run, args.checkpoint)
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    if not report["agrees"]:
        # Non-zero, because this runs in the C2 chain: a baseline whose checkpoint does
        # not follow its own declared rule must stop the phase, not decorate a log.
        print("\nSELECTION DOES NOT FOLLOW THE RULE")
        return 1
    print(f"\nselection agrees with DESIGN §7.2: epoch {report['selected_epoch']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
