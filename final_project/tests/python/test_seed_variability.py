"""C5's variability instrument: aggregate only runs that are the same experiment.

The refusals carry the scientific weight: issue #18 is answered by the spread of
the frozen recipe across seeds, and a run that quietly trained on different data,
a different budget, or the same seed twice would make that spread meaningless
while still producing a plausible-looking mean±std.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wildlife_trigger.validate import seed_variability as V

CLASS_NAMES = ["bobcat", "empty"]

RECIPE = {
    "width": 64, "height": 64, "head_epochs": 1, "max_epochs": 3, "batch_size": 4,
    "head_lr": 1e-3, "full_lr": 3e-4, "weight_decay": 1e-4,
    "early_stopping_patience": 6, "amp": True, "exclude_empty_class": False,
}


def make_run(
    root: Path,
    seed: int,
    trans_f2_by_epoch: list[float],
    best_epoch: int,
    manifests: dict | None = None,
    **recipe_overrides,
) -> Path:
    run_dir = root / f"c5_m0_fp32_seed{seed}_20260716T000000Z"
    run_dir.mkdir()

    def entry(epoch: int, trans_f2: float) -> dict:
        return {
            "epoch": epoch,
            "phase": "A" if epoch == 0 else "B",
            "selection_score": {"primary": (0.6 + trans_f2) / 2},
            "cis_val_clean": {
                "frame_f2": 0.6,
                "sequence_balanced_recall": 0.7,
                "event_capture_rate": 0.8,
            },
            "trans_val": {
                "frame_f2": trans_f2,
                "sequence_balanced_recall": trans_f2,
                "event_capture_rate": trans_f2,
            },
        }

    history = {
        "run_name": f"m0_fp32_seed{seed}",
        "best_epoch": best_epoch,
        "class_names": CLASS_NAMES,
        "config": {**RECIPE, **recipe_overrides, "seed": seed},
        "history": [entry(i, f2) for i, f2 in enumerate(trans_f2_by_epoch)],
    }
    (run_dir / "history.json").write_text(json.dumps(history))
    (run_dir / "hashes.json").write_text(json.dumps(
        manifests
        or {
            "manifest:cis_val_clean": {"sha256": "a" * 64},
            "manifest:trans_val": {"sha256": "b" * 64},
            "checkpoint:best": {"sha256": "c" * 64},
        }
    ))
    return run_dir


class TestAggregation:
    def test_means_and_spread_are_the_runs_own_numbers(self, tmp_path) -> None:
        runs = V.load_runs([
            make_run(tmp_path, 17, [0.0, 0.10, 0.20], best_epoch=2),
            make_run(tmp_path, 42, [0.0, 0.10, 0.10], best_epoch=1),
            make_run(tmp_path, 73, [0.0, 0.40, 0.30], best_epoch=1),
        ])
        V.verify_same_experiment(runs)
        summaries = sorted(
            (V.summarise_run(run) for run in runs), key=lambda s: s["seed"]
        )
        aggregate = V.aggregate(summaries)

        assert [s["seed"] for s in summaries] == [17, 42, 73]
        # at the selected checkpoints: 0.20, 0.10, 0.40
        assert aggregate["trans_val"]["frame_f2"]["mean"] == pytest.approx(0.7 / 3)
        assert aggregate["trans_val"]["frame_f2"]["values"] == [0.20, 0.10, 0.40]
        # the trajectory max is phase-B only and per seed: 0.20, 0.10, 0.40
        assert aggregate["trans_f2_max_over_epochs"]["values"] == [0.20, 0.10, 0.40]
        # epoch 0 is phase A and must not enter the trajectory
        assert summaries[0]["trans_f2_trajectory"]["phase_b_epochs"] == 2

    def test_ap_travels_only_where_recorded(self, tmp_path) -> None:
        """The baseline predates per-epoch AP (issue #19); its summary must not
        invent the field."""
        run_dir = make_run(tmp_path, 42, [0.0, 0.1], best_epoch=1)
        history = json.loads((run_dir / "history.json").read_text())
        history["history"][1]["trans_val"]["average_precision"] = 0.5
        (run_dir / "history.json").write_text(json.dumps(history))

        (summary,) = [V.summarise_run(run) for run in V.load_runs([run_dir])]
        assert summary["at_best"]["trans_val"]["average_precision"] == 0.5
        assert "average_precision" not in summary["at_best"]["cis_val_clean"]


class TestRefusals:
    def test_a_different_recipe_is_a_different_experiment(self, tmp_path) -> None:
        runs = V.load_runs([
            make_run(tmp_path, 17, [0.0, 0.1], best_epoch=1),
            make_run(tmp_path, 42, [0.0, 0.1], best_epoch=1, max_epochs=30),
        ])
        with pytest.raises(RuntimeError, match="different experiments"):
            V.verify_same_experiment(runs)

    def test_duplicate_seeds_are_refused(self, tmp_path) -> None:
        a = make_run(tmp_path, 42, [0.0, 0.1], best_epoch=1)
        b = tmp_path / "copy"
        b.mkdir()
        for name in ("history.json", "hashes.json"):
            (b / name).write_text((a / name).read_text())
        with pytest.raises(RuntimeError, match="duplicate seeds"):
            V.verify_same_experiment(V.load_runs([a, b]))

    def test_different_data_is_refused(self, tmp_path) -> None:
        runs = V.load_runs([
            make_run(tmp_path, 17, [0.0, 0.1], best_epoch=1),
            make_run(tmp_path, 42, [0.0, 0.1], best_epoch=1, manifests={
                "manifest:cis_val_clean": {"sha256": "d" * 64},
                "manifest:trans_val": {"sha256": "b" * 64},
            }),
        ])
        with pytest.raises(RuntimeError, match="different dataset manifests"):
            V.verify_same_experiment(runs)

    def test_one_run_is_not_variability(self, tmp_path) -> None:
        runs = V.load_runs([make_run(tmp_path, 42, [0.0, 0.1], best_epoch=1)])
        with pytest.raises(RuntimeError, match="at least two"):
            V.verify_same_experiment(runs)
