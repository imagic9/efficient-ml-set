"""The C3 calibration tool, end to end on fabricated run directories (PLAN C3).

Each test builds a run directory the way `train.py` + `validate.dump_predictions`
would have left it — history, hashes, predictions.npz — and drives
`wildlife_trigger.calibrate` over it. The scenarios map one-to-one onto DESIGN
§6.3's three registered statuses, and the assertions that matter most are the
honesty ones: the failure statuses must survive, verbatim, into every artifact
that could be quoted without the rest of the record.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from wildlife_trigger import calibrate as C
from wildlife_trigger import metrics
from wildlife_trigger.policy import validate_policy

CLASS_NAMES = [
    "opossum", "raccoon", "squirrel", "bobcat", "skunk", "dog", "coyote", "rabbit",
    "bird", "cat", "badger", "empty", "car", "deer", "fox", "rodent",
]
BOBCAT = CLASS_NAMES.index("bobcat")
CHECKPOINT_HASH = "c" * 64


def domain_arrays(scores, present, seqs):
    """Full 16-class arrays with the bobcat column carrying the scenario."""
    frames = len(scores)
    probabilities = np.zeros((frames, len(CLASS_NAMES)), dtype=np.float32)
    probabilities[:, BOBCAT] = scores
    present_matrix = np.zeros((frames, len(CLASS_NAMES)), dtype=np.float32)
    present_matrix[:, BOBCAT] = present
    return probabilities, present_matrix, np.array(seqs)


def write_run(tmp_path, cis, trans, best_epoch=11):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "history.json").write_text(
        json.dumps(
            {"run_name": "test_run", "best_epoch": best_epoch, "class_names": CLASS_NAMES}
        )
    )
    (run_dir / "hashes.json").write_text(
        json.dumps(
            {
                "manifest:cis_val_clean": {"path": "m/cis.jsonl", "sha256": "1" * 64},
                "manifest:trans_val": {"path": "m/trans.jsonl", "sha256": "2" * 64},
                "checkpoint:best": {"path": "run/best.pt", "sha256": CHECKPOINT_HASH},
            }
        )
    )
    payload = {
        "run_name": "test_run",
        "class_names": np.array(CLASS_NAMES),
        "best_epoch": best_epoch,
    }
    for name, (scores, present, seqs) in {"cis_val_clean": cis, "trans_val": trans}.items():
        probabilities, present_matrix, seq_array = domain_arrays(scores, present, seqs)
        payload[f"{name}/probabilities"] = probabilities
        payload[f"{name}/present"] = present_matrix
        payload[f"{name}/seq_ids"] = seq_array
        payload[f"{name}/image_ids"] = seq_array
    np.savez_compressed(run_dir / "predictions.npz", **payload)
    return run_dir


def clean_domain():
    """Separable: every visit caught, no negative above 0.2 -> primary rule met."""
    scores = [0.95, 0.93, 0.91] * 4 + [0.05] * 60
    present = [1.0] * 12 + [0.0] * 60
    seqs = [f"visit{i // 3}" for i in range(12)] + [f"neg{i // 3}" for i in range(60)]
    return scores, present, seqs


def constrained_domain():
    """The expected C3 shape: one separable visit, one below the negatives."""
    scores = [0.90, 0.85, 0.02] + [0.10] * 40
    present = [1.0, 1.0, 1.0] + [0.0] * 40
    seqs = ["visit_a", "visit_a", "visit_b"] + [f"neg{i}" for i in range(40)]
    return scores, present, seqs


def hopeless_domain():
    """Every negative outscores every positive -> nothing is admissible."""
    scores = [0.30, 0.20] + [0.90] * 30
    present = [1.0, 1.0] + [0.0] * 30
    seqs = ["a", "b"] + [f"n{i}" for i in range(30)]
    return scores, present, seqs


def run_calibrate(tmp_path, run_dir, **kwargs):
    return C.calibrate(
        run_dir,
        replicates=kwargs.pop("replicates", 30),
        output_root=tmp_path / "results",
        artifacts_root=tmp_path / "artifacts",
        **kwargs,
    )


class TestPrimaryRuleMet:
    def test_policy_is_written_and_loadable(self, tmp_path) -> None:
        run_dir = write_run(tmp_path, clean_domain(), clean_domain())
        result = run_calibrate(tmp_path, run_dir)

        assert result["selection"]["status"] == "primary_rule_met"
        policy_path = tmp_path / "artifacts" / "policies" / "bobcat_v1.json"
        policy = json.loads(policy_path.read_text())
        validate_policy(
            policy,
            json.loads((tmp_path / "artifacts" / "class_map.json").read_text()),
            model_sha256=CHECKPOINT_HASH,
        )
        assert policy["targets"] == [
            {"class": "bobcat", "threshold": result["selection"]["threshold"]}
        ]

    def test_class_map_binding_is_to_the_bytes_on_disk(self, tmp_path) -> None:
        """The C++ loader hashes the file it reads; the policy must bind to exactly
        that, not to a re-serialisation that happens to look the same."""
        run_dir = write_run(tmp_path, clean_domain(), clean_domain())
        run_calibrate(tmp_path, run_dir)

        policy = json.loads(
            (tmp_path / "artifacts" / "policies" / "bobcat_v1.json").read_text()
        )
        on_disk = (tmp_path / "artifacts" / "class_map.json").read_bytes()
        assert policy["class_map_sha256"] == hashlib.sha256(on_disk).hexdigest()

    def test_class_map_keeps_the_training_order(self, tmp_path) -> None:
        run_dir = write_run(tmp_path, clean_domain(), clean_domain())
        run_calibrate(tmp_path, run_dir)
        class_map = json.loads((tmp_path / "artifacts" / "class_map.json").read_text())
        assert class_map["classes"] == CLASS_NAMES


class TestRecallFloorInfeasible:
    def test_the_failure_travels_into_the_policy_itself(self, tmp_path) -> None:
        """`recall_floor_infeasible` ships an operating point and is not a pass.
        The policy artifact is the thing most likely to be read in isolation, so
        the verdict must be inside it — quotable only with the failure attached."""
        run_dir = write_run(tmp_path, constrained_domain(), constrained_domain())
        result = run_calibrate(tmp_path, run_dir)

        assert result["selection"]["status"] == "recall_floor_infeasible"
        policy = json.loads(
            (tmp_path / "artifacts" / "policies" / "bobcat_v1.json").read_text()
        )
        assert policy["calibration"]["status"] == "recall_floor_infeasible"
        assert policy["calibration"]["primary_rule_met"] is False
        assert "NOT satisfied" in policy["calibration"]["rule"]
        assert policy["calibration"]["unmet_constraint"], (
            "the recall each domain actually reached is part of the verdict"
        )

    def test_the_record_carries_curve_bootstrap_strata_and_histograms(
        self, tmp_path
    ) -> None:
        run_dir = write_run(tmp_path, constrained_domain(), constrained_domain())
        result = run_calibrate(tmp_path, run_dir)

        record = json.loads(
            (tmp_path / "results" / "test_run" / "calibration.json").read_text()
        )
        assert record["selection"]["recall_false_fire_curve"], "§6.3 step 5"
        assert len(record["threshold_bootstrap"]["thresholds"]) == 30, "§6.3 step 7"
        assert record["length_strata"]["cis_val_clean"]["strata"]["1-2"]["supported"]
        histogram = record["score_histograms"]["cis_val_clean"]
        frames = len(constrained_domain()[0])
        assert sum(histogram["positive"]) + sum(histogram["negative"]) == frames
        assert record["selection"]["threshold"] == result["selection"]["threshold"]


class TestFireBudgetInfeasible:
    def test_no_policy_is_written(self, tmp_path) -> None:
        """§6.3 step 4: no operating point exists, so naming one — even a clearly
        labelled one — would be inventing it. The record still exists."""
        run_dir = write_run(tmp_path, hopeless_domain(), hopeless_domain())
        result = run_calibrate(tmp_path, run_dir)

        assert result["selection"]["status"] == "fire_budget_infeasible"
        assert not (tmp_path / "artifacts" / "policies" / "bobcat_v1.json").exists()
        assert not (tmp_path / "artifacts" / "class_map.json").exists()
        assert (tmp_path / "results" / "test_run" / "calibration.json").exists()


class TestGuards:
    def test_class_order_mismatch_is_refused(self, tmp_path) -> None:
        """predictions.npz from one run, history from another: the bobcat column
        may be a different animal. Refuse, never reconcile."""
        run_dir = write_run(tmp_path, clean_domain(), clean_domain())
        history = json.loads((run_dir / "history.json").read_text())
        history["class_names"] = sorted(CLASS_NAMES)
        (run_dir / "history.json").write_text(json.dumps(history))
        with pytest.raises(RuntimeError, match="class order"):
            run_calibrate(tmp_path, run_dir)

    def test_epoch_mismatch_is_refused(self, tmp_path) -> None:
        run_dir = write_run(tmp_path, clean_domain(), clean_domain())
        history = json.loads((run_dir / "history.json").read_text())
        history["best_epoch"] = 7
        (run_dir / "history.json").write_text(json.dumps(history))
        with pytest.raises(RuntimeError, match="epoch"):
            run_calibrate(tmp_path, run_dir)

    def test_fast_path_drift_refuses_to_calibrate(self, tmp_path, monkeypatch) -> None:
        """Two implementations of one registered rule: if they disagree, neither
        number ships. The tool must stop, not pick its favourite."""
        run_dir = write_run(tmp_path, clean_domain(), clean_domain())
        monkeypatch.setattr(
            metrics, "select_threshold_point", lambda *a, **k: (0.123, "primary_rule_met")
        )
        with pytest.raises(RuntimeError, match="drifted"):
            run_calibrate(tmp_path, run_dir)

    def test_missing_hashes_are_refused(self, tmp_path) -> None:
        run_dir = write_run(tmp_path, clean_domain(), clean_domain())
        (run_dir / "hashes.json").write_text(json.dumps({}))
        with pytest.raises(RuntimeError, match="checkpoint:best"):
            run_calibrate(tmp_path, run_dir)

    def test_unknown_target_is_refused(self, tmp_path) -> None:
        run_dir = write_run(tmp_path, clean_domain(), clean_domain())
        with pytest.raises(ValueError, match="unicorn"):
            run_calibrate(tmp_path, run_dir, target="unicorn")

    def test_non_animal_target_is_refused(self, tmp_path) -> None:
        run_dir = write_run(tmp_path, clean_domain(), clean_domain())
        with pytest.raises(ValueError, match="empty"):
            run_calibrate(tmp_path, run_dir, target="empty")

    def test_no_threshold_catalog_target_is_refused(self, tmp_path) -> None:
        """A badger policy must be impossible to generate end-to-end, not just
        impossible to load: DESIGN §4 gives badger no operating point."""
        run_dir = write_run(tmp_path, clean_domain(), clean_domain())
        with pytest.raises(ValueError, match="badger"):
            run_calibrate(tmp_path, run_dir, target="badger")


class TestProvenance:
    def test_the_record_names_everything_the_numbers_depend_on(self, tmp_path) -> None:
        run_dir = write_run(tmp_path, clean_domain(), clean_domain())
        run_calibrate(tmp_path, run_dir)
        record = json.loads(
            (tmp_path / "results" / "test_run" / "calibration.json").read_text()
        )
        inputs = record["inputs"]
        assert inputs["checkpoint:best"]["sha256"] == CHECKPOINT_HASH
        assert inputs["manifest:cis_val_clean"]["sha256"] == "1" * 64
        assert inputs["manifest:trans_val"]["sha256"] == "2" * 64
        assert len(inputs["predictions_npz_sha256"]) == 64

    def test_the_policy_names_its_datasets_and_record(self, tmp_path) -> None:
        run_dir = write_run(tmp_path, clean_domain(), clean_domain())
        run_calibrate(tmp_path, run_dir)
        policy = json.loads(
            (tmp_path / "artifacts" / "policies" / "bobcat_v1.json").read_text()
        )
        calibration = policy["calibration"]
        assert calibration["datasets"]["manifest:cis_val_clean"] == "1" * 64
        assert calibration["run_id"] == "test_run"
        assert "calibration.json" in calibration["record"]
        assert policy["model"]["kind"] == "pytorch_checkpoint"
