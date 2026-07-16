"""The comparison table tool: a row is derived from evidence or not written at all.

Each test fabricates a run directory the way `train.py` + `calibrate` + `export`
would have left it — real checkpoint, real hashes, a policy bound to the artifact —
and drives `wildlife_trigger.comparison` over it. The refusal tests matter most:
the table feeds D6's shortlist and the report's headline numbers, so a row that
mixes models, describes a re-exported file, or admits a failed gate would poison
every decision downstream of it.
"""

from __future__ import annotations

import json

import pytest
import torch

from wildlife_trigger import comparison as C
from wildlife_trigger.models.mobilenet import build_mobilenet_v2
from wildlife_trigger.runs import sha256_file

CLASS_NAMES = [
    "opossum", "raccoon", "squirrel", "bobcat", "skunk", "dog", "coyote", "rabbit",
    "bird", "cat", "badger", "empty", "car", "deer", "fox", "rodent",
]


def domain_metrics(f2: float) -> dict:
    return {
        "threshold": 0.5,
        "frame_f2": f2,
        "frame_recall": 0.5,
        "frame_precision": 0.5,
        "sequence_balanced_recall": 0.5,
        "event_capture_rate": 0.5,
        "false_fire_rate": 0.03,
        "fire_rate": 0.05,
    }


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """A run directory + policy + parity report that agree with each other."""
    root = tmp_path_factory.mktemp("comparison")
    run_dir = root / "c5_m0_fp32_seed17_20260716T000000Z"
    run_dir.mkdir()

    torch.manual_seed(0)
    model = build_mobilenet_v2(num_classes=len(CLASS_NAMES), pretrained=False)
    torch.save({"model": model.state_dict(), "epoch": 3}, run_dir / "best.pt")

    onnx = root / "model.onnx"
    onnx.write_bytes(b"not-a-real-graph, only its bytes are compared")

    history = {
        "run_name": "m0_fp32_seed17",
        "best_epoch": 3,
        "class_names": CLASS_NAMES,
        "config": {"seed": 17, "width": 64, "height": 64, "batch_size": 4},
        "history": [
            {
                "epoch": epoch,
                "phase": "B",
                "selection_score": {"primary": 0.1 * epoch},
                "cis_val_clean": domain_metrics(0.1 * epoch),
                "trans_val": {**domain_metrics(0.05 * epoch), "bobcat_ap": 0.4},
            }
            for epoch in range(5)
        ],
    }
    (run_dir / "history.json").write_text(json.dumps(history))
    (run_dir / "hashes.json").write_text(json.dumps({
        "checkpoint:best": {"sha256": sha256_file(run_dir / "best.pt")},
    }))

    parity_path = root / "p2_fp32.json"
    parity_path.write_text(json.dumps({
        "onnx": {"sha256": sha256_file(onnx)},
        "verdict": {"passed": True, "failed_fixtures": []},
    }))

    policy_path = root / "bobcat_v1.json"
    policy_path.write_text(json.dumps({
        "policy_id": "bobcat_v1",
        "model_sha256": sha256_file(onnx),
        "model": {"artifact": str(onnx), "parity": str(parity_path)},
        "targets": [{"class": "bobcat", "threshold": 0.538088}],
        "calibration": {
            "run_id": run_dir.name,
            "status": "recall_floor_infeasible",
            "primary_rule_met": False,
            "per_domain": {
                "cis_val_clean": domain_metrics(0.63),
                "trans_val": domain_metrics(0.10),
            },
        },
    }))

    return {
        "root": root,
        "run_dir": run_dir,
        "onnx": onnx,
        "policy_path": policy_path,
        "parity_path": parity_path,
    }


def write_row(world, table, model_id="M0", kind="fp32_baseline"):
    history, policy = C.load_row_inputs(world["run_dir"], world["policy_path"])
    artifact, onnx_sha = C.verify_artifact(policy)
    C.verify_parity(world["parity_path"], onnx_sha)
    params = C.count_parameters(world["run_dir"], history)
    row = C.build_row(
        model_id, kind, world["run_dir"], history, policy, world["policy_path"],
        artifact, onnx_sha, world["parity_path"], params, macs=1_000_000,
    )
    return C.update_table(table, row), row


class TestRow:
    def test_row_is_derived_from_the_evidence(self, world, tmp_path) -> None:
        rows, row = write_row(world, tmp_path / "comparison.jsonl")

        assert row["seed"] == 17 and row["best_epoch"] == 3
        assert row["validation_at_0p5"]["cis_f2"] == pytest.approx(0.3)
        assert row["validation_at_0p5"]["trans_f2"] == pytest.approx(0.15)
        # AP is present only where the run recorded it (post-#19 runs; trans here)
        assert row["validation_at_0p5"]["trans_ap"] == 0.4
        assert "cis_ap" not in row["validation_at_0p5"]
        assert row["operating_point"]["threshold"] == 0.538088
        assert row["operating_point"]["status"] == "recall_floor_infeasible"
        assert row["operating_point"]["primary_rule_met"] is False
        assert row["model"]["bytes"] == world["onnx"].stat().st_size

        expected = sum(
            p.numel()
            for p in build_mobilenet_v2(
                num_classes=len(CLASS_NAMES), pretrained=False
            ).parameters()
        )
        assert row["params"] == expected

    def test_rewriting_a_row_replaces_it(self, world, tmp_path) -> None:
        """The table is one row per candidate, however many times its evidence is
        regenerated — a duplicate M0 would double-count the baseline in D6."""
        table = tmp_path / "comparison.jsonl"
        write_row(world, table)
        rows, _ = write_row(world, table)
        assert len(rows) == 1
        lines = table.read_text().splitlines()
        assert len(lines) == 1

    def test_other_rows_survive_and_order_is_stable(self, world, tmp_path) -> None:
        table = tmp_path / "comparison.jsonl"
        write_row(world, table, model_id="M1", kind="int8_ptq")
        rows, _ = write_row(world, table, model_id="M0")
        assert [r["model_id"] for r in rows] == ["M0", "M1"]


class TestRefusals:
    def test_a_policy_for_another_run_is_refused(self, world, tmp_path) -> None:
        foreign = tmp_path / "foreign_policy.json"
        policy = json.loads(world["policy_path"].read_text())
        policy["calibration"]["run_id"] = "c2_someone_else_20260101T000000Z"
        foreign.write_text(json.dumps(policy))
        with pytest.raises(RuntimeError, match="device that does not exist"):
            C.load_row_inputs(world["run_dir"], foreign)

    def test_a_reexported_artifact_is_refused(self, world, tmp_path) -> None:
        """The row must describe the file the policy was calibrated against, not
        whatever a later export left at the same path."""
        policy = json.loads(world["policy_path"].read_text())
        drifted = tmp_path / "drifted.onnx"
        drifted.write_bytes(b"different bytes entirely")
        policy["model"]["artifact"] = str(drifted)
        with pytest.raises(RuntimeError, match="not the model the policy"):
            C.verify_artifact(policy)

    def test_a_failed_parity_report_is_refused(self, world, tmp_path) -> None:
        failed = tmp_path / "failed_p2.json"
        report = json.loads(world["parity_path"].read_text())
        report["verdict"]["passed"] = False
        failed.write_text(json.dumps(report))
        with pytest.raises(RuntimeError, match="did not pass"):
            C.verify_parity(failed, sha256_file(world["onnx"]))

    def test_a_parity_report_about_another_artifact_is_refused(self, world) -> None:
        with pytest.raises(RuntimeError, match="different artifact"):
            C.verify_parity(world["parity_path"], "0" * 64)

    def test_a_tampered_checkpoint_is_refused(self, world, tmp_path) -> None:
        run_dir = tmp_path / "tampered_run"
        run_dir.mkdir()
        for name in ("history.json", "best.pt"):
            (run_dir / name).write_bytes((world["run_dir"] / name).read_bytes())
        (run_dir / "hashes.json").write_text(json.dumps({
            "checkpoint:best": {"sha256": "0" * 64},
        }))
        history = json.loads((run_dir / "history.json").read_text())
        with pytest.raises(RuntimeError, match="does not hash to the run's record"):
            C.count_parameters(run_dir, history)
