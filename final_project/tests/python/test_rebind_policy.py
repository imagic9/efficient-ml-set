"""The policy re-bind: the only sanctioned move of `model_sha256` (PLAN C4).

C3 bound the policy to the checkpoint so it would fail loudly against any ONNX
until the parity proof existed. Every test here is about the refusals — a
re-bind that can be talked into moving without the proof would undo that design
— plus the one happy path where proof, weights and file all line up.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from wildlife_trigger import rebind_policy as R
from wildlife_trigger.policy import build_class_map, build_policy, write_canonical_json

CLASS_NAMES = [
    "opossum", "raccoon", "squirrel", "bobcat", "skunk", "dog", "coyote", "rabbit",
    "bird", "cat", "badger", "empty", "car", "deer", "fox", "rodent",
]
CHECKPOINT = "c" * 64


@pytest.fixture()
def world(tmp_path):
    """Policy bound to a checkpoint, a dummy ONNX, and a passing P2 report."""
    class_map = build_class_map(CLASS_NAMES)
    class_map_path = tmp_path / "class_map.json"
    class_map_sha256 = write_canonical_json(class_map_path, class_map)

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx-bytes-standing-in-for-a-graph")
    onnx_sha256 = hashlib.sha256(onnx_path.read_bytes()).hexdigest()

    policy = build_policy(
        policy_id="bobcat_v1",
        targets=[{"class": "bobcat", "threshold": 0.5381}],
        class_map=class_map,
        class_map_sha256=class_map_sha256,
        model_sha256=CHECKPOINT,
        metadata={
            "model": {"kind": "pytorch_checkpoint", "artifact": "run/best.pt"},
            "calibration": {"status": "recall_floor_infeasible", "run_id": "r1"},
        },
    )
    policy_path = tmp_path / "bobcat_v1.json"
    write_canonical_json(policy_path, policy)

    report_path = tmp_path / "p2_fp32.json"
    report_path.write_text(json.dumps({
        "verdict": {"passed": True},
        "checkpoint_sha256": CHECKPOINT,
        "onnx": {"path": str(onnx_path), "sha256": onnx_sha256},
    }))

    return {
        "policy_path": policy_path,
        "class_map_path": class_map_path,
        "onnx_path": onnx_path,
        "onnx_sha256": onnx_sha256,
        "report_path": report_path,
    }


class TestHappyPath:
    def test_rebinds_and_keeps_the_history_inside_the_artifact(self, world) -> None:
        R.rebind(world["policy_path"], world["class_map_path"],
                 world["onnx_path"], world["report_path"])
        policy = json.loads(world["policy_path"].read_text())

        assert policy["model_sha256"] == world["onnx_sha256"]
        assert policy["model"]["kind"] == "onnx"
        assert policy["model"]["previous"]["model_sha256"] == CHECKPOINT
        assert policy["model"]["previous"]["kind"] == "pytorch_checkpoint"
        assert "p2_fp32" in policy["model"]["parity"]

    def test_the_calibration_block_is_untouched(self, world) -> None:
        """The calibration describes an event that happened — datasets, verdict,
        thresholds. Moving the binding does not rewrite history."""
        before = json.loads(world["policy_path"].read_text())["calibration"]
        R.rebind(world["policy_path"], world["class_map_path"],
                 world["onnx_path"], world["report_path"])
        after = json.loads(world["policy_path"].read_text())["calibration"]
        assert after == before

    def test_thresholds_are_untouched(self, world) -> None:
        before = json.loads(world["policy_path"].read_text())["targets"]
        R.rebind(world["policy_path"], world["class_map_path"],
                 world["onnx_path"], world["report_path"])
        after = json.loads(world["policy_path"].read_text())["targets"]
        assert after == before


class TestRefusals:
    def test_no_report_no_rebind(self, world) -> None:
        world["report_path"].unlink()
        with pytest.raises(RuntimeError, match="No P2 report, no re-bind"):
            R.rebind(world["policy_path"], world["class_map_path"],
                     world["onnx_path"], world["report_path"])

    def test_a_failed_verdict_rebinds_nothing(self, world) -> None:
        report = json.loads(world["report_path"].read_text())
        report["verdict"]["passed"] = False
        world["report_path"].write_text(json.dumps(report))
        with pytest.raises(RuntimeError, match="verdict"):
            R.rebind(world["policy_path"], world["class_map_path"],
                     world["onnx_path"], world["report_path"])

    def test_proof_about_other_weights_is_refused(self, world) -> None:
        """A P2 report for a different checkpoint proves nothing about the
        weights this policy's thresholds were measured on."""
        report = json.loads(world["report_path"].read_text())
        report["checkpoint_sha256"] = "d" * 64
        world["report_path"].write_text(json.dumps(report))
        with pytest.raises(RuntimeError, match="different weights"):
            R.rebind(world["policy_path"], world["class_map_path"],
                     world["onnx_path"], world["report_path"])

    def test_an_onnx_changed_since_the_proof_is_refused(self, world) -> None:
        """The file being bound must be the file that passed — re-hashed now,
        not trusted from the report."""
        world["onnx_path"].write_bytes(b"different graph bytes")
        with pytest.raises(RuntimeError, match="not the file that was proven"):
            R.rebind(world["policy_path"], world["class_map_path"],
                     world["onnx_path"], world["report_path"])

    def test_nothing_is_written_when_refused(self, world) -> None:
        before = world["policy_path"].read_bytes()
        world["onnx_path"].write_bytes(b"different graph bytes")
        with pytest.raises(RuntimeError):
            R.rebind(world["policy_path"], world["class_map_path"],
                     world["onnx_path"], world["report_path"])
        assert world["policy_path"].read_bytes() == before
