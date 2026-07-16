"""The candidate evidence chain: ORT evaluation → policy → comparison row.

Same philosophy as test_comparison: fabricate the directory the tools would
have left behind, then verify that every number a downstream reader sees is
derived from evidence and that every inconsistency is a refusal, not a warning.
The chain under test is the one M1 ships through: `evaluate_onnx` scores the
artifact itself, `calibrate_candidate` binds a policy to that artifact's hash,
and `comparison --candidate` admits the row only with the parity proof.
"""

from __future__ import annotations

import json
import shutil

import cv2
import numpy as np
import pytest
import torch
import yaml

from wildlife_trigger.optimize import calibrate_candidate as cc
from wildlife_trigger.optimize import evaluate_onnx as eo
from wildlife_trigger import comparison as C
from wildlife_trigger.models.export import export_onnx
from wildlife_trigger.policy import build_class_map, write_canonical_json
from wildlife_trigger.runs import sha256_file

CLASS_NAMES = [
    "opossum", "raccoon", "squirrel", "bobcat", "skunk", "dog", "coyote", "rabbit",
    "bird", "cat", "badger", "empty", "car", "deer", "fox", "rodent",
]
BOBCAT = CLASS_NAMES.index("bobcat")
WIDTH, HEIGHT = 64, 48


class TinyNet(torch.nn.Module):
    def __init__(self, classes: int = 16):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.head = torch.nn.Linear(8, classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv(x))).flatten(1)
        return self.head(x)


def write_validation_world(root):
    """Two validation manifests + JPEGs + the frozen classes config."""
    rng = np.random.default_rng(1)
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    for split, count in (("cis_val_clean", 6), ("trans_val", 4)):
        records = []
        for i in range(count):
            name = f"{split}_{i:03d}.jpg"
            pixels = rng.integers(0, 255, size=(60, 90, 3), dtype=np.uint8)
            assert cv2.imwrite(str(images / name), pixels)
            label = "bobcat" if i % 2 else "empty"
            records.append(
                {
                    "image_id": f"{split}_{i:03d}",
                    "file_name": name,
                    "labels": [label],
                    "primary_label": label,
                    "multi_class": False,
                    "location": "10",
                    "seq_id": f"{split}_seq{i // 2:03d}",
                }
            )
        (root / "manifests").mkdir(exist_ok=True)
        (root / "manifests" / f"{split}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in records)
        )
    classes_config = root / "classes.yaml"
    classes_config.write_text(
        yaml.safe_dump({"classes": [{"index": i, "name": n} for i, n in enumerate(CLASS_NAMES)]})
    )
    return images, root / "manifests", classes_config


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("chain")
    images, manifests, classes_config = write_validation_world(root)

    torch.manual_seed(0)
    model_path = root / "candidate" / "model.onnx"
    model_path.parent.mkdir(parents=True)
    export_onnx(TinyNet(), model_path, torch.randn(1, 3, HEIGHT, WIDTH))

    evaluation = eo.evaluate(
        model_path,
        label="d1_m1_ptq_minmax",
        output_dir=model_path.parent,
        manifests_dir=manifests,
        images_dir=images,
        cache_dir=root / "no-cache",
        classes_config=classes_config,
        intra_op_threads=1,
    )

    candidate = {
        "candidate_id": "d1_m1_ptq_minmax",
        "model_id": "M1-candidate",
        "kind": "int8_ptq",
        "method": "minmax",
        "source_run_id": "c2_m0_fp32_seed42_20260716T061203Z",
        "source_onnx": {"path": "m0.onnx", "sha256": "f" * 64},
        "calibration": {"manifest": "calibration_1024.jsonl", "sha256": "a" * 64, "images": 1024},
        "model": {
            "sha256": evaluation["model"]["sha256"],
            "quantization": {
                "scheme": "S8S8", "format": "QDQ", "per_channel": True,
                "calibration_method": "minmax",
            },
        },
        "integer_execution": True,
    }
    (model_path.parent / "candidate.json").write_text(json.dumps(candidate, indent=2))

    artifacts = root / "artifacts"
    (artifacts / "policies").mkdir(parents=True)
    class_map = build_class_map(CLASS_NAMES)
    write_canonical_json(artifacts / "class_map.json", class_map)

    calibration = cc.calibrate_candidate(
        model_path.parent,
        policy_id="bobcat_m1_ptq_minmax_v1",
        replicates=25,
        artifacts_root=artifacts,
    )

    return {
        "root": root,
        "candidate_dir": model_path.parent,
        "model_path": model_path,
        "evaluation": evaluation,
        "artifacts": artifacts,
        "calibration": calibration,
    }


class TestEvaluateOnnx:
    def test_npz_schema_and_regime_are_recorded(self, world):
        data = np.load(world["candidate_dir"] / "predictions.npz", allow_pickle=False)
        for split, frames in (("cis_val_clean", 6), ("trans_val", 4)):
            assert data[f"{split}/probabilities"].shape == (frames, 16)
            assert data[f"{split}/present"].shape == (frames, 16)
            assert len(data[f"{split}/seq_ids"]) == frames
            assert len(data[f"{split}/image_ids"]) == frames
            # softmax rows sum to 1 — these are probabilities, not logits
            np.testing.assert_allclose(
                data[f"{split}/probabilities"].sum(axis=1), 1.0, atol=1e-5
            )
        assert str(data["model_sha256"]) == sha256_file(world["model_path"])
        assert str(data["provider"]) == "CPUExecutionProvider"

    def test_evaluation_record_carries_the_selection_yardstick(self, world):
        record = world["evaluation"]
        assert record["yardstick_threshold"] == 0.5
        assert record["selection_score"]["primary_metric"] == "mean_bobcat_frame_f2_at_0.5"
        for domain in ("cis_val_clean", "trans_val"):
            target = record["domains"][domain]["target"]
            assert 0.0 <= target["frame_f2"] <= 1.0
            assert "average_precision" in target
        assert record["regime"]["batch_size"] == 1
        assert record["regime"]["input"] == f"{WIDTH}x{HEIGHT}"

    def test_geometry_is_read_from_the_graph(self, world):
        assert eo.model_geometry(world["model_path"]) == (WIDTH, HEIGHT)

    def test_scoring_is_deterministic(self, world, tmp_path):
        second = eo.evaluate(
            world["model_path"],
            label="again",
            output_dir=tmp_path,
            manifests_dir=world["root"] / "manifests",
            images_dir=world["root"] / "images",
            cache_dir=world["root"] / "no-cache",
            classes_config=world["root"] / "classes.yaml",
            intra_op_threads=1,
        )
        first = world["evaluation"]
        assert (
            second["domains"]["cis_val_clean"]["target"]["frame_f2"]
            == first["domains"]["cis_val_clean"]["target"]["frame_f2"]
        )
        a = np.load(world["candidate_dir"] / "predictions.npz")
        b = np.load(tmp_path / "predictions.npz")
        np.testing.assert_array_equal(
            a["cis_val_clean/probabilities"], b["cis_val_clean/probabilities"]
        )


class TestCalibrateCandidate:
    def test_policy_binds_directly_to_the_scored_artifact(self, world):
        result = world["calibration"]
        policy_path = world["artifacts"] / "policies" / "bobcat_m1_ptq_minmax_v1.json"
        policy = json.loads(policy_path.read_text())
        assert policy["model_sha256"] == sha256_file(world["model_path"])
        assert policy["policy_id"] == "bobcat_m1_ptq_minmax_v1"
        assert policy["calibration"]["run_id"] == "d1_m1_ptq_minmax"
        assert policy["calibration"]["status"] == result["selection"]["status"]
        # The §6.3 verdict travels verbatim inside the policy — a reader of the
        # artifact cannot quote the threshold without its status.
        assert "primary_rule_met" in policy["calibration"]
        assert (world["candidate_dir"] / "calibration.json").exists()

    def test_moved_model_bytes_are_refused(self, world, tmp_path):
        clone = tmp_path / "cand"
        shutil.copytree(world["candidate_dir"], clone)
        # evaluation.json points at the original path; move the bytes there
        evaluation = json.loads((clone / "evaluation.json").read_text())
        evaluation["model"]["path"] = str(clone / "model.onnx")
        (clone / "evaluation.json").write_text(json.dumps(evaluation))
        (clone / "model.onnx").write_bytes(b"different bytes now")
        with pytest.raises(RuntimeError, match="calibration would not describe"):
            cc.calibrate_candidate(
                clone, policy_id="x_v1", replicates=5, artifacts_root=world["artifacts"]
            )

    def test_class_order_mismatch_is_refused(self, world, tmp_path):
        artifacts = tmp_path / "artifacts"
        (artifacts / "policies").mkdir(parents=True)
        reordered = build_class_map(list(reversed(CLASS_NAMES)))
        write_canonical_json(artifacts / "class_map.json", reordered)
        with pytest.raises(RuntimeError, match="frozen class map"):
            cc.calibrate_candidate(
                world["candidate_dir"],
                policy_id="bobcat_x_v1",
                replicates=5,
                artifacts_root=artifacts,
            )

    def test_no_threshold_targets_are_refused(self, world):
        with pytest.raises(ValueError, match="catalog"):
            cc.calibrate_candidate(
                world["candidate_dir"], policy_id="badger_v1", target="badger"
            )


class TestComparisonCandidateMode:
    @pytest.fixture()
    def table(self, world, tmp_path):
        table = tmp_path / "comparison.jsonl"
        m0 = {
            "model_id": "M0", "kind": "fp32_baseline", "seed": 42,
            "params": 2244368, "macs": 293402624,
            "model": {"bytes": 8950645},
            "validation_at_0p5": {"cis_f2": 0.6, "trans_f2": 0.1},
        }
        table.write_text(json.dumps(m0) + "\n")
        return table

    def make_parity(self, world, passed=True):
        parity = world["candidate_dir"] / "p3_quantized.json"
        parity.write_text(json.dumps({
            "onnx": {"sha256": sha256_file(world["model_path"])},
            "verdict": {"passed": passed},
        }))
        return parity

    def test_row_is_derived_from_candidate_evidence(self, world, table):
        parity = self.make_parity(world)
        policy_path = world["artifacts"] / "policies" / "bobcat_m1_ptq_minmax_v1.json"
        candidate, evaluation, policy = C.load_candidate_row_inputs(
            world["candidate_dir"], policy_path
        )
        artifact, onnx_sha = C.verify_artifact(policy)
        C.verify_parity(parity, onnx_sha)
        base = C.base_row(table, "M0")
        row = C.build_candidate_row(
            "M1", "int8_ptq", candidate, evaluation, policy, policy_path,
            artifact, onnx_sha, parity, base,
        )
        assert row["params"] == 2244368 and row["macs"] == 293402624
        assert row["seed"] == 42
        assert row["run_id"] == "d1_m1_ptq_minmax"
        assert row["quantization"]["method"] == "minmax"
        assert row["model"]["sha256"] == sha256_file(world["model_path"])
        assert (
            row["validation_at_0p5"]["cis_f2"]
            == evaluation["domains"]["cis_val_clean"]["target"]["frame_f2"]
        )
        rows = C.update_table(table, row)
        assert [r["model_id"] for r in rows] == ["M0", "M1"]

    def test_missing_base_row_is_refused(self, world, tmp_path):
        empty = tmp_path / "empty.jsonl"
        with pytest.raises(RuntimeError, match="must be written before"):
            C.base_row(empty, "M0")
        empty.write_text(json.dumps({"model_id": "M9"}) + "\n")
        with pytest.raises(RuntimeError, match="cannot .* invent|has no M0"):
            C.base_row(empty, "M0")

    def test_failed_parity_is_refused(self, world):
        parity = self.make_parity(world, passed=False)
        with pytest.raises(RuntimeError, match="did not pass"):
            C.verify_parity(parity, sha256_file(world["model_path"]))

    def test_policy_for_another_candidate_is_refused(self, world, tmp_path):
        policy_path = world["artifacts"] / "policies" / "bobcat_m1_ptq_minmax_v1.json"
        policy = json.loads(policy_path.read_text())
        policy["calibration"]["run_id"] = "d1_m1_ptq_entropy"
        other = tmp_path / "other_policy.json"
        other.write_text(json.dumps(policy))
        with pytest.raises(RuntimeError, match="does not exist$|calibrated for"):
            C.load_candidate_row_inputs(world["candidate_dir"], other)
