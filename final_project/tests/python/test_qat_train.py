"""The D2 QAT trainer: initialization proofs, recipe rails, the candidate contract.

The full end-to-end (real MobileNetV2, real JPEGs, one epoch) runs in
`TestEndToEnd` and is the expensive part of this file (~1 min on gx10): what it
buys is the whole chain — hash-verified init from a checkpoint, real-data
observer calibration, a training step that moves weights, §7.2 epoch selection,
and an exported candidate directory that `evaluate_onnx`/`select_ptq` can
consume unchanged. The cheap tests around it pin the refusals.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
import torch
import yaml

from wildlife_trigger.models.mobilenet import build_mobilenet_v2
from wildlife_trigger.optimize import qat_train as qt
from wildlife_trigger.runs import sha256_file

CLASS_NAMES = [
    "opossum", "raccoon", "squirrel", "bobcat", "skunk", "dog", "coyote", "rabbit",
    "bird", "cat", "badger", "empty", "car", "deer", "fox", "rodent",
]
WIDTH, HEIGHT = 64, 48


def write_images(root, prefix: str, count: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(count):
        name = f"{prefix}{i:03d}.jpg"
        pixels = rng.integers(0, 255, size=(72, 96, 3), dtype=np.uint8)
        assert cv2.imwrite(str(root / name), pixels)
        label = "bobcat" if i % 2 else "empty"
        records.append({
            "image_id": f"{prefix}{i:03d}", "file_name": name,
            "labels": [label], "primary_label": label, "multi_class": False,
            "location": "10", "seq_id": f"{prefix}seq{i // 2:03d}",
        })
    return records


def write_manifest(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """A miniature §7.2 world plus an M0-like checkpoint to initialize from."""
    root = tmp_path_factory.mktemp("qat")
    images = root / "images"

    manifests = root / "manifests"
    write_manifest(manifests / "train.jsonl", write_images(images, "tr", 8, 0))
    write_manifest(manifests / "cis_val_clean.jsonl", write_images(images, "cv", 4, 1))
    write_manifest(manifests / "trans_val.jsonl", write_images(images, "tv", 4, 2))
    supplement = write_manifest(
        manifests / "cct_empty_train_v1.jsonl",
        [
            {**r, "relative_path": f"supp/{r['file_name']}"}
            for r in write_images(root / "supp", "su", 4, 3)
        ],
    )
    calibration = write_manifest(
        manifests / "calibration_1024.jsonl",
        [dict(r, source="cct20") for r in write_images(images, "ca", 4, 4)],
    )

    classes_config = root / "classes.yaml"
    classes_config.write_text(
        yaml.safe_dump({"classes": [{"index": i, "name": n} for i, n in enumerate(CLASS_NAMES)]})
    )

    torch.manual_seed(0)
    checkpoint_path = root / "m0_best.pt"
    m0 = build_mobilenet_v2(num_classes=16, pretrained=False)
    torch.save(
        {"model": m0.state_dict(), "epoch": 11, "class_names": CLASS_NAMES,
         "run_id": "c2_fake"},
        checkpoint_path,
    )

    config = qt.QatConfig(
        source_run_id="c2_fake",
        source_checkpoint=str(checkpoint_path),
        source_checkpoint_sha256=sha256_file(checkpoint_path),
        calibration_manifest=str(calibration),
        calibration_manifest_sha256=sha256_file(calibration),
        manifests_dir=str(manifests),
        images_dir=str(images),
        supplement_manifest=str(supplement),
        supplement_dir=str(root / "supp"),
        cache_dir=str(root / "no-cache"),
        classes_config=str(classes_config),
        width=WIDTH,
        height=HEIGHT,
        epochs=1,
        batch_size=4,
        workers=0,
        output_root=str(root / "out"),
    )
    return root, config


class TestRefusals:
    def test_off_range_learning_rates_are_refused(self, world):
        _, config = world
        for lr in (5e-6, 1e-4, 1e-3):
            with pytest.raises(ValueError, match="documented range"):
                qt.train_arm(config, lr)

    def test_wrong_checkpoint_bytes_are_refused(self, world):
        _, config = world
        import dataclasses
        bad = dataclasses.replace(config, source_checkpoint_sha256="0" * 64)
        with pytest.raises(RuntimeError, match="initializes from M0"):
            qt.train_arm(bad, 1e-5)

    def test_wrong_class_order_is_refused(self, world, tmp_path):
        root, config = world
        import dataclasses
        reordered = tmp_path / "m0_reordered.pt"
        m0 = build_mobilenet_v2(num_classes=16, pretrained=False)
        torch.save(
            {"model": m0.state_dict(), "epoch": 11,
             "class_names": list(reversed(CLASS_NAMES))},
            reordered,
        )
        bad = dataclasses.replace(
            config,
            source_checkpoint=str(reordered),
            source_checkpoint_sha256=sha256_file(reordered),
        )
        with pytest.raises(RuntimeError, match="class order"):
            qt.train_arm(bad, 1e-5)

    def test_config_loader_requires_the_pins(self, tmp_path):
        path = tmp_path / "m2.yaml"
        path.write_text(yaml.safe_dump({"source_run_id": "x"}))
        with pytest.raises(ValueError, match="must pin"):
            qt.load_config(path)
        path.write_text(yaml.safe_dump({"unknown_knob": 1}))
        with pytest.raises(ValueError, match="unknown keys"):
            qt.load_config(path)


def test_arm_labels_are_stable():
    assert qt.arm_label(1e-5) == "lr1e-5"
    assert qt.arm_label(3e-5) == "lr3e-5"
    assert qt.arm_label(5e-5) == "lr5e-5"


class TestEndToEnd:
    @pytest.fixture(scope="class")
    def result(self, world):
        _, config = world
        return qt.train_arm(config, 1e-5)

    def test_history_records_the_whole_recipe(self, result):
        assert result["best_epoch"] == 1
        assert result["initialized_from"]["epoch"] == 11
        assert result["observer_calibration"]["images"] == 4
        assert result["relu6_removal_equivalence"]["exact"] is True
        (entry,) = result["history"]
        assert entry["observers_frozen"] is False  # epoch 1 observes
        assert "primary" in entry["selection_score"]

    def test_candidate_directory_speaks_the_d1_contract(self, world, result):
        root, config = world
        candidate_dir = root / "out" / "lr1e-5"
        candidate = json.loads((candidate_dir / "candidate.json").read_text())
        assert candidate["candidate_id"] == "d2_m2_qat_lr1e-5"
        assert candidate["kind"] == "int8_qat"
        assert candidate["method"] == "lr1e-5"
        assert candidate["model"]["quantization"]["scheme"] == "S8S8"
        assert candidate["source_checkpoint"]["sha256"] == config.source_checkpoint_sha256
        assert (candidate_dir / "model.onnx").exists()
        assert (candidate_dir / "coverage.json").exists()
        assert candidate["model"]["sha256"] == sha256_file(candidate_dir / "model.onnx")

    def test_shipped_artifact_is_a_real_int8_graph(self, world, result):
        """DESIGN §8.2: INT8 initializers on disk, proven bitwise-equal."""
        root, _ = world
        candidate_dir = root / "out" / "lr1e-5"
        candidate = json.loads((candidate_dir / "candidate.json").read_text())
        fold = candidate["weight_fold"]
        assert fold["folded_bytes"] < fold["source_bytes"]
        assert "int8" in fold["initializer_dtypes"]
        assert fold["equivalence"]["exact"] is True
        assert set(fold["opset"]) <= {"", "ai.onnx"}
        # The fakequant intermediate stays beside it, inspectable.
        assert (candidate_dir / "model.fakequant.onnx").exists()
        assert (
            (candidate_dir / "model.onnx").stat().st_size
            < (candidate_dir / "model.fakequant.onnx").stat().st_size
        )

    def test_export_only_reproduces_the_candidate(self, world, result):
        root, config = world
        run_dir = root / "out" / "runs" / "d2" / result["run_id"]
        before = sha256_file(root / "out" / "lr1e-5" / "model.onnx")
        candidate = qt.reexport_arm(config, run_dir)
        assert candidate["candidate_id"] == "d2_m2_qat_lr1e-5"
        assert candidate["best_epoch"] == result["best_epoch"]
        after = sha256_file(root / "out" / "lr1e-5" / "model.onnx")
        # Same checkpoint, same path, same bytes: the export is a function of
        # the weights, not of when it ran.
        assert after == before

    def test_exported_graph_executes_as_integer(self, world, result):
        root, _ = world
        coverage = json.loads((root / "out" / "lr1e-5" / "coverage.json").read_text())
        assert coverage["verdict"]["integer_execution"] is True

    def test_export_is_reproducible_from_the_best_checkpoint(self, world, result):
        # The exported ONNX must describe the selected epoch — the candidate
        # records the qat run id and best epoch so the chain is auditable.
        assert result["export"]["best_epoch"] == result["best_epoch"]
        assert result["export"]["qat_run_id"] == result["run_id"]
