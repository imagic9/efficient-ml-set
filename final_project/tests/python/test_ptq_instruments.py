"""The M1 PTQ machinery: manifest-driven calibration, hash refusals, evidence.

The world here is deliberately tiny — a 16-class conv net and a handful of
generated JPEGs — because what these tests pin is not accuracy but *wiring*:
the reader feeds batch-1 tensors in manifest order and rewinds; the config path
refuses moved bytes; a candidate directory comes out holding the quantized
graph, its description, and the coverage verdict. Accuracy is D1's evidence
run's job, on gx10, on the real corpus.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
import torch
import yaml

from wildlife_trigger.optimize import ptq
from wildlife_trigger.data.dataset import WildlifeDataset
from wildlife_trigger.data.preprocess import PreprocessConfig
from wildlife_trigger.models.export import export_onnx
from wildlife_trigger.runs import sha256_file

CLASS_NAMES = [
    "opossum", "raccoon", "squirrel", "bobcat", "skunk", "dog", "coyote", "rabbit",
    "bird", "cat", "badger", "empty", "car", "deer", "fox", "rodent",
]

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


def write_images(root, count: int, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(count):
        name = f"img{i:03d}.jpg"
        pixels = rng.integers(0, 255, size=(80, 100, 3), dtype=np.uint8)
        assert cv2.imwrite(str(root / name), pixels)
        records.append(
            {
                "image_id": f"img{i:03d}",
                "file_name": name,
                "labels": ["bobcat" if i % 2 else "empty"],
                "primary_label": "bobcat" if i % 2 else "empty",
                "multi_class": False,
                "location": "10",
                "seq_id": f"seq{i // 2:03d}",
                "source": "cct20",
            }
        )
    return records


def write_manifest(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Source ONNX + calibration manifest + images, hashes all recorded."""
    root = tmp_path_factory.mktemp("ptq")
    images = root / "images"
    records = write_images(images, 6)
    manifest = write_manifest(root / "manifests" / "calibration_6.jsonl", records)

    classes_config = root / "classes.yaml"
    classes_config.write_text(
        yaml.safe_dump({"classes": [{"index": i, "name": n} for i, n in enumerate(CLASS_NAMES)]})
    )

    torch.manual_seed(0)
    source = root / "m0.onnx"
    export_onnx(TinyNet(), source, torch.randn(1, 3, HEIGHT, WIDTH))

    config = {
        "source_run_id": "c2_test_run",
        "source_onnx": str(source),
        "source_onnx_sha256": sha256_file(source),
        "calibration_manifest": str(manifest),
        "calibration_manifest_sha256": sha256_file(manifest),
        "calibration_images": 6,
        "images_dir": str(images),
        "supplement_dir": str(root / "supplement"),
        "cache_dir": str(root / "no-cache"),
        "classes_config": str(classes_config),
        "width": WIDTH,
        "height": HEIGHT,
        "methods": ["minmax"],
        "output_root": str(root / "out"),
    }
    config_path = root / "m1_ptq.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return root, config, config_path


class TestManifestCalibrationReader:
    def test_batches_are_batch1_in_manifest_order_and_rewindable(self, world):
        root, config, _ = world
        dataset = WildlifeDataset(
            root / "manifests" / "calibration_6.jsonl",
            CLASS_NAMES,
            PreprocessConfig(width=WIDTH, height=HEIGHT),
            root / "images",
        )
        reader = ptq.ManifestCalibrationReader(dataset, "input")
        first_pass = []
        while (batch := reader.get_next()) is not None:
            assert set(batch) == {"input"}
            assert batch["input"].shape == (1, 3, HEIGHT, WIDTH)
            assert batch["input"].dtype == np.float32
            first_pass.append(batch["input"])
        assert len(first_pass) == 6

        reader.rewind()
        second_first = reader.get_next()["input"]
        # Same order, same bytes: the three calibration methods compare on
        # identical data or the comparison means nothing.
        np.testing.assert_array_equal(second_first, first_pass[0])


class TestConfig:
    def test_missing_keys_are_refused(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(yaml.safe_dump({"methods": ["minmax"]}))
        with pytest.raises(ValueError, match="lacks required keys"):
            ptq.load_config(bad)

    def test_unknown_methods_are_refused(self, world, tmp_path):
        _, config, _ = world
        bad = tmp_path / "bad.yaml"
        bad.write_text(yaml.safe_dump({**config, "methods": ["kl_divergence"]}))
        with pytest.raises(ValueError, match="unknown calibration methods"):
            ptq.load_config(bad)


class TestGenerateCandidates:
    def test_moved_source_bytes_are_refused(self, world):
        _, config, _ = world
        with pytest.raises(RuntimeError, match="not the M0"):
            ptq.generate_candidates({**config, "source_onnx_sha256": "0" * 64})

    def test_moved_manifest_bytes_are_refused(self, world):
        _, config, _ = world
        with pytest.raises(RuntimeError, match="frozen"):
            ptq.generate_candidates(
                {**config, "calibration_manifest_sha256": "0" * 64}
            )

    def test_wrong_image_count_is_refused(self, world):
        _, config, _ = world
        with pytest.raises(RuntimeError, match="different corpus"):
            ptq.generate_candidates({**config, "calibration_images": 1024})

    def test_candidate_directory_holds_the_evidence(self, world):
        root, config, config_path = world
        summary = ptq.generate_candidates(ptq.load_config(config_path))

        candidate_dir = root / "out" / "minmax"
        model = candidate_dir / "model.onnx"
        candidate = json.loads((candidate_dir / "candidate.json").read_text())
        coverage = json.loads((candidate_dir / "coverage.json").read_text())

        assert model.exists()
        assert candidate["model"]["sha256"] == sha256_file(model)
        assert candidate["candidate_id"] == "d1_m1_ptq_minmax"
        assert candidate["source_onnx"]["sha256"] == config["source_onnx_sha256"]
        assert candidate["calibration"]["images"] == 6
        assert candidate["model"]["quantization"]["scheme"] == "S8S8"
        assert candidate["model"]["quantization"]["per_channel"] is True
        # The verdict is recorded either way; for this all-conv toy it must be
        # integer — a float fallback here would mean the toolchain regressed.
        assert candidate["integer_execution"] is True
        assert coverage["verdict"]["integer_execution"] is True
        assert (candidate_dir / f"m1_ptq_minmax.optimized.onnx").exists()
        assert summary["minmax"]["model_sha256"] == candidate["model"]["sha256"]
