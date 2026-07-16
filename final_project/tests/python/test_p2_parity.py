"""P2's Python instruments: the frozen fixture selection and the parity gates.

The selection tests pin the registered composition (near-threshold band, top
bobcat, seeded stride) on fabricated predictions where the right answer is
countable by hand. The parity tests run the real mechanism end to end — a real
(randomly initialised) MobileNetV2, a real ONNX export, real JPEGs — because the
gate's job is to compare frameworks, and a mocked framework would test the mock.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import pytest
import torch

from wildlife_trigger.data.preprocess import PreprocessConfig, preprocess_file
from wildlife_trigger.models.export import export_onnx
from wildlife_trigger.models.mobilenet import build_mobilenet_v2, example_input
from wildlife_trigger.runs import sha256_file
from wildlife_trigger.validate import p2_fixtures as P2F
from wildlife_trigger.validate import parity as P2
from wildlife_trigger.validate.image_fixture import write_fixture

CLASS_NAMES = [
    "opossum", "raccoon", "squirrel", "bobcat", "skunk", "dog", "coyote", "rabbit",
    "bird", "cat", "badger", "empty", "car", "deer", "fox", "rodent",
]
BOBCAT = CLASS_NAMES.index("bobcat")


def fabricated_npz(bobcat_probabilities: dict[str, list[float]]) -> dict:
    """A predictions.npz-shaped dict with the bobcat column under control."""
    data = {}
    for domain, probabilities in bobcat_probabilities.items():
        n = len(probabilities)
        matrix = np.full((n, len(CLASS_NAMES)), 0.01, dtype=np.float32)
        matrix[:, BOBCAT] = probabilities
        data[f"{domain}/probabilities"] = matrix
        data[f"{domain}/image_ids"] = np.array([f"{domain}_{i}" for i in range(n)])
        data[f"{domain}/seq_ids"] = np.array([f"seq{i // 3}" for i in range(n)])
    return data


class TestFixtureSelection:
    def test_every_near_threshold_frame_is_taken(self) -> None:
        """The carve-out must be exercised by real borderline frames; missing one
        would leave a decision flip untested exactly where flips happen."""
        threshold = 0.5
        data = fabricated_npz({
            "cis_val_clean": [0.46, 0.54, 0.10, 0.90],
            "trans_val": [0.451, 0.549, 0.20],
        })
        chosen = P2F.select_fixtures(
            data, CLASS_NAMES, threshold, np.random.default_rng(0)
        )
        near = [f for f in chosen if f["reason"] == "near_threshold"]
        assert {f["image_id"] for f in near} == {
            "cis_val_clean_0", "cis_val_clean_1", "trans_val_0", "trans_val_1",
        }

    def test_a_frame_is_one_fixture_with_its_first_reason(self) -> None:
        """A top-bobcat frame inside the band must not appear twice."""
        data = fabricated_npz({"cis_val_clean": [0.52, 0.1], "trans_val": [0.1]})
        chosen = P2F.select_fixtures(
            data, CLASS_NAMES, 0.5, np.random.default_rng(0)
        )
        ids = [f["image_id"] for f in chosen]
        assert len(ids) == len(set(ids))
        (borderline,) = [f for f in chosen if f["image_id"] == "cis_val_clean_0"]
        assert borderline["reason"] == "near_threshold", "first rule wins"

    def test_selection_is_deterministic(self) -> None:
        data = fabricated_npz({
            "cis_val_clean": list(np.linspace(0.0, 0.4, 300)),
            "trans_val": list(np.linspace(0.0, 0.4, 300)),
        })
        first = P2F.select_fixtures(data, CLASS_NAMES, 0.5, np.random.default_rng(P2F.SEED))
        second = P2F.select_fixtures(data, CLASS_NAMES, 0.5, np.random.default_rng(P2F.SEED))
        assert first == second

    def test_stride_fills_to_the_target(self) -> None:
        data = fabricated_npz({
            "cis_val_clean": list(np.linspace(0.0, 0.4, 300)),
            "trans_val": list(np.linspace(0.0, 0.4, 300)),
        })
        chosen = P2F.select_fixtures(data, CLASS_NAMES, 0.5, np.random.default_rng(0))
        assert len(chosen) == P2F.TARGET_TOTAL
        assert any(f["reason"] == "stride" for f in chosen)


@pytest.fixture(scope="module")
def parity_world(tmp_path_factory):
    """A real model, its real export, three real JPEGs — the mechanism, small."""
    root = tmp_path_factory.mktemp("p2")
    torch.manual_seed(0)
    model = build_mobilenet_v2(num_classes=len(CLASS_NAMES), pretrained=False)
    model.eval()

    onnx_path = root / "model.onnx"
    export_onnx(model, onnx_path, example_input((1, 3, 64, 64)))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    config = PreprocessConfig(width=64, height=64)
    images = {}
    for index in range(3):
        path = root / f"frame{index}.jpg"
        write_fixture(path, (200, 150), seed=index)
        images[f"img{index}"] = path

    # The npz probabilities ARE this model's own outputs, so the consistency
    # guard holds by construction — exactly as it must for the real calibrated
    # checkpoint and its committed predictions.
    npz_lookup = {}
    for image_id, path in images.items():
        tensor, _ = preprocess_file(path, config)
        with torch.inference_mode():
            logits = model(torch.from_numpy(tensor[None, ...]))
        npz_lookup[image_id] = float(torch.softmax(logits[0], 0)[BOBCAT])

    return {
        "model": model,
        "session": session,
        "config": config,
        "images": images,
        "npz_lookup": npz_lookup,
    }


def run_compare(world, image_id, threshold=0.5, npz_override=None):
    path = world["images"][image_id]
    fixture = {
        "image_id": image_id,
        "split": "cis_val_clean",
        "reason": "stride",
        "file_name": path.name,
        "sha256": sha256_file(path),
    }
    return P2.compare_fixture(
        fixture, path, world["config"], world["model"], world["session"],
        CLASS_NAMES, threshold,
        npz_override or world["npz_lookup"],
    )


class TestParityGates:
    def test_torch_and_ort_agree_on_the_real_mechanism(self, parity_world) -> None:
        """The honest end-to-end: same tensor, two frameworks, registered gates.
        If this fails, P2 could never pass on the real model either."""
        for image_id in parity_world["images"]:
            result = run_compare(parity_world, image_id)
            assert result["passed"], result["failures"]
            assert result["logits_max_abs"] <= P2.LOGITS_MAX_ABS
            assert result["top1_match"] and result["fire_match"]

    def test_a_tampered_image_is_refused(self, parity_world) -> None:
        image_id = "img0"
        path = parity_world["images"][image_id]
        fixture = {
            "image_id": image_id, "split": "cis_val_clean", "reason": "stride",
            "file_name": path.name, "sha256": "0" * 64,
        }
        with pytest.raises(RuntimeError, match="different bytes"):
            P2.compare_fixture(
                fixture, path, parity_world["config"], parity_world["model"],
                parity_world["session"], CLASS_NAMES, 0.5, parity_world["npz_lookup"],
            )

    def test_npz_gap_is_reported_not_gated(self, parity_world) -> None:
        """Issue #30's lesson, pinned: the npz holds TF32-batched values, so its
        distance from torch-CPU is a *regime measurement*, not a per-fixture
        defect. The gap must be in the record — it is the calibration-vs-
        deployment numeric distance — and must not fail the fixture. Weight
        identity is the hash chain's job; behavioural identity is
        verify_npz_regime's, under the npz's own regime."""
        drifted = dict(parity_world["npz_lookup"])
        drifted["img1"] += 0.05
        result = run_compare(parity_world, "img1", npz_override=drifted)
        assert result["npz_probability_gap"] == pytest.approx(0.05, abs=1e-6)
        assert result["passed"], result["failures"]

    def test_regime_check_reports_skip_without_cuda(self, parity_world, monkeypatch) -> None:
        """On a CUDA-less machine the regime check must say it did not run —
        a skip that looked like a pass would hide exactly what issue #30 found."""
        from wildlife_trigger.validate import parity as P2mod

        monkeypatch.setattr(P2mod.torch.cuda, "is_available", lambda: False)
        outcome = P2mod.verify_npz_regime(
            parity_world["model"], [], {}, None, parity_world["config"],
            parity_world["npz_lookup"], 3, batch_size=4,
        )
        assert outcome["status"] == "skipped_no_cuda"
        assert "passed" not in outcome, "a skip is not a pass and must not vote"

    def test_the_carve_out_names_the_borderline_frame(self, parity_world) -> None:
        """A probability within 1e-4 of the threshold may legitimately flip; the
        fixture is flagged and listed, never silently excused."""
        probability = parity_world["npz_lookup"]["img2"]
        result = run_compare(parity_world, "img2", threshold=probability + 5e-5)
        assert result["within_threshold_tolerance"] is True
        assert result["passed"], result["failures"]

    def test_far_from_threshold_is_not_carved_out(self, parity_world) -> None:
        result = run_compare(parity_world, "img2", threshold=0.99)
        assert result["within_threshold_tolerance"] is False
