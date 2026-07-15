"""Tests for the label semantics the evaluation depends on (B3/C1a).

These are not API tests. Each one corresponds to a way the pipeline can produce a
confident, wrong number without anything crashing — and two of them are bugs that
actually happened.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from wildlife_trigger import metrics as M
from wildlife_trigger.data.dataset import WildlifeDataset, class_weights
from wildlife_trigger.data.preprocess import PreprocessConfig

CLASSES = ["bobcat", "coyote", "empty"]


def write_manifest(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "m.jsonl"
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    return path


def make_image(tmp_path: Path, name: str) -> None:
    import cv2

    cv2.imwrite(str(tmp_path / name), np.full((40, 60, 3), 120, dtype=np.uint8))


def record(image_id: str, labels: list[str], seq: str = "s1") -> dict:
    return {
        "image_id": image_id,
        "file_name": f"{image_id}.jpg",
        "labels": sorted(labels),
        "primary_label": labels[0] if len(labels) == 1 else None,
        "multi_class": len(labels) > 1,
        "location": 1,
        "seq_id": seq,
    }


def build(tmp_path: Path, records: list[dict], class_names: list[str]) -> WildlifeDataset:
    for r in records:
        make_image(tmp_path, r["file_name"])
    manifest = write_manifest(tmp_path, records)
    return WildlifeDataset(
        manifest, class_names, PreprocessConfig(width=64, height=48), tmp_path
    )


def test_returned_index_is_the_dataset_index_not_a_class_index(tmp_path: Path) -> None:
    """The bug that nearly shipped: a loop variable shadowed the dataset index.

    Evaluation uses item["index"] to look up the frame's seq_id. If it returns a class
    index instead, sequence-balanced recall is computed against the wrong sequences —
    for every frame, silently, with a plausible number at the end.
    """
    records = [record(f"img{i}", ["bobcat"], seq=f"s{i}") for i in range(3)]
    dataset = build(tmp_path, records, CLASSES)

    for i in range(len(dataset)):
        assert dataset[i]["index"] == i, "item['index'] must address the dataset"


def test_multi_class_frame_is_ignored_by_ce_but_still_present(tmp_path: Path) -> None:
    """DESIGN B3: exclude the seven from cross-entropy, keep them for presence.

    Deleting them would quietly shrink the positive count recall is measured against.
    """
    records = [record("multi", ["bobcat", "coyote"])]
    dataset = build(tmp_path, records, CLASSES)
    item = dataset[0]

    assert item["target"] == -1, "CrossEntropyLoss(ignore_index=-1) must skip it"
    assert item["present"][CLASSES.index("bobcat")] == 1.0
    assert item["present"][CLASSES.index("coyote")] == 1.0


def test_unmodelled_label_means_no_animal_present(tmp_path: Path) -> None:
    """DESIGN §5.2's 15-output arm has no `empty` class, but validation is full of them.

    For that head an empty frame is genuinely "no animal present" — a negative for every
    target, which is exactly what the false-fire rate counts.
    """
    animals = ["bobcat", "coyote"]
    records = [record("e", ["empty"])]
    dataset = build(tmp_path, records, animals)

    assert dataset.unmodelled_labels == ["empty"], "must be recorded, not swallowed"
    item = dataset[0]
    assert item["target"] == -1
    assert item["present"].sum() == 0.0


def test_class_weights_ignore_multi_class_and_survive_an_absent_class() -> None:
    """CCT-20 train has no `empty` at all before the supplement; 1/0 would poison it."""
    records = [record("a", ["bobcat"]), record("b", ["bobcat"]), record("c", ["coyote"]),
               record("d", ["bobcat", "coyote"])]
    weights = class_weights(records, CLASSES)

    assert torch.isfinite(weights).all(), "an absent class must not produce inf"
    assert weights[CLASSES.index("coyote")] > weights[CLASSES.index("bobcat")], (
        "the rarer class must carry the larger weight"
    )


# --- the metrics that decide the project -------------------------------------


def test_sequence_balanced_recall_does_not_let_one_long_burst_dominate() -> None:
    """The reason DESIGN §6.3 uses it at all.

    Sequence A is a 10-frame burst, all detected. Sequence B is a 1-frame visit, missed.
    Frame recall says 10/11 = 91% — excellent. But the trigger missed half the visits,
    and a one-frame visit is a real event the product must not silently ignore.
    """
    scores = np.array([0.9] * 10 + [0.1])
    present = np.ones(11)
    seq_ids = ["burst"] * 10 + ["single"]

    result = M.target_presence_metrics(scores, present, seq_ids, threshold=0.5)

    assert result["frame_recall"] == pytest.approx(10 / 11, abs=1e-3)
    assert result["sequence_balanced_recall"] == pytest.approx(0.5), (
        "each visit counts once, so missing one of two visits is 50%"
    )
    assert result["event_capture_rate"] == pytest.approx(0.5)


def test_threshold_rule_picks_the_largest_meeting_the_recall_floor() -> None:
    """DESIGN §6.3: largest, not best.

    Among thresholds meeting the 90% floor, the highest fires least often. F2 does not
    get to trade the floor away.
    """
    scores = np.array([0.95, 0.85, 0.75, 0.05, 0.04])
    present = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    seqs = ["a", "b", "c", "d", "e"]
    domains = {"cis": (scores, present, seqs), "trans": (scores, present, seqs)}

    result = M.select_threshold(domains, min_sequence_recall=0.90)

    assert result["threshold"] == pytest.approx(0.75), (
        "0.75 catches all three positives; anything higher misses one"
    )
    assert result["unmet_constraint"] is None
    assert "primary" in result["rule"]


def test_the_rule_never_returns_a_fire_on_everything_threshold() -> None:
    """"Non-trivial" in DESIGN §6.3 is load-bearing, and dropping it is silent.

    Candidates are the observed scores, so the smallest one fires on every frame and
    scores 100% recall by construction. Without the non-trivial qualifier the 90% floor
    is *always* satisfiable, the fallback becomes dead code, and a hopeless model gets an
    operating point that photographs everything.

    Here the one positive scores below both negatives, so the only way to catch it is to
    fire on all three.
    """
    scores = np.array([0.10, 0.50, 0.90])
    present = np.array([1.0, 0.0, 0.0])
    seqs = ["a", "b", "c"]
    domains = {"cis": (scores, present, seqs), "trans": (scores, present, seqs)}

    result = M.select_threshold(domains, min_sequence_recall=0.90)

    assert result["chosen_is_trivial"] is False, "a trigger that fires on every frame is not an operating point"
    assert "fallback" in result["rule"]
    assert result["unmet_constraint"], "the unmet constraint must be recorded, not hidden"


def test_macro_f1_excludes_classes_without_real_support() -> None:
    """badger has one validation image. Averaging it in makes macro F1 a coin flip."""
    probabilities = np.tile(np.array([[0.8, 0.1, 0.1]]), (30, 1))
    present = np.zeros((30, 3))
    present[:, 0] = 1.0  # bobcat: 30 images
    present[0, 1] = 1.0  # coyote: 1 image
    seq_ids = [f"s{i}" for i in range(30)]

    result = M.per_class_metrics(probabilities, present, CLASSES, seq_ids)

    assert "bobcat" in result["macro_f1_included_classes"]
    assert "coyote" not in result["macro_f1_included_classes"], (
        "1 image is below the 20-image / 5-sequence support floor"
    )


def test_bootstrap_resamples_sequences_not_frames() -> None:
    """Frames within a burst are near-duplicates; resampling them independently would
    report an interval several times narrower than the evidence supports."""
    scores = np.concatenate([np.full(50, 0.9), np.full(50, 0.1)])
    present = np.concatenate([np.ones(50), np.zeros(50)])
    # Only two sequences: the interval must be wide, because there are really only two
    # independent observations here.
    seq_ids = ["a"] * 50 + ["b"] * 50

    result = M.bootstrap_sequence_clusters(
        scores, present, seq_ids, threshold=0.5, replicates=200
    )

    assert result["resampled"] == "seq_id clusters, not frames"
    assert result["ci95_high"] - result["ci95_low"] > 0.1, (
        "two sequences cannot yield a tight interval"
    )
