"""Gate P4's comparator: corpus parity is proven, never assumed.

The fabricated world is one split of six frames with every alignment the gate
reads: ordered ids, echoed labels, scores, decisions, and the header/footer
bindings. Each test then breaks exactly one thing the C++ runner could get
wrong — a reordered manifest, a silent skip, a score gap, a decision flip, a
stale policy — and asserts the comparator names it as a failure rather than
averaging it away.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from wildlife_trigger.validate import p4_dataset_parity as p4

CLASS_NAMES = [
    "opossum", "raccoon", "squirrel", "bobcat", "skunk", "dog", "coyote", "rabbit",
    "bird", "cat", "badger", "empty", "car", "deer", "fox", "rodent",
]
BOBCAT = CLASS_NAMES.index("bobcat")
THRESHOLD = 0.5
MODEL_SHA = "a" * 64
POLICY = {
    "policy_id": "bobcat_m1_ptq_minmax_v1",
    "model_sha256": MODEL_SHA,
    "class_map_sha256": "c" * 64,
    "targets": [{"class": "bobcat", "threshold": THRESHOLD}],
}

# Six frames: two clear fires, one clear quiet, one near-threshold, two empties.
SCORES = np.array([0.9, 0.8, 0.1, 0.50005, 0.02, 0.03])
LABELS = [["bobcat"], ["bobcat"], ["bobcat"], ["coyote"], ["empty"], ["empty"]]


def build_npz(split: str = "cis_val_clean"):
    frames = len(SCORES)
    probabilities = np.full((frames, 16), (1 - SCORES[:, None]) / 15)
    probabilities[:, BOBCAT] = SCORES
    present = np.zeros((frames, 16), dtype=np.float32)
    for i, labels in enumerate(LABELS):
        for label in labels:
            present[i, CLASS_NAMES.index(label)] = 1.0
    return {
        f"{split}/probabilities": probabilities.astype(np.float32),
        f"{split}/present": present,
        f"{split}/image_ids": np.array([f"img{i}" for i in range(frames)]),
        f"{split}/seq_ids": np.array([f"seq{i // 2}" for i in range(frames)]),
        "class_names": np.array(CLASS_NAMES),
        "model_sha256": np.array(MODEL_SHA),
    }


def write_world(tmp_path, split="cis_val_clean", cpp_scores=None, cpp_fire=None,
                header_overrides=None, skip_ids=(), reorder=False):
    manifest = tmp_path / f"{split}.jsonl"
    manifest.write_text("".join(
        json.dumps({
            "image_id": f"img{i}", "file_name": f"img{i}.jpg",
            "labels": LABELS[i], "seq_id": f"seq{i // 2}",
        }) + "\n"
        for i in range(len(SCORES))
    ))

    npz_path = tmp_path / "predictions.npz"
    np.savez_compressed(npz_path, **build_npz(split))

    from wildlife_trigger.runs import sha256_file

    scores = SCORES if cpp_scores is None else cpp_scores
    fires = (scores >= THRESHOLD) if cpp_fire is None else cpp_fire
    header = {
        "kind": "run_dataset_header",
        "model_sha256": MODEL_SHA,
        "policy_id": POLICY["policy_id"],
        "class_map_sha256": POLICY["class_map_sha256"],
        "manifest_sha256": sha256_file(manifest),
        "threads": 1,
        "onnxruntime_version": "1.27.0",
    }
    header.update(header_overrides or {})

    rows = []
    order = list(range(len(SCORES)))
    if reorder:
        order[0], order[1] = order[1], order[0]
    for i in order:
        if f"img{i}" in skip_ids:
            rows.append({"image_id": f"img{i}", "error": "decode failed", "skipped": True})
            continue
        rows.append({
            "image_id": f"img{i}", "seq_id": f"seq{i // 2}", "labels": LABELS[i],
            "target_scores": {"bobcat": float(scores[i])},
            "shutter_trigger": int(fires[i]),
            "top1_index": BOBCAT if scores[i] >= 0.5 else CLASS_NAMES.index("empty"),
        })
    footer = {
        "kind": "run_dataset_footer",
        "processed": sum(1 for r in rows if not r.get("skipped")),
        "skipped": sum(1 for r in rows if r.get("skipped")),
        "fired": sum(r.get("shutter_trigger", 0) for r in rows),
    }
    cpp_path = tmp_path / f"cpp_{split}.jsonl"
    cpp_path.write_text("".join(json.dumps(x) + "\n" for x in [header, *rows, footer]))
    return npz_path, cpp_path, manifest


def run_compare(tmp_path, **world_kwargs):
    npz_path, cpp_path, manifest = write_world(tmp_path, **world_kwargs)
    npz = np.load(npz_path, allow_pickle=False)
    return p4.compare_split(
        "cis_val_clean", npz, cpp_path, manifest, MODEL_SHA, POLICY,
        "bobcat", THRESHOLD, CLASS_NAMES,
    )


def test_identical_outputs_pass(tmp_path):
    result = run_compare(tmp_path)
    assert result["passed"], result["failures"]
    # The npz is float32, the JSONL float64: the fabricated world itself carries
    # one float32 rounding, which is far inside the registered 1e-4 gate.
    assert result["worst_score_gap"] < 1e-7
    assert result["confusion_python"] == result["confusion_cpp"]
    # The near-threshold frame is counted as near-threshold, not as noise.
    assert result["near_threshold_frames"] == 1


def test_score_gap_beyond_gate_fails(tmp_path):
    bad = SCORES.copy()
    bad[0] += 5e-4  # 5x the registered gate
    result = run_compare(tmp_path, cpp_scores=bad)
    assert not result["passed"]
    assert any("score gap" in f for f in result["failures"])


def test_decision_flip_outside_carve_out_fails(tmp_path):
    fires = SCORES >= THRESHOLD
    fires[2] = True  # python says quiet at 0.1; c++ fires — a real disagreement
    result = run_compare(tmp_path, cpp_fire=fires)
    assert not result["passed"]
    assert any("decisions differ outside the carve-out" in f for f in result["failures"])
    assert any("confusion matrices differ" in f for f in result["failures"])


def test_near_threshold_flip_is_carved_out_and_listed(tmp_path):
    fires = SCORES >= THRESHOLD
    fires[3] = not fires[3]  # the 0.50005 frame: inside the 1e-4 carve-out
    result = run_compare(tmp_path, cpp_fire=fires)
    assert result["passed"], result["failures"]
    assert [f["image_id"] for f in result["carved_out_frames"]] == ["img3"]


def test_silent_skip_fails(tmp_path):
    result = run_compare(tmp_path, skip_ids=("img4",))
    assert not result["passed"]
    assert any("skipped" in f for f in result["failures"])


def test_reordered_frames_fail(tmp_path):
    result = run_compare(tmp_path, reorder=True)
    assert not result["passed"]
    assert any("ordered image ids differ" in f for f in result["failures"])


def test_wrong_model_hash_in_header_fails(tmp_path):
    result = run_compare(tmp_path, header_overrides={"model_sha256": "b" * 64})
    assert not result["passed"]
    assert any("model_sha256" in f for f in result["failures"])


def test_wrong_policy_in_header_fails(tmp_path):
    result = run_compare(tmp_path, header_overrides={"policy_id": "bobcat_v1"})
    assert not result["passed"]
    assert any("policy_id" in f for f in result["failures"])


def test_missing_footer_is_refused(tmp_path):
    npz_path, cpp_path, manifest = write_world(tmp_path)
    lines = cpp_path.read_text().splitlines()
    cpp_path.write_text("\n".join(lines[:-1]) + "\n")  # drop the footer
    npz = np.load(npz_path, allow_pickle=False)
    with pytest.raises(RuntimeError, match="footer"):
        p4.compare_split(
            "cis_val_clean", npz, cpp_path, manifest, MODEL_SHA, POLICY,
            "bobcat", THRESHOLD, CLASS_NAMES,
        )
