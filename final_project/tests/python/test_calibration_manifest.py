"""The D1 calibration manifest builder: fixed, stratified, training-only.

The manifest these tests guard is the data every PTQ candidate calibrates on —
a wrong draw here is invisible downstream (the quantizer happily computes scales
from anything), so the properties are pinned by test rather than by review:
exact total, determinism byte for byte, stratification with floors, multi-class
exclusion, and refusal of anything that smells of an evaluation split.
"""

from __future__ import annotations

import json

import pytest

from wildlife_trigger.optimize import calibration_manifest as cm


def record(image_id: str, label: str | None, source_extra: dict | None = None) -> dict:
    row = {
        "image_id": image_id,
        "file_name": f"{image_id}.jpg",
        "labels": [label] if label else ["bobcat", "coyote"],
        "primary_label": label,
        "multi_class": label is None,
        "location": "10",
        "seq_id": f"seq-{image_id}",
    }
    row.update(source_extra or {})
    return row


def write_manifest(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


@pytest.fixture()
def world(tmp_path):
    """A miniature training world: 3 CCT classes + a supplement, sizes known."""
    train = write_manifest(
        tmp_path / "train.jsonl",
        [record(f"a{i:03d}", "bobcat") for i in range(30)]
        + [record(f"b{i:03d}", "coyote") for i in range(20)]
        + [record(f"c{i:03d}", "empty") for i in range(40)]
        + [record("multi01", None)],  # multi-class: must never be drawn
    )
    supplement = write_manifest(
        tmp_path / "cct_empty_train_v1.jsonl",
        [
            record(f"s{i:03d}", "empty", {"relative_path": f"empty_supplement/s{i:03d}.jpg"})
            for i in range(50)
        ],
    )
    return train, supplement


def test_total_is_exact_and_multiclass_excluded(world):
    train, supplement = world
    chosen, report = cm.build(train, supplement, total=64, floor=4, seed=7)
    assert len(chosen) == 64
    assert report["total"] == 64
    assert all(r["primary_label"] is not None for r in chosen)
    assert not any(r["image_id"] == "multi01" for r in chosen)


def test_every_stratum_meets_its_floor(world):
    train, supplement = world
    chosen, report = cm.build(train, supplement, total=64, floor=4, seed=7)
    for key, counts in report["strata"].items():
        assert counts["drawn"] >= min(4, counts["eligible"]), key
    # Both sources are present — "including supplemental empty images" is a
    # property of the build, not a hope.
    assert report["drawn_by_source"]["cct20"] > 0
    assert report["drawn_by_source"]["empty_supplement"] > 0


def test_deterministic_byte_for_byte(world, tmp_path):
    train, supplement = world
    first, _ = cm.build(train, supplement, total=64, floor=4, seed=7)
    second, _ = cm.build(train, supplement, total=64, floor=4, seed=7)
    assert first == second
    # And a different seed is a different draw — the seed is load-bearing.
    third, _ = cm.build(train, supplement, total=64, floor=4, seed=8)
    assert third != first


def test_source_tags_travel_with_records(world):
    train, supplement = world
    chosen, _ = cm.build(train, supplement, total=64, floor=4, seed=7)
    sources = {r["source"] for r in chosen}
    assert sources == {"cct20", "empty_supplement"}
    for r in chosen:
        if r["source"] == "empty_supplement":
            assert r["relative_path"].startswith("empty_supplement/")


def test_allocation_is_roughly_proportional(world):
    train, supplement = world
    # 140 eligible singles, ask for half: each stratum should get about half its
    # size (floors distort the smallest strata only).
    chosen, report = cm.build(train, supplement, total=70, floor=2, seed=7)
    for key, counts in report["strata"].items():
        share = counts["drawn"] / counts["eligible"]
        assert 0.3 <= share <= 0.7, (key, share)


def test_refuses_val_and_test_manifests(world, tmp_path):
    train, _ = world
    for bad_name in ("cis_val_clean.jsonl", "trans_test.jsonl"):
        bad = write_manifest(tmp_path / bad_name, [record("x001", "empty")])
        with pytest.raises(ValueError, match="evaluation split"):
            cm.build(train, bad)


def test_refuses_impossible_totals(world):
    train, supplement = world
    with pytest.raises(ValueError, match="eligible"):
        cm.build(train, supplement, total=10_000, floor=4, seed=7)


def test_refuses_floors_exceeding_total(world):
    train, supplement = world
    with pytest.raises(ValueError, match="floors alone"):
        cm.build(train, supplement, total=8, floor=8, seed=7)


def test_output_sorted_and_unique(world):
    train, supplement = world
    chosen, _ = cm.build(train, supplement, total=64, floor=4, seed=7)
    ids = [r["image_id"] for r in chosen]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)
