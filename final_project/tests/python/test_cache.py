#!/usr/bin/env python3
"""B3 — the preprocessing cache: identity, staleness refusal, and decode guards.

Closes the two B3 test bullets DESIGN §5.5 cares about most:

  * the cache builder and the on-the-fly path produce **bit-identical** tensors, and
  * a changed preprocessing config **invalidates** the cache — `open_cache` (and the
    dataset built on it) *refuses* rather than silently serving stale pixels.
    "A cache that outlives its config trains on stale pixels and nothing downstream
    can detect it" (cache.py) — so this asserts the refusal actually fires.

Plus the missing/corrupt-file guard on `decode` (the "missing/corrupt files" half of
the B3 unit-test bullet) and a shape/determinism check on the training augmentation.

Everything synthesises its own tiny JPEGs, so it needs no CCT-20 download and runs in
the normal suite (no `needs_data` marker).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from wildlife_trigger.data.cache import build, cache_dir_for, open_cache
from wildlife_trigger.data.dataset import Augmentation, WildlifeDataset
from wildlife_trigger.data.preprocess import (
    PreprocessConfig,
    decode,
    letterbox_bgr,
)

W, H = 64, 48  # small, and not square, so the letterbox actually pads


def _write_images(images_dir: Path, n: int) -> list[dict]:
    """n tiny JPEGs of alternating aspect ratio + their manifest records."""
    images_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    records = []
    for i in range(n):
        h, w = (40, 60) if i % 2 == 0 else (55, 45)  # both landscape- and portrait-ish
        bgr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        name = f"img{i:03d}.jpg"
        assert cv2.imwrite(str(images_dir / name), bgr), "failed to write fixture JPEG"
        records.append(
            {
                "file_name": name,
                "image_id": f"id{i:03d}",
                "labels": ["bobcat"],
                "primary_label": "bobcat",
                "multi_class": False,
            }
        )
    return records


def _write_manifest(path: Path, records: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


@pytest.fixture
def world(tmp_path):
    images = tmp_path / "images"
    records = _write_images(images, 6)
    manifest = _write_manifest(tmp_path / "m.jsonl", records)
    cache_root = tmp_path / "cache"
    return images, manifest, cache_root, records


class TestCacheIdentity:
    """Bullet: the cache builder and the on-the-fly path produce identical tensors."""

    def test_cached_row_is_bit_identical_to_on_the_fly(self, world):
        images, manifest, cache_root, records = world
        cfg = PreprocessConfig(width=W, height=H)
        build(manifest, images, cache_root, cfg, workers=2)

        pixels, meta = open_cache(manifest, cache_root, cfg)
        assert meta["images"] == len(records)
        for i, r in enumerate(records):
            on_the_fly, _ = letterbox_bgr(decode(images / r["file_name"]), cfg)
            assert np.array_equal(pixels[i], on_the_fly), f"cache row {i} != decode path"

    def test_dataset_cache_and_decode_paths_agree(self, world):
        """End to end: the same dataset with and without a cache reads the same pixels."""
        images, manifest, cache_root, records = world
        cfg = PreprocessConfig(width=W, height=H)
        build(manifest, images, cache_root, cfg, workers=2)

        cached = WildlifeDataset(manifest, ["bobcat"], cfg, images, cache_root=cache_root)
        decoded = WildlifeDataset(manifest, ["bobcat"], cfg, images, cache_root=None)
        assert cached.pixels is not None and decoded.pixels is None
        for i in range(len(records)):
            assert np.array_equal(cached.letterbox(i), decoded.letterbox(i))


class TestStalenessIsRefused:
    """Bullet: a changed config invalidates the cache rather than being ignored."""

    def test_a_different_geometry_is_a_separate_cache(self, world):
        images, manifest, cache_root, _ = world
        build(manifest, images, cache_root, PreprocessConfig(W, H), workers=2)
        # The 224-vs-256 control needs both to coexist, so the shape is in the dir name.
        assert cache_dir_for(cache_root, manifest, PreprocessConfig(W, H)) != cache_dir_for(
            cache_root, manifest, PreprocessConfig(224, 224)
        )
        with pytest.raises(FileNotFoundError):
            open_cache(manifest, cache_root, PreprocessConfig(224, 224))

    def test_a_changed_pad_value_is_refused_not_ignored(self, world):
        """Same geometry -> same directory, but a different config -> must refuse."""
        images, manifest, cache_root, _ = world
        build(manifest, images, cache_root, PreprocessConfig(W, H, pad_value=114), workers=2)
        with pytest.raises(RuntimeError, match="stale"):
            open_cache(manifest, cache_root, PreprocessConfig(W, H, pad_value=100))

    def test_the_dataset_refuses_a_stale_cache(self, world):
        """WildlifeDataset falls back on a *missing* cache but must not serve a stale one."""
        images, manifest, cache_root, _ = world
        build(manifest, images, cache_root, PreprocessConfig(W, H, pad_value=114), workers=2)
        with pytest.raises(RuntimeError, match="stale"):
            WildlifeDataset(
                manifest,
                ["bobcat"],
                PreprocessConfig(W, H, pad_value=100),
                images,
                cache_root=cache_root,
            )

    def test_an_edited_manifest_is_refused(self, world):
        images, manifest, cache_root, records = world
        cfg = PreprocessConfig(W, H)
        build(manifest, images, cache_root, cfg, workers=2)
        # A new record changes the manifest hash -> the fingerprint no longer matches.
        _write_manifest(manifest, records + [dict(records[0], image_id="idNEW")])
        with pytest.raises(RuntimeError):
            open_cache(manifest, cache_root, cfg)

    def test_a_reordered_manifest_is_refused(self, world):
        images, manifest, cache_root, records = world
        cfg = PreprocessConfig(W, H)
        build(manifest, images, cache_root, cfg, workers=2)
        _write_manifest(manifest, list(reversed(records)))
        with pytest.raises(RuntimeError):
            open_cache(manifest, cache_root, cfg)


class TestRebuild:
    def test_rebuild_with_the_same_config_is_idempotent(self, world):
        images, manifest, cache_root, _ = world
        cfg = PreprocessConfig(W, H)
        first = build(manifest, images, cache_root, cfg, workers=2)
        second = build(manifest, images, cache_root, cfg, workers=2)  # "cache is current"
        assert second["fingerprint"] == first["fingerprint"]
        assert second["image_id_order_sha256"] == first["image_id_order_sha256"]


class TestDecodeGuards:
    """Bullet: missing/corrupt files are an explicit error, never a silent grey tensor."""

    def test_a_missing_file_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="cannot decode"):
            decode(tmp_path / "does_not_exist.jpg")

    def test_a_corrupt_file_raises(self, tmp_path):
        bad = tmp_path / "bad.jpg"
        bad.write_bytes(b"this is not a JPEG")
        with pytest.raises(RuntimeError, match="cannot decode"):
            decode(bad)


class TestAugmentation:
    """Bullet: the training transform is photometric (shape-preserving) and seedable."""

    def test_is_shape_preserving_and_seed_deterministic(self):
        img = np.random.default_rng(1).integers(0, 256, (H, W, 3), dtype=np.uint8)
        a = Augmentation(seed=7)(img.copy())
        b = Augmentation(seed=7)(img.copy())
        assert a.shape == img.shape and a.dtype == np.uint8  # never moves the animal
        assert np.array_equal(a, b)  # same seed -> same transform, so val/test can pin it
        assert a.min() >= 0 and a.max() <= 255
