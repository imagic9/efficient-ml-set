"""The frozen golden tensors, re-derived and checked.

A frozen file that nothing reads is decoration. This is the test that gives
`golden_tensors_256x192.json` its purpose: run today's preprocessing over the same
fixtures and confirm it still produces what C1a froze.

**Skipped when the raw images are absent**, which is the normal case away from gx10 —
the JPEGs are 6.3 GB and gitignored. Skipped, not silently passed: a test that reports
success without the data it tests is worse than one that says it did not run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from wildlife_trigger.data.preprocess import PreprocessConfig
from wildlife_trigger.validate.golden_tensors import freeze_one, sha256_array

FIXTURES = Path("tests/fixtures/golden_tensors_256x192.json")
IMAGES_DIR = Path("data/raw/extracted/eccv_18_all_images_sm")

# Floats do not survive a change of machine bit-for-bit; the geometry does. See the
# `exactness` block in the frozen document.
TENSOR_TOLERANCE = 1e-5


def load_document() -> dict:
    if not FIXTURES.exists():
        pytest.skip(f"{FIXTURES} not generated yet (C1a)")
    return json.loads(FIXTURES.read_text())


def available_fixtures(document: dict) -> list[dict]:
    present = [f for f in document["fixtures"] if (IMAGES_DIR / f["file_name"]).exists()]
    if not present:
        pytest.skip(f"raw images not present under {IMAGES_DIR} (gitignored; gx10 has them)")
    return present


def test_document_matches_the_selected_input():
    """The frozen shape must be the one C1a selected, not whatever was handy."""
    document = load_document()
    assert document["input"] == "256x192"
    assert document["preprocess_fingerprint"]["width"] == 256
    assert document["preprocess_fingerprint"]["height"] == 192
    assert document["count"] >= 20  # PLAN C0 requires at least 20


def test_geometry_is_reproduced_exactly():
    """Integer arithmetic. Any drift here is a real change to DESIGN §5.5's contract."""
    document = load_document()
    config = PreprocessConfig(width=256, height=192)

    for fixture in available_fixtures(document):
        result = freeze_one(IMAGES_DIR / fixture["file_name"], config)
        assert result["geometry"] == fixture["geometry"], (
            f"{fixture['file_name']}: letterbox geometry changed. "
            f"frozen {fixture['geometry']}, now {result['geometry']}"
        )


def test_tensors_are_reproduced_within_tolerance():
    """The pixels, checked the way floats allow: close, not bit-identical."""
    document = load_document()
    config = PreprocessConfig(width=256, height=192)

    for fixture in available_fixtures(document):
        result = freeze_one(IMAGES_DIR / fixture["file_name"], config)
        frozen = fixture["tensor_float32"]
        now = result["tensor_float32"]

        assert now["shape"] == frozen["shape"]
        for channel, (a, b) in enumerate(
            zip(now["mean_per_channel"], frozen["mean_per_channel"])
        ):
            assert a == pytest.approx(b, abs=TENSOR_TOLERANCE), (
                f"{fixture['file_name']} channel {channel}: normalisation drifted"
            )
        assert now["min"] == pytest.approx(frozen["min"], abs=TENSOR_TOLERANCE)
        assert now["max"] == pytest.approx(frozen["max"], abs=TENSOR_TOLERANCE)


def test_raw_images_still_hash_to_what_c0_froze():
    """The fixtures must still be the frames the tensors were derived from."""
    document = load_document()
    for fixture in available_fixtures(document):
        digest = hashlib.sha256((IMAGES_DIR / fixture["file_name"]).read_bytes()).hexdigest()
        assert digest == fixture["raw_sha256"], f"{fixture['file_name']} is not C0's frame"


def test_the_hash_binds_shape_so_a_transpose_cannot_pass():
    """CHW and HWC over the same bytes must not collide — that is the bug it guards."""
    import numpy as np

    chw = np.zeros((3, 192, 256), dtype=np.float32)
    hwc = np.zeros((192, 256, 3), dtype=np.float32)
    assert sha256_array(chw) != sha256_array(hwc)

    # And dtype: the same values as float64 are a different tensor.
    assert sha256_array(chw) != sha256_array(chw.astype(np.float64))
