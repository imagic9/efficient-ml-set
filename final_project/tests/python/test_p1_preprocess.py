"""P1's Python half: the comparator and the synthetic supplement (PLAN C4).

The comparator is the instrument that will declare P1 passed or failed, so these
tests feed it fabricated C++ dumps where the right verdict is known: dumps copied
from the Python pipeline itself must pass, and dumps with injected drift — in the
content, the pads, or the geometry — must fail with the registered gate named.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from wildlife_trigger.data.preprocess import PreprocessConfig, preprocess_file
from wildlife_trigger.validate import p1_preprocess as P1
from wildlife_trigger.validate.image_fixture import write_fixture
from wildlife_trigger.validate.p1_supplement import SUPPLEMENT, write_supplement


def fake_cpp_dump(cpp_dir, name, mode, tensor, letterbox, opencv="4.6.0"):
    """What dump-tensor would have written, fabricated from a known tensor."""
    cpp_dir.mkdir(parents=True, exist_ok=True)
    flat = np.ascontiguousarray(tensor, dtype=np.float32)
    (cpp_dir / f"{name}.{mode}.bin").write_bytes(flat.tobytes())
    (cpp_dir / f"{name}.{mode}.json").write_text(json.dumps({
        "letterbox": {
            "resized": [letterbox.resized_width, letterbox.resized_height],
            "pad_left": letterbox.pad_left,
            "pad_top": letterbox.pad_top,
            "scale": letterbox.scale,
        },
        "tensor_sha256": hashlib.sha256(flat.tobytes()).hexdigest(),
        "opencv_version": opencv,
    }))


@pytest.fixture()
def scene(tmp_path):
    """A generated JPEG, its Python tensor, and a cpp dir to fabricate into."""
    image = tmp_path / "frame.jpg"
    write_fixture(image, (320, 200), seed=3)
    config = PreprocessConfig()
    tensor, letterbox = preprocess_file(image, config)
    return {
        "image": image,
        "config": config,
        "tensor": tensor,
        "letterbox": letterbox,
        "cpp_dir": tmp_path / "cpp",
    }


class TestComparator:
    def test_identical_tensors_pass(self, scene) -> None:
        for mode in ("fused", "reference"):
            fake_cpp_dump(scene["cpp_dir"], "f", mode, scene["tensor"], scene["letterbox"])
        result = P1.compare_fixture(
            "f", scene["image"], "test", scene["cpp_dir"], scene["config"]
        )
        assert result["passed"], result["failures"]
        assert result["errors"]["python_vs_fused"]["max_abs"] == 0.0

    def test_realistic_resize_drift_passes_the_registered_gate(self, scene) -> None:
        """The gate exists to admit exactly this world: a cross-version
        INTER_LINEAR that lands +-1 uint8 LSB on a *minority* of interior pixels
        (most interpolated values agree exactly; the disagreements sit where the
        fractional coordinates round differently). One LSB through the largest
        channel gain is (1/255)/0.229 after normalisation; 10% of content pixels
        drifted gives a mean well under the 2e-3 gate and a max at 1 LSB. If this
        fails, the gate has been tightened past the expected world."""
        drifted = scene["tensor"].copy().reshape(3, 192, 256)
        letterbox = scene["letterbox"]
        rng = np.random.default_rng(0)
        content = np.zeros(drifted.shape, dtype=bool)
        content[:, letterbox.pad_top : letterbox.pad_top + letterbox.resized_height,
                letterbox.pad_left : letterbox.pad_left + letterbox.resized_width] = True
        chosen = content & (rng.random(drifted.shape) < 0.10)
        drifted[chosen] += (1.0 / 255.0) / 0.229  # 1 LSB through the largest gain
        fake_cpp_dump(scene["cpp_dir"], "f", "fused", drifted, letterbox)
        fake_cpp_dump(scene["cpp_dir"], "f", "reference", drifted, letterbox)

        result = P1.compare_fixture(
            "f", scene["image"], "test", scene["cpp_dir"], scene["config"]
        )
        assert result["passed"], result["failures"]

    def test_content_drift_beyond_two_lsb_fails(self, scene) -> None:
        drifted = scene["tensor"].copy().reshape(3, 192, 256)
        letterbox = scene["letterbox"]
        drifted[:, letterbox.pad_top + 5, letterbox.pad_left + 5] += 0.05
        fake_cpp_dump(scene["cpp_dir"], "f", "fused", drifted, letterbox)
        fake_cpp_dump(scene["cpp_dir"], "f", "reference", scene["tensor"], letterbox)

        result = P1.compare_fixture(
            "f", scene["image"], "test", scene["cpp_dir"], scene["config"]
        )
        assert not result["passed"]
        assert any("python_vs_fused max abs" in f for f in result["failures"])
        # And the three-way design localises it: reference agreed with Python, so
        # the failure also names the fused-vs-reference gate.
        assert any("fusion bug" in f for f in result["failures"])

    def test_pad_corruption_fails_regardless_of_version(self, scene) -> None:
        """Pads never pass through the resize; a wrong pad is a wrong constant."""
        drifted = scene["tensor"].copy().reshape(3, 192, 256)
        letterbox = scene["letterbox"]
        assert letterbox.pad_top > 0, "fixture must actually have a pad to corrupt"
        drifted[:, 0, :] += 0.01  # first padded row
        fake_cpp_dump(scene["cpp_dir"], "f", "fused", drifted, letterbox)
        fake_cpp_dump(scene["cpp_dir"], "f", "reference", scene["tensor"], letterbox)

        result = P1.compare_fixture(
            "f", scene["image"], "test", scene["cpp_dir"], scene["config"]
        )
        assert not result["passed"]
        assert any("pad region" in f for f in result["failures"])

    def test_geometry_mismatch_is_named_not_averaged(self, scene) -> None:
        letterbox = scene["letterbox"]
        fake_cpp_dump(scene["cpp_dir"], "f", "fused", scene["tensor"], letterbox)
        # The reference dump claims a different pad — as a real off-by-one would.
        broken = json.loads((scene["cpp_dir"] / "f.fused.json").read_text())
        broken["letterbox"]["pad_top"] += 1
        (scene["cpp_dir"] / "f.reference.bin").write_bytes(
            (scene["cpp_dir"] / "f.fused.bin").read_bytes()
        )
        (scene["cpp_dir"] / "f.reference.json").write_text(json.dumps(broken))

        result = P1.compare_fixture(
            "f", scene["image"], "test", scene["cpp_dir"], scene["config"]
        )
        assert not result["passed"]
        assert any("geometry pad_top" in f for f in result["failures"])

    def test_a_bin_that_disagrees_with_its_own_json_is_refused(self, scene) -> None:
        """The .bin/.json pair travels together; a mismatch means the directory
        holds leftovers from two different runs, and comparing them would measure
        the wrong thing with full confidence."""
        fake_cpp_dump(scene["cpp_dir"], "f", "fused", scene["tensor"], scene["letterbox"])
        blob = bytearray((scene["cpp_dir"] / "f.fused.bin").read_bytes())
        blob[0] ^= 0xFF
        (scene["cpp_dir"] / "f.fused.bin").write_bytes(bytes(blob))
        fake_cpp_dump(scene["cpp_dir"], "f", "reference", scene["tensor"], scene["letterbox"])

        with pytest.raises(RuntimeError, match="not from one run"):
            P1.compare_fixture("f", scene["image"], "test", scene["cpp_dir"], scene["config"])


class TestSupplement:
    def test_generation_is_deterministic(self, tmp_path) -> None:
        first = write_supplement(tmp_path / "a")
        second = write_supplement(tmp_path / "b")
        for name in SUPPLEMENT:
            assert (
                first["fixtures"][name]["sha256"] == second["fixtures"][name]["sha256"]
            ), f"{name} is not reproducible; the committed manifest would drift"

    def test_committed_fixtures_match_their_manifest(self) -> None:
        """The committed JPEGs are what the manifest froze — P1 evidence depends
        on this being true on every machine that checks out the repo."""
        from pathlib import Path

        project_root = Path(__file__).resolve().parents[2]
        manifest_path = project_root / "tests/fixtures/p1_supplement/manifest.json"
        if not manifest_path.exists():
            pytest.skip("supplement not generated yet")
        manifest = json.loads(manifest_path.read_text())
        for name, entry in manifest["fixtures"].items():
            path = manifest_path.parent / Path(entry["path"]).name
            measured = hashlib.sha256(path.read_bytes()).hexdigest()
            assert measured == entry["sha256"], f"{name} drifted from its manifest"

    def test_the_suite_covers_what_the_goldens_cannot(self, tmp_path) -> None:
        """Portrait (pad_left > 0), upscale, and exact geometry claims — the
        reason the supplement exists at all (DESIGN §10 amendment)."""
        manifest = write_supplement(tmp_path / "s")
        config = PreprocessConfig()

        portrait = tmp_path / "s" / "portrait_747x1024.jpg"
        _, info = preprocess_file(portrait, config)
        assert info.pad_left > 0, "portrait must exercise the left pad"

        tiny = tmp_path / "s" / "tiny_100x80.jpg"
        _, tiny_info = preprocess_file(tiny, config)
        assert tiny_info.scale > 1.0, "tiny must exercise the upscale path"

        for name, entry in manifest["fixtures"].items():
            assert entry["width"] == SUPPLEMENT[name][0]
            assert entry["height"] == SUPPLEMENT[name][1]
