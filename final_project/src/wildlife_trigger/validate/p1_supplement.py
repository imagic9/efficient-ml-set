#!/usr/bin/env python3
"""Generate the committed synthetic supplement for P1's fixture coverage.

DESIGN §10 P1 demands fixtures covering landscape, portrait, grayscale-looking IR
and odd dimensions. The 20 frozen golden fixtures are real CCT frames — which is
their virtue — but the real corpus contains only two geometries (1024x747 and
1024x768, both landscape), so **no real fixture ever exercises `pad_left > 0`,
odd dimensions, or the upscale path**. A parity suite that never runs a branch
proves nothing about it.

These five frames close that gap. They are synthetic and committed (precedent:
`tests/fixtures/frame_1024x747.jpg`), so this part of P1 is reproducible from a
fresh clone without gx10. The IR-like frame is three equal low-contrast channels,
which is what a night-time camera-trap exposure actually decodes to.

Regenerating with the same seed reproduces the same bytes; the manifest records
each file's sha256 so a drifted regeneration is loud.

Usage:
    python -m wildlife_trigger.validate.p1_supplement \
        --output-dir tests/fixtures/p1_supplement
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .image_fixture import JPEG_QUALITY, write_fixture

# name -> (width, height, why this geometry is in the suite)
SUPPLEMENT = {
    "portrait_747x1024": (
        747,
        1024,
        "portrait: pad_left > 0, the branch no real CCT fixture reaches",
    ),
    "square_512x512": (512, 512, "square: both pads zero on one axis exactly"),
    "odd_1023x767": (1023, 767, "odd dimensions: rounding and floor-divided pads"),
    "tiny_100x80": (100, 80, "smaller than the target: the upscale path"),
    "ir_night_1024x747": (
        1024,
        747,
        "grayscale-looking IR: three equal low-contrast channels, ordinary night input",
    ),
}


def ir_frame(width: int, height: int, seed: int = 7) -> np.ndarray:
    """A night-IR-like frame: one luminance plane in three equal channels.

    Low contrast on purpose — IR exposures cluster in a narrow band — with enough
    structure (noise, a bright hotspot, a horizon line) that the resize has real
    gradients to interpolate.
    """
    rng = np.random.default_rng(seed)
    luminance = np.full((height, width), 96.0, dtype=np.float32)
    luminance += rng.normal(0.0, 6.0, luminance.shape)
    luminance[int(height * 0.6) :, :] *= 0.8  # ground, slightly darker

    # A hotspot as an IR reflection: bright but not clipped.
    cv2.circle(
        luminance,
        center=(int(width * 0.7), int(height * 0.4)),
        radius=int(height * 0.08),
        color=170.0,
        thickness=-1,
    )
    channel = np.clip(luminance, 0, 255).astype(np.uint8)
    return np.stack([channel, channel, channel], axis=2)


def write_supplement(output_dir: Path) -> dict:
    entries = {}
    for name, (width, height, why) in SUPPLEMENT.items():
        path = output_dir / f"{name}.jpg"
        if name.startswith("ir_"):
            frame = ir_frame(width, height)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]):
                raise RuntimeError(f"OpenCV could not write {path}")
            entries[name] = {
                "path": str(path),
                "width": width,
                "height": height,
                "jpeg_quality": JPEG_QUALITY,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        else:
            entries[name] = write_fixture(path, (width, height), seed=0)
        entries[name]["why"] = why
        entries[name]["provenance"] = (
            "Synthetic P1 supplement. Not CCT-20, not a photograph, never training "
            "data. Exists because the real golden corpus is all landscape and "
            "cannot exercise this branch."
        )

    manifest = {
        "purpose": "P1 preprocessing-parity coverage the real golden corpus cannot give",
        "fixtures": entries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest = write_supplement(args.output_dir)
    for name, entry in manifest["fixtures"].items():
        print(f"{name}: {entry['width']}x{entry['height']} sha256={entry['sha256'][:16]}...")
    print(f"wrote {args.output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
