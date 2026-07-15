#!/usr/bin/env python3
"""Generate the deterministic JPEG the A4 vertical slice runs on.

A4 exercises the full path — decode, preprocess, infer, decide — before Gate A, and
Gate A is what permits the CCT-20 download. So the slice needs an image that is not
CCT-20 and is not pretending to be.

This is a synthetic frame, not a photograph of anything, and it exists to make the
decode and letterbox paths do real work:

  - **1024x747**, the dominant CCT-20 `_sm` frame geometry (DESIGN §5.5), so the
    letterbox arithmetic under test is the arithmetic the real data will hit;
  - structured content rather than flat colour, so JPEG actually compresses something
    and the resize has gradients to interpolate;
  - a bright shape near the frame edge, so a preprocessor that centre-cropped would
    visibly lose it — the failure DESIGN §5.5 forbids by name.

Committed under `tests/fixtures/` because the slice must be reproducible from a fresh
clone without a network fetch. Its content is meaningless: no A4 prediction on it is a
result, and it is never training data.

Usage:
    python -m wildlife_trigger.validate.image_fixture --output tests/fixtures/frame.jpg
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

# DESIGN §5.5: the dominant frame of the `_sm` archive this pipeline consumes.
CCT_SM_DOMINANT = (1024, 747)  # (width, height)

# JPEG quality. High enough that the fixture is not dominated by block artifacts,
# and fixed so the file's hash is stable across rebuilds.
JPEG_QUALITY = 92


def synthetic_frame(width: int, height: int, seed: int = 0) -> np.ndarray:
    """Build a deterministic BGR frame with structure a resize can act on."""
    rng = np.random.default_rng(seed)

    # A vertical gradient standing in for sky-to-ground luminance.
    gradient = np.linspace(60, 180, height, dtype=np.float32)
    frame = np.repeat(gradient[:, None], width, axis=1)
    frame = np.stack([frame * 0.9, frame, frame * 1.1], axis=2)

    # Mild noise so JPEG has high-frequency content to quantize; without it the
    # decode path is unrealistically easy.
    frame += rng.normal(0.0, 4.0, frame.shape)

    # A dark band along the bottom: ground.
    frame[int(height * 0.75) :, :, :] *= 0.55

    # A bright ellipse deliberately near the LEFT EDGE. A centre crop at any of the
    # candidate input shapes would cut it; the letterbox must keep it.
    cv2.ellipse(
        frame,
        center=(int(width * 0.06), int(height * 0.55)),
        axes=(int(width * 0.04), int(height * 0.09)),
        angle=20,
        startAngle=0,
        endAngle=360,
        color=(230, 235, 240),
        thickness=-1,
    )

    # A few rectangles for edges the interpolation has to resolve.
    for i in range(6):
        x = int(width * (0.25 + 0.1 * i))
        y = int(height * 0.35)
        cv2.rectangle(
            frame,
            (x, y),
            (x + int(width * 0.05), y + int(height * 0.12)),
            color=(40 + 30 * i, 90, 200 - 20 * i),
            thickness=-1,
        )

    return np.clip(frame, 0, 255).astype(np.uint8)


def write_fixture(output: Path, size: tuple[int, int], seed: int = 0) -> dict:
    width, height = size
    frame = synthetic_frame(width, height, seed)

    output.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(
        str(output), frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    )
    if not ok:
        raise RuntimeError(f"OpenCV could not write {output}")

    return {
        "path": str(output),
        "width": width,
        "height": height,
        "seed": seed,
        "jpeg_quality": JPEG_QUALITY,
        "bytes": output.stat().st_size,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "provenance": (
            "Synthetic. Not CCT-20, not a photograph, not training data. Exists so "
            "the A4 slice can decode a real JPEG at the dominant CCT `_sm` geometry "
            "before Gate A permits the dataset download. No prediction on it is a "
            "result."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--width", type=int, default=CCT_SM_DOMINANT[0])
    parser.add_argument("--height", type=int, default=CCT_SM_DOMINANT[1])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    info = write_fixture(args.output, (args.width, args.height), args.seed)
    print(json.dumps(info, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(info, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
