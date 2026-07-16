#!/usr/bin/env python3
"""B3 — the canonical preprocessing contract, in Python (DESIGN §5.5).

This is the Python half of a pair. `cpp/src/preprocess.cpp` is the other half, and P1
exists to measure the gap between them. Every decision here mirrors the C++ deliberately
— the same `min` scale, the same `lround`, the same integer pad arithmetic, the same
channel inversion — because a difference that is *intentional* can be measured and a
difference that is accidental cannot even be found.

The steps, from DESIGN §5.5:

  1. decode JPEG as 8-bit BGR
  2. BGR -> RGB
  3. resize preserving aspect ratio to fit inside (width, height)
  4. centre-pad the remainder with RGB (114, 114, 114)
  5. float32, divide by 255
  6. normalise with ImageNet mean/std
  7. HWC RGB -> contiguous NCHW

**Steps 1-4 produce a uint8 letterbox and are what the cache stores.** Steps 5-7 are
pure arithmetic on that array and run per batch. The split is not arbitrary: 1-4 are the
expensive, deterministic part, and 5-7 are the part augmentation must sit inside.

The only step whose implementation could legitimately differ between the two languages
is the INTER_LINEAR resize — which is exactly why the OpenCV 4.6 (C++, bookworm apt)
versus 4.13 (Python wheel) gap is a named P1 risk rather than a curiosity. Everything
else is exactly defined by the spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# DESIGN §5.5 line 603, and cpp/include/wildlife_trigger/preprocess.hpp. Duplicated
# across the language boundary by necessity.
#
# Nothing yet asserts the two copies agree — P1 is that comparison, and
# `tests/fixtures/golden_tensors_256x192.json` is what it will be built from. Read the
# `exactness` block there first: the geometry can be demanded exactly, the float tensors
# only within a tolerance.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Mid-grey. Not black: a black bar is a plausible night-time pixel value, whereas this
# is not confusable with content.
PAD_VALUE = 114


@dataclass(frozen=True)
class PreprocessConfig:
    """The Core input is 256x192, frozen by C1a on 2026-07-16 (DESIGN §5.5).

    Width and height stay parameters rather than becoming constants: C1a needed to build
    a 224x224 cache and train an arm against it to make the choice, and the same lever is
    what a future re-open would need. The default is the decision.
    """

    width: int = 256
    height: int = 192
    pad_value: int = PAD_VALUE
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD

    def fingerprint(self) -> dict:
        """What the cache is keyed by. A change here must invalidate cached pixels."""
        return {
            "width": self.width,
            "height": self.height,
            "pad_value": self.pad_value,
            # mean/std deliberately excluded: they are applied *after* the cache, so
            # changing them does not stale a cached letterbox. Including them would
            # force an hour of rebuild for an arithmetic change.
            "interpolation": "INTER_LINEAR",
            "steps_cached": "1-4 (decode, BGR->RGB, aspect-preserving resize, pad)",
        }


@dataclass
class LetterboxInfo:
    source_width: int
    source_height: int
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int
    scale: float
    # The canvas actually padded into. Carried rather than re-derived from the pads:
    # see pixel_utilisation.
    target_width: int
    target_height: int

    def pixel_utilisation(self) -> float:
        """Fraction of the tensor holding real pixels rather than grey bars.

        The denominator is the real canvas, not `resized + 2 * pad`. The pads are
        floor-divided, so when the difference is odd the far side carries one extra
        pixel and `resized + 2 * pad` lands one short of the canvas — which inflates
        the result. 1024x747 at 256x192 pads 2 rows top and 3 bottom, and the old form
        returned 97.9% against DESIGN §5.5's 97.4%.
        """
        tensor_px = self.target_width * self.target_height
        if tensor_px <= 0:
            return 0.0
        return (self.resized_width * self.resized_height) / tensor_px


def letterbox_geometry(
    source_width: int, source_height: int, config: PreprocessConfig
) -> tuple[int, int, float]:
    """The content size inside the letterbox, and the scale that produced it.

    Split out from `letterbox_bgr` because the geometry is answerable from two integers
    while the pixels are not: C1a's input-cost table needs the real-pixel utilization of
    57,864 frames across candidate geometries, and decoding them all to learn something
    already recorded in the manifests would be absurd. The alternative — a second copy of
    this arithmetic in the reporting tool — is how the two copies drift and the reported
    utilization stops describing the tensor the network is actually fed.
    """
    scale = min(config.width / source_width, config.height / source_height)

    # round(), not truncate: 1024x747 into 256x192 gives 186.75 rows, and truncating
    # loses a row of animal for no reason. Python's round() is banker's rounding, so
    # np.rint would disagree with C++'s std::lround on exact .5 — use floor(x + 0.5).
    #
    # Clamped: rounding can exceed the target by one pixel at some aspect ratios, which
    # would overflow the canvas.
    resized_width = min(config.width, max(1, int(np.floor(source_width * scale + 0.5))))
    resized_height = min(config.height, max(1, int(np.floor(source_height * scale + 0.5))))
    return resized_width, resized_height, scale


def letterbox_bgr(bgr: np.ndarray, config: PreprocessConfig) -> tuple[np.ndarray, LetterboxInfo]:
    """Steps 2-4: BGR in, uint8 RGB letterbox out. This is what the cache stores.

    Mirrors cpp/src/preprocess.cpp line for line. `min` not `max` for the scale: `max`
    would fill the target and crop the overflow, which is the centre-crop DESIGN §5.5
    forbids because it can remove a small animal at the frame edge.
    """
    if bgr is None or bgr.size == 0:
        raise ValueError("preprocess: empty image")
    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError(f"preprocess: expected 8-bit 3-channel BGR, got {bgr.shape} {bgr.dtype}")

    source_height, source_width = bgr.shape[:2]
    resized_width, resized_height, scale = letterbox_geometry(
        source_width, source_height, config
    )

    resized = cv2.resize(
        bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )

    canvas = np.full(
        (config.height, config.width, 3), config.pad_value, dtype=np.uint8
    )
    pad_left = (config.width - resized_width) // 2
    pad_top = (config.height - resized_height) // 2
    canvas[pad_top : pad_top + resized_height, pad_left : pad_left + resized_width] = resized

    # Step 2, done last on the letterboxed canvas rather than first on the full frame:
    # identical result, one third of the pixels touched. The pad value is grey, so
    # inverting its channels is a no-op.
    rgb = canvas[:, :, ::-1]

    info = LetterboxInfo(
        source_width=source_width,
        source_height=source_height,
        resized_width=resized_width,
        resized_height=resized_height,
        pad_left=pad_left,
        pad_top=pad_top,
        scale=scale,
        target_width=config.width,
        target_height=config.height,
    )
    return np.ascontiguousarray(rgb), info


def decode(path: str | Path) -> np.ndarray:
    """Step 1. A corrupt frame raises rather than becoming a silently grey tensor."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(
            f"cannot decode image: {path} (missing, truncated, or not an image). "
            "A corrupt frame must be an explicit error, never a grey tensor "
            "indistinguishable from a legitimately empty night frame."
        )
    return bgr


def normalise(rgb_uint8: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """Steps 5-7: uint8 HWC RGB letterbox -> float32 CHW normalised.

    Runs after the cache and after augmentation, on an array that is already the final
    geometry.
    """
    scaled = rgb_uint8.astype(np.float32) / 255.0
    scaled -= np.asarray(config.mean, dtype=np.float32)
    scaled /= np.asarray(config.std, dtype=np.float32)
    return np.ascontiguousarray(scaled.transpose(2, 0, 1))


def preprocess_file(path: str | Path, config: PreprocessConfig) -> tuple[np.ndarray, LetterboxInfo]:
    """The whole contract, steps 1-7. The reference path, and what P1 compares."""
    letterbox, info = letterbox_bgr(decode(path), config)
    return normalise(letterbox, config), info
