"""The letterbox geometry, and the utilisation number the input decision rests on.

`pixel_utilisation` is not a diagnostic here — PLAN C1a picks the Core input using it, so
an inflated value argues for the wrong tensor shape. It is asserted exactly, against
values derived by hand, because the bound that used to guard it (0.96 < u < 0.99) was
wide enough to admit a wrong denominator for as long as it took someone to read the code.
"""

from __future__ import annotations

import numpy as np
import pytest

from wildlife_trigger.data.preprocess import (
    PreprocessConfig,
    letterbox_bgr,
    letterbox_geometry,
)

# CCT-20's dominant `_sm` frame, and the reason 256x192 was proposed at all.
CCT_FRAME = (1024, 747)


def test_dominant_frame_at_256x192_matches_design():
    """DESIGN §5.5 predicts 97.4%. The tensor is 256x192; the content is 256x187."""
    config = PreprocessConfig(width=256, height=192)
    resized_width, resized_height, scale = letterbox_geometry(*CCT_FRAME, config)

    assert (resized_width, resized_height) == (256, 187)  # 747 * 0.25 = 186.75 -> 187
    assert scale == 0.25  # 256/1024 exactly, which is why libjpeg can do it for free

    _, info = letterbox_bgr(np.zeros((747, 1024, 3), dtype=np.uint8), config)
    assert info.pixel_utilisation() == pytest.approx((256 * 187) / (256 * 192))
    assert info.pixel_utilisation() == pytest.approx(0.9740, abs=1e-4)


def test_dominant_frame_at_224x224_wastes_a_quarter():
    """DESIGN §5.5 predicts 72.8% — the measured claim behind not choosing a square."""
    config = PreprocessConfig(width=224, height=224)
    _, info = letterbox_bgr(np.zeros((747, 1024, 3), dtype=np.uint8), config)

    assert (info.resized_width, info.resized_height) == (224, 163)
    assert info.pixel_utilisation() == pytest.approx((224 * 163) / (224 * 224))
    assert info.pixel_utilisation() == pytest.approx(0.7277, abs=1e-4)


def test_utilisation_denominator_is_the_canvas_not_the_pads():
    """The regression: an odd pad difference must not inflate the result.

    1024x747 into 256x192 leaves 5 rows of padding, which split 2 top and 3 bottom. The
    old denominator, `resized + 2 * pad`, reconstructed 191 rows instead of 192 and
    reported 97.9% where DESIGN says 97.4%. Small, and in the flattering direction — it
    argued for the shape it was measuring.
    """
    config = PreprocessConfig(width=256, height=192)
    _, info = letterbox_bgr(np.zeros((747, 1024, 3), dtype=np.uint8), config)

    assert info.pad_top == 2
    assert info.resized_height + 2 * info.pad_top == 191  # the asymmetry, made explicit
    assert info.target_height == 192

    old_formula = (info.resized_width * info.resized_height) / (
        (info.resized_width + 2 * info.pad_left) * (info.resized_height + 2 * info.pad_top)
    )
    assert old_formula == pytest.approx(0.9791, abs=1e-4)
    assert info.pixel_utilisation() < old_formula


def test_utilisation_is_one_when_the_frame_fits_exactly():
    """No padding, no waste. The upper bound the number must actually be able to reach."""
    config = PreprocessConfig(width=256, height=192)
    _, info = letterbox_bgr(np.zeros((192, 256, 3), dtype=np.uint8), config)

    assert (info.pad_left, info.pad_top) == (0, 0)
    assert info.pixel_utilisation() == pytest.approx(1.0)


def test_the_whole_frame_survives_the_letterbox():
    """`min` scale, never `max`: DESIGN §5.5 forbids a crop that can drop the animal.

    The claim is checked against `source * scale`, not against the source aspect ratio.
    Those differ once a side gets short: a 3000x100 frame scales to 256x8.53, and 8.53
    rows can only be stored as 9 — a 5% aspect error that no implementation can avoid
    and that says nothing about cropping. Asserting the ratio directly would demand the
    code beat integer arithmetic.
    """
    config = PreprocessConfig(width=256, height=192)
    for source_width, source_height in [
        (1024, 747),  # the dominant CCT frame
        (1024, 768),  # the other observed geometry
        (640, 480),
        (100, 3000),  # extreme portrait
        (3000, 100),  # extreme landscape
    ]:
        resized_width, resized_height, scale = letterbox_geometry(
            source_width, source_height, config
        )

        # Fits: nothing is cropped away.
        assert resized_width <= config.width
        assert resized_height <= config.height

        # Both sides took the same scale, to within the one pixel rounding costs.
        assert abs(resized_width - source_width * scale) <= 1
        assert abs(resized_height - source_height * scale) <= 1

        # And the scale is the largest that fits, not merely some scale that does.
        assert scale == min(config.width / source_width, config.height / source_height)


def test_extreme_aspect_ratios_keep_at_least_one_pixel():
    """A degenerate frame must not round to a zero-width tensor and divide by nothing."""
    config = PreprocessConfig(width=256, height=192)
    resized_width, resized_height, _ = letterbox_geometry(10000, 1, config)
    assert resized_width >= 1
    assert resized_height >= 1
