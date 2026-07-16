"""The acceptance test's own arithmetic (issue #19).

The tool exists to allow "revert" as an outcome, so the tests feed it series where the
right verdict is failure — an acceptance test that cannot fail is advocacy with extra
steps.
"""

from __future__ import annotations

import numpy as np
import pytest

from wildlife_trigger.validate.ap_stability import relative_step_change, score_series


def entry(epoch: int, phase: str, cis_ap: float, trans_ap: float, cis_f2: float, trans_f2: float) -> dict:
    return {
        "epoch": epoch,
        "phase": phase,
        "cis_val_clean": {"average_precision": cis_ap, "frame_f2": cis_f2},
        "trans_val": {"average_precision": trans_ap, "frame_f2": trans_f2},
    }


def test_score_series_reads_phase_b_only_and_both_metrics() -> None:
    """Phase A is excluded for the same reason selection excludes it."""
    history = [
        entry(0, "A", 0.9, 0.9, 0.9, 0.9),  # must not appear
        entry(1, "B", 0.6, 0.2, 0.5, 0.1),
        entry(2, "B", 0.7, 0.3, 0.6, 0.0),
    ]
    series = score_series(history)

    assert series["epochs"] == [1, 2]
    assert series["ap"] == pytest.approx([0.4, 0.5])
    assert series["f2_at_half"] == pytest.approx([0.3, 0.3])


def test_relative_step_change_is_scale_free() -> None:
    """The two metrics live on different scales; the criterion must not care.

    A series and the same series times ten have identical relative jitter — otherwise
    the comparison would reward whichever metric happens to sit higher.
    """
    series = [0.30, 0.36, 0.28, 0.35]
    scaled = [v * 10 for v in series]
    assert relative_step_change(series) == pytest.approx(relative_step_change(scaled))


def test_relative_step_change_by_hand() -> None:
    """[0.4, 0.5, 0.3]: |Δ| = [0.1, 0.2], mean 0.15; mean value 0.4 → 0.375."""
    assert relative_step_change([0.4, 0.5, 0.3]) == pytest.approx(0.375)


def test_a_steady_series_beats_a_jittery_one() -> None:
    """The direction of the criterion, on series shaped like the real ones.

    `jittery` is trans F2@0.5's actual failure mode — threefold swings between
    adjacent epochs; `steady` drifts. The criterion must rank them accordingly, and
    with the jitter this large the 0.5x bar must also pass.
    """
    steady = [0.40, 0.42, 0.41, 0.43, 0.44]
    jittery = [0.10, 0.31, 0.09, 0.28, 0.11]

    assert relative_step_change(steady) <= 0.5 * relative_step_change(jittery)


def test_the_criterion_can_fail() -> None:
    """An AP series as jittery as the F2 series must produce a revert, not a pass."""
    same_jitter = [0.10, 0.31, 0.09, 0.28, 0.11]
    assert not (
        relative_step_change(same_jitter) <= 0.5 * relative_step_change(same_jitter)
    ), "equal jitter is not 'at most half': the amendment reverts"


def test_short_or_degenerate_series_is_nan_not_a_verdict() -> None:
    """One phase-B epoch cannot measure jitter; the tool must not fake one."""
    assert np.isnan(relative_step_change([0.4]))
    assert np.isnan(relative_step_change([]))
