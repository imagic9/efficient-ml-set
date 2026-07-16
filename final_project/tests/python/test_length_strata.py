"""DESIGN §6.3's length-strata report (PLAN C3).

The registered rule refuses to exclude or down-weight short positive sequences — a
one-frame visit is a real event. This report is the other half of that bargain: the
strata make visible what the equal weighting averages together, without ever feeding
back into the threshold. Every case is small enough to check by hand.
"""

from __future__ import annotations

import numpy as np

from wildlife_trigger import metrics as M


def report(scores, present, seqs, threshold=0.5):
    return M.positive_sequence_length_report(
        np.array(scores), np.array(present), seqs, threshold
    )


def test_length_is_positive_frames_not_burst_frames() -> None:
    """A 4-frame burst where the bobcat crosses one frame is a one-frame visit:
    one frame is all the trigger gets, however long the camera kept firing."""
    scores = [0.9, 0.1, 0.1, 0.1]
    present = [1.0, 0.0, 0.0, 0.0]
    seqs = ["burst"] * 4
    result = report(scores, present, seqs)
    assert result["length_distribution"] == {"1": 1}
    assert result["strata"]["1-2"]["sequences"] == 1


def test_strata_boundaries_are_the_registered_ones() -> None:
    """Sequences of length 1,2,3,5,6,9 land in 1-2 / 3-5 / >5 with no gaps: the
    boundary lengths 2->3 and 5->6 are where an off-by-one would hide."""
    scores, present, seqs = [], [], []
    for name, length in [("a", 1), ("b", 2), ("c", 3), ("d", 5), ("e", 6), ("f", 9)]:
        scores += [0.9] * length
        present += [1.0] * length
        seqs += [name] * length
    result = report(scores, present, seqs)
    assert result["strata"]["1-2"]["sequences"] == 2
    assert result["strata"]["3-5"]["sequences"] == 2
    assert result["strata"][">5"]["sequences"] == 2
    assert result["positive_sequences"] == 6
    assert sum(s["sequences"] for s in result["strata"].values()) == 6, (
        "every positive sequence is in exactly one stratum; none may be excluded"
    )


def test_per_stratum_recall_is_checkable_by_hand() -> None:
    """Short visits caught, long visit missed: the strata must say so separately.

    1-2: two one-frame visits, both fire -> recall 1.0, capture 1.0.
    >5:  one six-frame visit, one frame fires -> frame recall 1/6, capture 1.0.
    3-5: empty -> unsupported, not zero.
    """
    scores = [0.9] + [0.9] + [0.9, 0.1, 0.1, 0.1, 0.1, 0.1]
    present = [1.0] * 8
    seqs = ["one_a", "one_b"] + ["long"] * 6
    result = report(scores, present, seqs)

    short = result["strata"]["1-2"]
    assert short["frame_recall"] == 1.0
    assert short["sequence_balanced_recall"] == 1.0
    assert short["event_capture_rate"] == 1.0

    long = result["strata"][">5"]
    assert long["frame_recall"] == 1 / 6
    assert long["sequence_balanced_recall"] == 1 / 6
    assert long["event_capture_rate"] == 1.0, "one hit still photographs the visit"

    assert result["strata"]["3-5"] == {"sequences": 0, "supported": False}, (
        "an empty stratum reports its emptiness, not a zero that reads as failure"
    )


def test_distribution_counts_every_length() -> None:
    scores = [0.5] * 7
    present = [1.0] * 7
    seqs = ["a", "b", "b", "c", "c", "c", "c"]
    result = report(scores, present, seqs)
    assert result["length_distribution"] == {"1": 1, "2": 1, "4": 1}


def test_event_capture_needs_only_one_frame() -> None:
    """The product question — was the visit photographed at all — is satisfied by
    a single trigger, which is why capture can be 1.0 while frame recall is 0.5."""
    scores = [0.9, 0.1, 0.9, 0.1]
    present = [1.0] * 4
    seqs = ["a", "a", "b", "b"]
    result = report(scores, present, seqs)
    stratum = result["strata"]["1-2"]
    assert stratum["frame_recall"] == 0.5
    assert stratum["event_capture_rate"] == 1.0
