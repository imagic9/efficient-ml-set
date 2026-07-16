"""Bobcat average precision — the checkpoint primary after the §7.2 amendment (#19).

AP is easy to implement subtly wrongly (tie handling, the first-point convention), and
this one now decides which model every optimization is compared against. Every case here
is small enough to integrate the precision-recall curve by hand in the docstring.
"""

from __future__ import annotations

import numpy as np
import pytest

from wildlife_trigger import metrics as M


def ap(scores: list[float], present: list[float]) -> float:
    return M.average_precision(np.array(scores), np.array(present))


def test_perfect_ranking_is_one() -> None:
    """Every positive above every negative: precision 1.0 at every recall level."""
    assert ap([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == pytest.approx(1.0)


def test_worst_ranking() -> None:
    """Every positive below every negative.

    Sweep: the two negatives arrive first (recall stays 0), then each positive —
    precision at recall 0.5 is 1/3, at recall 1.0 is 2/4. AP = 0.5·(1/3) + 0.5·(1/2).
    """
    assert ap([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == pytest.approx(1 / 6 + 1 / 4)


def test_interleaved_by_hand() -> None:
    """scores [.9,.8,.7], present [1,0,1].

    Group ends: rank1 P=1 R=1/2; rank2 P=1/2 R=1/2; rank3 P=2/3 R=1.
    AP = (1/2)·1 + 0·(1/2) + (1/2)·(2/3) = 5/6.
    """
    assert ap([0.9, 0.8, 0.7], [1, 0, 1]) == pytest.approx(5 / 6)


def test_tied_scores_are_one_threshold() -> None:
    """A tie no threshold can split must not be split by the metric.

    Both frames score 0.9; one is a bobcat. No threshold separates them, so the only
    honest operating point takes both: precision 1/2 at recall 1. AP = 0.5 — and NOT
    1.0, which is what a naive sweep that visits the positive first would report.
    """
    assert ap([0.9, 0.9], [1, 0]) == pytest.approx(0.5)
    assert ap([0.9, 0.9], [0, 1]) == pytest.approx(0.5), "order within a tie is noise"


def test_no_positives_is_zero_not_a_crash() -> None:
    """A domain that cannot be scored drags the selection mean down, not the run."""
    assert ap([0.9, 0.1], [0, 0]) == 0.0


def test_all_positives_is_one() -> None:
    assert ap([0.9, 0.1], [1, 1]) == pytest.approx(1.0)


def test_score_shift_does_not_move_ap_but_moves_f2() -> None:
    """The property the amendment buys: AP reads the ranking, not the values.

    Dividing every score by 10 pushes them all under the 0.5 line — F2@0.5 collapses to
    zero while the model's ranking, and therefore AP, is untouched. This is trans-val's
    failure mode in miniature: scores cluster low, and the fixed threshold reads that
    as model change.
    """
    scores = [0.9, 0.6, 0.4, 0.2]
    present = [1.0, 0.0, 1.0, 0.0]
    seqs = ["a", "b", "c", "d"]

    high = ap(scores, present)
    low = ap([s / 10 for s in scores], present)
    assert high == pytest.approx(low), "AP must be invariant to monotone rescaling"

    f2_high = M.target_presence_metrics(
        np.array(scores), np.array(present), seqs, 0.5
    )["frame_f2"]
    f2_low = M.target_presence_metrics(
        np.array([s / 10 for s in scores]), np.array(present), seqs, 0.5
    )["frame_f2"]
    assert f2_high > 0 and f2_low == 0.0, (
        "the fixed threshold reads a rescaling as collapse; that is issue #19"
    )


def test_selection_score_primary_is_f2_after_the_verdict_and_says_so() -> None:
    """The primary is F2@0.5 again — the AP amendment was reverted by its own test.

    The dict names its own primary because the histories written during the amendment's
    one day carry AP means under the same `primary` key, and a bare number cannot say
    which rule produced it.
    """
    score = M.selection_score(
        cis={"frame_f2": 0.6, "sequence_balanced_recall": 0.75, "average_precision": 0.9},
        trans={"frame_f2": 0.2, "sequence_balanced_recall": 0.25, "average_precision": 0.9},
        macro_f1=0.44,
    )
    assert score["primary"] == pytest.approx(0.4), "F2 mean, not the AP mean of 0.9"
    assert score["primary_metric"] == M.PRIMARY_METRIC
    assert "f2" in M.PRIMARY_METRIC, "the verdict reverted the primary to F2"
    assert M.selection_key(score) == pytest.approx((0.4, 0.5, 0.44)), (
        "tie-breaks unchanged throughout: recall first, macro F1 second"
    )


def test_bootstrap_supports_ap_without_a_threshold() -> None:
    """The amendment's acceptance test needs seq_id-cluster CIs *of AP*."""
    rng = np.random.default_rng(0)
    scores = rng.random(200)
    present = (rng.random(200) < 0.3).astype(float)
    seqs = [f"s{i // 4}" for i in range(200)]

    result = M.bootstrap_sequence_clusters(
        scores, present, seqs, threshold=0.5, metric="average_precision", replicates=200
    )

    assert result["metric"] == "average_precision"
    assert result["point_estimate"] == pytest.approx(
        M.average_precision(scores, present)
    )
    assert result["ci95_low"] <= result["point_estimate"] <= result["ci95_high"]
