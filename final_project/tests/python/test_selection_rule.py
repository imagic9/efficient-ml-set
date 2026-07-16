"""DESIGN §7.2's checkpoint rule, including the two tie-breaks (issue #12).

The rule names three quantities and an order for them. Before this, only the first was
compared, so the other two were computed, stored, and never allowed to decide anything —
which is worse than not having them, because the run summary looked like the rule was
being applied.

These tests are about which checkpoint the rule *chooses*, so each one constructs a pair
that can only be separated at one level of the ladder. `selection_score` values are ratios
of finite frame and sequence counts; the equalities below are ordinary arithmetic, not
contrived floats.
"""

from __future__ import annotations

import pytest

from wildlife_trigger import metrics as M


def score(primary: float, recall: float, macro_f1: float) -> dict:
    """A selection score with the three values the rule reads, spelled out."""
    return {
        "primary": primary,
        "tiebreak_1_sequence_balanced_recall": recall,
        "tiebreak_2_macro_f1": macro_f1,
    }


def test_selection_order_matches_what_selection_score_produces() -> None:
    """The ladder is indexed by name; a rename must break here, not go silent."""
    produced = M.selection_score(
        cis={"frame_f2": 0.5, "sequence_balanced_recall": 0.75},
        trans={"frame_f2": 0.25, "sequence_balanced_recall": 0.25},
        macro_f1=0.44,
    )
    assert set(M.SELECTION_ORDER) <= set(produced), (
        "the comparator reads keys selection_score does not produce"
    )
    assert M.selection_key(produced) == pytest.approx((0.375, 0.5, 0.44))
    assert M.SELECTION_ORDER[0] == "primary", "F2 is the primary, not a tie-break"


def test_primary_decides_before_any_tiebreak() -> None:
    """A better primary wins even when both tie-breaks are worse. Order matters."""
    incumbent = score(primary=0.40, recall=0.95, macro_f1=0.90)
    candidate = score(primary=0.41, recall=0.10, macro_f1=0.10)

    assert M.is_better_checkpoint(candidate, incumbent)
    assert not M.is_better_checkpoint(incumbent, candidate)


def test_equal_primary_selects_higher_sequence_balanced_recall() -> None:
    """Issue #12's first tie-break: F2 ties, recall separates.

    This is the case the old `score["primary"] > best["score"]` got wrong: it kept the
    incumbent, so the epoch that caught more of the visits was discarded.
    """
    incumbent = score(primary=0.4280, recall=0.61, macro_f1=0.90)
    candidate = score(primary=0.4280, recall=0.62, macro_f1=0.10)

    assert M.is_better_checkpoint(candidate, incumbent), (
        "equal F2: the checkpoint that catches more visits wins, and macro F1 — which "
        "is below it on the ladder — does not get to overrule that"
    )
    assert not M.is_better_checkpoint(incumbent, candidate)


def test_equal_primary_and_recall_select_higher_macro_f1() -> None:
    """Issue #12's second tie-break: only macro F1 is left to separate them."""
    incumbent = score(primary=0.4280, recall=0.6150, macro_f1=0.512)
    candidate = score(primary=0.4280, recall=0.6150, macro_f1=0.513)

    assert M.is_better_checkpoint(candidate, incumbent)
    assert not M.is_better_checkpoint(incumbent, candidate)


def test_an_exact_tie_keeps_the_earlier_epoch() -> None:
    """The final tie-break, and it must be a rule rather than an accident.

    Two checkpoints equal on all three values are indistinguishable to the rule, so
    something has to decide. Strict `>` keeps the incumbent, and epochs are offered in
    ascending order, so the earliest wins: it reached that score with less training and
    is the one to defend.
    """
    identical = score(primary=0.4280, recall=0.6150, macro_f1=0.5120)

    assert not M.is_better_checkpoint(identical, dict(identical)), (
        "an equal candidate must not displace the incumbent"
    )

    # The loop in train.run(), reduced to the part under test.
    best: dict = {"score": None, "epoch": -1}
    for epoch in range(5):
        if M.is_better_checkpoint(dict(identical), best["score"]):
            best = {"score": dict(identical), "epoch": epoch}

    assert best["epoch"] == 0, "five identical epochs must select the first"


def test_the_first_offered_checkpoint_is_always_taken() -> None:
    """No sentinel score: phase B's first epoch has nothing to beat.

    The old code seeded `best` with -1.0, which works only because F2 cannot be
    negative — a sentinel that happens to be below the metric's range. None says the
    same thing without depending on that.
    """
    first = score(primary=0.0, recall=0.0, macro_f1=0.0)
    assert M.is_better_checkpoint(first, None), (
        "a run whose every epoch scores zero must still write a checkpoint"
    )


def test_the_comparator_is_a_strict_total_order_on_the_vector() -> None:
    """Deterministic: exactly one of a>b, b>a, tie — never both, never neither-with-a-diff."""
    vectors = [
        score(0.40, 0.60, 0.50),
        score(0.40, 0.60, 0.51),
        score(0.40, 0.61, 0.50),
        score(0.41, 0.59, 0.49),
    ]
    for a in vectors:
        for b in vectors:
            forward = M.is_better_checkpoint(a, b)
            backward = M.is_better_checkpoint(b, a)
            assert not (forward and backward), "both cannot win"
            if M.selection_key(a) != M.selection_key(b):
                assert forward or backward, "different vectors must be separable"
            else:
                assert not forward and not backward


def test_a_missing_tiebreak_value_is_an_error_not_a_default() -> None:
    """A score dict without the tie-breaks means the caller computed the wrong thing.

    Defaulting the absent value to 0.0 would silently rank a truncated score below a
    complete one and produce a plausible, wrong selection.
    """
    with pytest.raises(KeyError):
        M.is_better_checkpoint({"primary": 0.9}, score(0.4, 0.6, 0.5))
