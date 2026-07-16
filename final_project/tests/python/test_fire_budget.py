"""DESIGN §6.3's fire budget and the statuses it produces (issue #11).

The rule used to constrain recall and nothing else. Measured on the C1a arms it duly
returned a threshold of 0.0011 — 90.1% sequence-balanced recall on trans-val, reached by
firing on 77.9% of its frames at a 67.6% false-fire rate — and reported the primary rule
satisfied. Every assertion below is about that: a rule the model cannot meet must say it
cannot meet it, in a field a caller can read.

Each scenario is hand-built to isolate one branch, and the scores are laid out so the
arithmetic is checkable by hand. `sequence_balanced_recall` averages per-sequence recall,
so the `seq_ids` matter as much as the scores.
"""

from __future__ import annotations

import numpy as np
import pytest

from wildlife_trigger import metrics as M


def domain(scores: list[float], present: list[float], seqs: list[str]) -> tuple:
    return (np.array(scores), np.array(present), seqs)


def test_the_budget_is_registered_at_five_percent() -> None:
    """The number is a product decision, so it is pinned rather than inferred.

    If it changes, DESIGN §6.3's justification changes with it and every calibrated
    policy needs regenerating — which is exactly why it should be hard to change by
    accident.
    """
    assert M.MAX_FALSE_FIRE_RATE == 0.05


def test_full_recall_at_eighty_percent_false_fire_is_not_a_pass() -> None:
    """Issue #11's finding, in miniature.

    Two bobcat visits. Catching both needs threshold 0.30, which also fires on sixteen
    of the twenty negatives: 100% sequence recall at an 80% false-fire rate. The old
    rule returned exactly this and reported the primary rule satisfied, because 0.8 is
    not 1.0 and the `non_trivial` guard only rejected 1.0.
    """
    scores = [0.99, 0.30] + [0.40] * 16 + [0.01] * 4
    present = [1.0, 1.0] + [0.0] * 20
    seqs = ["visit_a", "visit_b"] + [f"neg{i}" for i in range(20)]
    domains = {"cis": domain(scores, present, seqs), "trans": domain(scores, present, seqs)}

    result = M.select_threshold(domains, min_sequence_recall=0.90)

    assert result["threshold"] != pytest.approx(0.30), (
        "the threshold that reaches the floor does so by photographing the clearing"
    )
    assert result["primary_rule_met"] is False
    assert result["chosen_is_admissible"] is True
    assert result["per_domain"]["cis"]["false_fire_rate"] <= 0.05, (
        "the shipped threshold stays inside the budget even when the rule fails"
    )

    # The rejected point is on the published curve, marked, rather than deleted from the
    # record: "we could have had 100% recall" is the reader's first question.
    hollow = [
        p
        for p in result["recall_false_fire_curve"]
        if p["min_sequence_balanced_recall"] >= 0.90
    ]
    assert hollow, "the curve must still show what the floor would have cost"
    assert all(not p["admissible"] for p in hollow)


def test_recall_floor_infeasible_ships_the_best_admissible_point_and_says_so() -> None:
    """The expected C3 branch: budget met, floor unreachable.

    One visit is separable, the other scores below the negatives and cannot be caught
    inside the budget. The rule must ship the best point it can *while reporting
    failure*: a constrained success is not a success.
    """
    scores = [0.90, 0.85, 0.02] + [0.10] * 40
    present = [1.0, 1.0, 1.0] + [0.0] * 40
    seqs = ["visit_a", "visit_a", "visit_b"] + [f"neg{i}" for i in range(40)]
    domains = {"cis": domain(scores, present, seqs), "trans": domain(scores, present, seqs)}

    result = M.select_threshold(domains, min_sequence_recall=0.90)

    assert result["status"] == "recall_floor_infeasible"
    assert result["primary_rule_met"] is False
    assert result["threshold"] == pytest.approx(0.85), "best admissible F2, not best F2"
    assert result["unmet_constraint"]["cis"] == pytest.approx(0.5), (
        "the recall actually reached is recorded, not just the fact of failure"
    )
    assert result["per_domain"]["cis"]["false_fire_rate"] <= 0.05
    assert "NOT satisfied" in result["rule"]


def test_without_the_budget_both_branches_walk_into_shooting_the_world() -> None:
    """Why the ceiling is the fix and a tighter `non_trivial` would not have been.

    46 of 100 frames hold a bobcat, as on trans-val. The lowest candidate fires on
    everything, which gives recall 1.0 — so it satisfies the floor by construction — and
    precision 0.46, which on an F2 that weights recall 4x is a *good* score. So with no
    budget the primary rule and the F2 fallback agree on the same useless point.

    Only 10 of the 46 positives are separable, so inside the budget the floor is out of
    reach and the rule falls back — to a threshold that fires on a tenth of the frames.
    """
    scores = [0.95] * 10 + [0.05] * 36 + [0.90] * 2 + [0.10] * 52
    present = [1.0] * 46 + [0.0] * 54
    seqs = [f"visit{i // 2}" for i in range(46)] + [f"neg{i}" for i in range(54)]
    busy = domain(scores, present, seqs)

    unbudgeted = M.select_threshold(
        {"cis": busy, "trans": busy}, min_sequence_recall=0.90, max_false_fire_rate=1.0
    )
    assert unbudgeted["primary_rule_met"] is True, "the old rule called this a pass"
    assert unbudgeted["per_domain"]["cis"]["fire_rate"] == 1.0
    assert unbudgeted["per_domain"]["cis"]["false_fire_rate"] == 1.0

    budgeted = M.select_threshold({"cis": busy, "trans": busy}, min_sequence_recall=0.90)
    assert budgeted["status"] == "recall_floor_infeasible"
    assert budgeted["per_domain"]["cis"]["false_fire_rate"] <= 0.05
    assert budgeted["per_domain"]["cis"]["fire_rate"] == pytest.approx(0.10)


def test_a_clean_model_still_meets_the_primary_rule() -> None:
    """The budget must not make the rule unsatisfiable in principle.

    A model that separates its classes catches every visit while firing on no negative.
    If this fails, the ceiling is not a product limit — it is a wall.
    """
    scores = [0.95, 0.91, 0.88] + [0.01] * 40
    present = [1.0, 1.0, 1.0] + [0.0] * 40
    seqs = ["a", "b", "c"] + [f"neg{i}" for i in range(40)]
    domains = {"cis": domain(scores, present, seqs), "trans": domain(scores, present, seqs)}

    result = M.select_threshold(domains, min_sequence_recall=0.90)

    assert result["status"] == "primary_rule_met"
    assert result["unmet_constraint"] is None
    assert result["threshold"] == pytest.approx(0.88), (
        "largest, not best: among thresholds meeting the floor, the highest fires least"
    )


def test_the_budget_binds_per_domain_and_is_not_averaged_away() -> None:
    """A device that behaves at one camera and fires blindly at the next has not passed.

    Both domains hold the same two visits. cis's negatives score 0.01 and trans's score
    0.92, so threshold 0.90 catches every visit and fires on nothing in cis — and on
    every negative in trans. Averaged, that is a 50% false-fire rate and it would pass a
    mean ceiling; per-domain it is rejected, and the rule drops to the one threshold
    that is clean on both.
    """
    seqs = ["a", "b"] + [f"n{i}" for i in range(20)]
    present = [1.0, 1.0] + [0.0] * 20
    clean = domain([0.95, 0.90] + [0.01] * 20, present, seqs)
    noisy = domain([0.95, 0.90] + [0.92] * 20, present, seqs)

    result = M.select_threshold({"cis": clean, "trans": noisy}, min_sequence_recall=0.90)

    assert result["threshold"] == pytest.approx(0.95), (
        "0.90 meets the floor and the budget on cis alone; trans is what rejects it"
    )
    assert result["status"] == "recall_floor_infeasible"
    assert result["per_domain"]["trans"]["false_fire_rate"] <= 0.05, (
        "the budget must hold on the domain that struggles, not on the mean"
    )


def test_the_curve_is_published_and_carries_the_chosen_point() -> None:
    """DESIGN §6.3 step 5: a hollow pass must be readable, not inferable.

    The verdict alone cannot show how far from the floor the model was, or what recall a
    slightly larger budget would have bought. The curve is what a reader argues with.
    """
    scores = [0.90, 0.85, 0.02] + [0.10] * 40
    present = [1.0, 1.0, 1.0] + [0.0] * 40
    seqs = ["a", "a", "b"] + [f"neg{i}" for i in range(40)]
    domains = {"cis": domain(scores, present, seqs), "trans": domain(scores, present, seqs)}

    result = M.select_threshold(domains, min_sequence_recall=0.90, curve_points=3)
    curve = result["recall_false_fire_curve"]

    assert any(p["threshold"] == result["threshold"] for p in curve), (
        "the point that was chosen must appear on the curve that justifies it"
    )
    for point in curve:
        assert {"min_sequence_balanced_recall", "max_false_fire_rate", "admissible"} <= set(point)
        assert "false_fire_rate" in point["per_domain"]["cis"]

    thresholds = [p["threshold"] for p in curve]
    assert thresholds == sorted(thresholds), "a curve is read in order"

    # Both quantities are monotone in the threshold: a frame that fires at t fires at
    # every smaller t. If this breaks, the admissible set is not an interval and
    # "largest threshold meeting the floor" stops meaning what it says.
    recalls = [p["min_sequence_balanced_recall"] for p in curve]
    false_fires = [p["max_false_fire_rate"] for p in curve]
    assert recalls == sorted(recalls, reverse=True)
    assert false_fires == sorted(false_fires, reverse=True)
