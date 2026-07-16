"""DESIGN §6.3 step 7: the bootstrap of the *selected threshold* (PLAN C3).

Two implementations of one registered rule exist after this change:
`select_threshold` (the slow, fully-documented record producer) and the vectorised
path inside `bootstrap_threshold_selection` (a thousand rule re-runs cannot afford
per-candidate metric dictionaries). The central test here holds them equal on
randomized data — every branch, every tie. If they ever disagree, the bootstrap is
measuring the stability of a rule the project did not register.
"""

from __future__ import annotations

import numpy as np
import pytest

from wildlife_trigger import metrics as M


def random_domains(seed: int) -> dict:
    """Small random two-domain datasets with deliberate score ties.

    Scores are rounded to two decimals so equal scores across frames and across
    domains are common, exercising the tie conventions (`>=` firing, largest-meeting,
    first-of-equal-F2) rather than dodging them with continuous values.
    """
    rng = np.random.default_rng(seed)
    domains = {}
    for name in ("cis", "trans"):
        sequences = rng.integers(6, 14)
        scores, present, seqs = [], [], []
        for s in range(sequences):
            length = int(rng.integers(1, 7))
            positive_sequence = rng.random() < 0.5
            for _ in range(length):
                scores.append(round(float(rng.random()), 2))
                present.append(float(positive_sequence and rng.random() < 0.7))
                seqs.append(f"{name}_seq{s}")
        domains[name] = (np.array(scores), np.array(present), seqs)
    return domains


class TestFastPathEquivalence:
    @pytest.mark.parametrize("seed", range(20))
    def test_fast_path_agrees_with_the_registered_rule(self, seed: int) -> None:
        """Twenty random datasets, every branch reachable: the verdicts must match."""
        domains = random_domains(seed)
        slow = M.select_threshold(domains, min_sequence_recall=0.90)
        fast_threshold, fast_status = M.select_threshold_point(
            domains, min_sequence_recall=0.90
        )
        assert fast_status == slow["status"]
        assert fast_threshold == pytest.approx(slow["threshold"], abs=0.0)

    @pytest.mark.parametrize("floor", [0.0, 0.5, 0.9, 1.0])
    def test_agreement_holds_across_recall_floors(self, floor: float) -> None:
        """Moving the floor moves the branch boundaries; the agreement must not care."""
        domains = random_domains(99)
        slow = M.select_threshold(domains, min_sequence_recall=floor)
        fast_threshold, fast_status = M.select_threshold_point(
            domains, min_sequence_recall=floor
        )
        assert (fast_threshold, fast_status) == (slow["threshold"], slow["status"])

    def test_agreement_on_the_infeasible_branch(self) -> None:
        """Every negative outscores every candidate ceiling: nothing is admissible."""
        scores = [0.9, 0.8] + [0.95] * 30
        present = [1.0, 1.0] + [0.0] * 30
        seqs = ["a", "b"] + [f"n{i}" for i in range(30)]
        domains = {"cis": (np.array(scores), np.array(present), seqs)}
        slow = M.select_threshold(domains, min_sequence_recall=0.90)
        fast = M.select_threshold_point(domains, min_sequence_recall=0.90)
        assert slow["status"] == "fire_budget_infeasible"
        assert fast == (slow["threshold"], slow["status"])


class TestBootstrap:
    def domains(self) -> dict:
        return random_domains(7)

    def test_same_seed_same_distribution(self) -> None:
        """A bootstrap that cannot be reproduced is an anecdote."""
        first = M.bootstrap_threshold_selection(self.domains(), replicates=50, seed=42)
        second = M.bootstrap_threshold_selection(self.domains(), replicates=50, seed=42)
        assert first["thresholds"] == second["thresholds"]
        assert first["statuses"] == second["statuses"]

    def test_interval_brackets_the_median_and_counts_add_up(self) -> None:
        result = M.bootstrap_threshold_selection(self.domains(), replicates=100, seed=1)
        assert result["ci95_low"] <= result["median"] <= result["ci95_high"]
        assert sum(result["statuses"].values()) == 100
        assert len(result["thresholds"]) == 100

    def test_point_estimate_is_the_registered_rule_verdict(self) -> None:
        """The full-data verdict rides inside the result so a caller can cross-check
        it against `select_threshold` — the calibration tool must refuse to write a
        policy if the two paths disagree."""
        domains = self.domains()
        result = M.bootstrap_threshold_selection(domains, replicates=10, seed=3)
        slow = M.select_threshold(domains, min_sequence_recall=0.90)
        assert result["point_estimate"]["threshold"] == pytest.approx(slow["threshold"])
        assert result["point_estimate"]["status"] == slow["status"]

    def test_separable_data_is_stable_under_resampling(self) -> None:
        """A model with a clean margin should keep its verdict in every replicate:
        if this flickers, the bootstrap is adding noise, not measuring it.

        Ten positive sequences among forty negative ones, because support matters:
        with only three, a replicate misses every positive sequence about 5% of the
        time and the *correct* verdict for that replicate is a failed floor — an
        earlier draft of this test built exactly that dataset and then blamed the
        bootstrap for reporting what it resampled."""
        positives = [0.90 + 0.005 * i for i in range(10)]
        scores = positives + [0.05] * 40
        present = [1.0] * 10 + [0.0] * 40
        seqs = [f"p{i}" for i in range(10)] + [f"n{i}" for i in range(40)]
        domain = (np.array(scores), np.array(present), seqs)
        result = M.bootstrap_threshold_selection(
            {"cis": domain, "trans": domain}, replicates=100, seed=5
        )
        assert result["statuses"] == {"primary_rule_met": 100}
        assert result["ci95_low"] >= 0.90

    def test_replicate_candidates_come_from_drawn_sequences_only(self) -> None:
        """The rule searches observed scores; a replicate that did not draw a
        sequence did not observe its scores. One sequence carries a unique score
        far above the rest — replicates that miss it must not select it.

        With 3 sequences, the probability that every one of 60 replicates draws
        sequence `hi` is (1-(2/3)^3)^60 ≈ 6e-8, so a lone 0.99 in the distribution
        is evidence of a leak, not bad luck."""
        scores = [0.99, 0.60, 0.55] + [0.05] * 10
        present = [1.0, 1.0, 1.0] + [0.0] * 10
        seqs = ["hi", "mid_a", "mid_b"] + ["n"] * 10
        domain = (np.array(scores), np.array(present), seqs)
        result = M.bootstrap_threshold_selection(
            {"cis": domain}, replicates=60, seed=11, min_sequence_recall=0.0
        )
        # With min_sequence_recall=0 every admissible threshold meets the floor, so
        # the rule picks the largest admissible candidate: 0.99 when `hi` was drawn,
        # 0.60 or below when it was not.
        assert any(t < 0.99 for t in result["thresholds"]), (
            "some replicate must miss the `hi` sequence and then must not use its score"
        )
