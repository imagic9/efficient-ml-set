#!/usr/bin/env python3
"""Evaluation metrics (DESIGN §6.3, §7.2, §12).

Two ideas here are not standard practice and are the reason this file exists.

**Sequence-balanced recall.** Camera traps fire in bursts, so a positive `seq_id` holds
several near-duplicate frames. Frame-level recall therefore weights a 20-frame visit ten
times as heavily as a 2-frame visit, and a model that nails long bursts while missing
short ones looks excellent. Sequence-balanced recall computes frame recall *inside* each
positive sequence and averages those with equal weight, so every visit counts once.
DESIGN §6.3 keeps frame recall as the reported product metric and uses this to choose
the threshold.

**Target presence, not top-1.** The product fires when `softmax[bobcat] >= threshold`,
regardless of whether another logit is larger (DESIGN §6.1). A frame holding a bobcat and
a coyote is a true bobcat frame either way. So every metric here works from the complete
label set, and `top-1 accuracy` is reported for information rather than used to decide
anything.

Uncertainty is bootstrapped over `seq_id` clusters, never over individual frames:
resampling frames treats twenty near-duplicates as twenty independent observations and
produces confidence intervals several times too narrow.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

# DESIGN §12: a class needs real support before its F1 is meaningful.
MACRO_F1_MIN_IMAGES = 20
MACRO_F1_MIN_SEQUENCES = 5

# F-beta with beta=2: recall weighted 4x precision. A missed bobcat is a lost photograph
# the photographer will never know about; a false fire costs a frame of storage.
F_BETA = 2.0

# DESIGN §6.3's fire budget: the device may fire on at most this share of the frames that
# hold no bobcat, in each domain separately. A registered product limit, not a tuned
# number — it is what "spends its shutter on bobcats" means, and it is the only thing
# standing between the recall floor and a trigger that photographs everything.
MAX_FALSE_FIRE_RATE = 0.05


def fbeta(precision: float, recall: float, beta: float = F_BETA) -> float:
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    b2 = beta * beta
    denominator = b2 * precision + recall
    return 0.0 if denominator == 0 else (1 + b2) * precision * recall / denominator


def target_presence_metrics(
    scores: np.ndarray,
    present: np.ndarray,
    seq_ids: list[str],
    threshold: float,
) -> dict:
    """Frame and sequence metrics for one target at one threshold.

    `scores` is softmax[target] per frame; `present` is 1 where the target is in the
    frame's complete label set.
    """
    fired = scores >= threshold
    positives = present > 0

    true_positive = int((fired & positives).sum())
    false_positive = int((fired & ~positives).sum())
    false_negative = int((~fired & positives).sum())

    precision = true_positive / (true_positive + false_positive) if fired.any() else 0.0
    recall = true_positive / positives.sum() if positives.any() else 0.0

    # Sequence-balanced recall: recall within each positive sequence, averaged with
    # equal weight per sequence.
    per_sequence: dict[str, list[bool]] = defaultdict(list)
    for index in np.flatnonzero(positives):
        per_sequence[seq_ids[index]].append(bool(fired[index]))
    sequence_recalls = [np.mean(hits) for hits in per_sequence.values()]

    # Event capture rate: positive sequences with at least one trigger. This is the
    # product question — did we photograph the visit at all?
    captured = sum(1 for hits in per_sequence.values() if any(hits))

    negatives = int((~positives).sum())
    return {
        "threshold": float(threshold),
        "frame_recall": float(recall),
        "frame_precision": float(precision),
        "frame_f2": fbeta(precision, recall),
        "sequence_balanced_recall": float(np.mean(sequence_recalls)) if sequence_recalls else 0.0,
        "positive_sequences": len(per_sequence),
        "event_capture_rate": captured / len(per_sequence) if per_sequence else 0.0,
        "false_fire_rate": false_positive / negatives if negatives else 0.0,
        "fire_rate": float(fired.mean()) if len(fired) else 0.0,
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "positive_frames": int(positives.sum()),
        "negative_frames": negatives,
    }


def per_class_metrics(
    probabilities: np.ndarray,
    present: np.ndarray,
    class_names: list[str],
    seq_ids: list[str],
) -> dict:
    """Top-1 per-class support/recall/precision/F1, plus support-aware macro F1."""
    predicted = probabilities.argmax(axis=1)

    per_class = {}
    for index, name in enumerate(class_names):
        positives = present[:, index] > 0
        chosen = predicted == index
        true_positive = int((chosen & positives).sum())
        precision = true_positive / chosen.sum() if chosen.any() else 0.0
        recall = true_positive / positives.sum() if positives.any() else 0.0
        sequences = len({seq_ids[i] for i in np.flatnonzero(positives)})
        per_class[name] = {
            "support_images": int(positives.sum()),
            "support_sequences": sequences,
            "recall": float(recall),
            "precision": float(precision),
            "f1": fbeta(precision, recall, beta=1.0),
        }

    # Macro F1 over classes with real support only. Averaging in a class with one
    # validation image makes the macro number a coin flip on that image.
    included = [
        name
        for name, m in per_class.items()
        if m["support_images"] >= MACRO_F1_MIN_IMAGES
        and m["support_sequences"] >= MACRO_F1_MIN_SEQUENCES
    ]
    macro_f1 = float(np.mean([per_class[n]["f1"] for n in included])) if included else 0.0

    return {
        "per_class": per_class,
        "support_aware_macro_f1": macro_f1,
        "macro_f1_included_classes": included,
        "macro_f1_thresholds": {
            "min_images": MACRO_F1_MIN_IMAGES,
            "min_sequences": MACRO_F1_MIN_SEQUENCES,
        },
        "top1_accuracy": float(
            np.mean([present[i, predicted[i]] > 0 for i in range(len(predicted))])
        ),
    }


def trade_off_curve(rows: list[dict], points: int = 200) -> list[dict]:
    """The recall/false-fire trade-off, thinned to `points` (DESIGN §6.3 step 5).

    Thinned by rank through the candidate list rather than by even steps in score, so
    the curve is dense exactly where the scores are — which is where the operating
    point lives. The chosen threshold is added by the caller; it must be on the curve
    whether or not the thinning happens to keep it.
    """
    if len(rows) > points:
        keep = [rows[i] for i in np.unique(np.linspace(0, len(rows) - 1, points).astype(int))]
    else:
        keep = list(rows)

    return [
        {
            "threshold": r["threshold"],
            "admissible": r["admissible"],
            "min_sequence_balanced_recall": r["min_sequence_balanced_recall"],
            "max_false_fire_rate": r["max_false_fire_rate"],
            "mean_frame_f2": r["mean_frame_f2"],
            "per_domain": {
                domain: {
                    key: m[key]
                    for key in (
                        "sequence_balanced_recall",
                        "frame_recall",
                        "false_fire_rate",
                        "fire_rate",
                        "frame_f2",
                    )
                }
                for domain, m in r["per_domain"].items()
            },
        }
        for r in keep
    ]


def select_threshold(
    scores_by_domain: dict[str, tuple[np.ndarray, np.ndarray, list[str]]],
    min_sequence_recall: float = 0.90,
    max_false_fire_rate: float = MAX_FALSE_FIRE_RATE,
    curve_points: int = 200,
) -> dict:
    """DESIGN §6.3's threshold rule: the recall floor, spent inside the fire budget.

    1. search all unique observed scores and keep the **admissible** ones — false-fire
       rate <= the budget on *every* domain;
    2. choose the LARGEST admissible threshold whose sequence-balanced recall is >= 90%
       on BOTH cis-val-clean and trans-val. Only this is the primary rule satisfied;
    3. if none reaches the floor, the model does not meet the rule: return
       `recall_floor_infeasible`, ship the admissible threshold maximising mean frame
       F2, and record the recall each domain actually reached;
    4. if nothing is admissible, return `fire_budget_infeasible` and name no operating
       point.

    Largest, not best: among thresholds that meet the floor, the highest one fires
    least often, and every unnecessary fire is a wasted frame. The floor is the
    requirement; F2 does not get to trade it away.

    **The budget is what makes the floor mean anything** (issue #11). The candidates are
    the observed scores, so the smallest fires on every frame and scores 100% recall by
    construction: constrain recall alone and the floor is *always* satisfiable by
    photographing everything. The old `non_trivial` guard rejected only a threshold
    firing on literally every frame, and 78% of frames at a 67.6% false-fire rate
    cleared it while being useless as a shutter trigger.

    **The fallback needs the budget more than the primary rule does.** Mean F2 weights
    recall 4x, and 46% of trans-val's frames are true bobcats — so on trans-val, F2 is
    maximised by firing on everything. Unconstrained, step 3 walks straight into the
    operating point step 2 was rescued from.

    Both domains separately, because trans-val holds far more bobcats than
    cis-val-clean and a pooled constraint would let trans-val's 793 hide a cis failure.
    False-fire rather than fire rate for the same reason in reverse: false-fire is
    conditioned on the negatives, so it means the same thing in both domains, while a
    fire-rate ceiling would be unmeetable on trans-val — where firing on nearly half the
    frames is the device working.
    """
    candidates = np.unique(
        np.concatenate([scores for scores, _, _ in scores_by_domain.values()])
    )

    rows = []
    for threshold in candidates:
        per_domain = {
            domain: target_presence_metrics(scores, present, seqs, threshold)
            for domain, (scores, present, seqs) in scores_by_domain.items()
        }
        max_false_fire = max(m["false_fire_rate"] for m in per_domain.values())
        rows.append(
            {
                "threshold": float(threshold),
                "per_domain": per_domain,
                "min_sequence_balanced_recall": min(
                    m["sequence_balanced_recall"] for m in per_domain.values()
                ),
                "max_false_fire_rate": float(max_false_fire),
                "mean_frame_f2": float(
                    np.mean([m["frame_f2"] for m in per_domain.values()])
                ),
                # DESIGN §6.3's fire budget, on every domain rather than on average: a
                # device that behaves at one camera and fires blindly at the next has
                # not met the budget, it has averaged it away.
                "admissible": bool(max_false_fire <= max_false_fire_rate),
                # Subsumed by the budget — 100% false-fire is not <= 5% — and kept
                # because it names the degenerate case the guard used to be about.
                "trivial": any(m["fire_rate"] >= 1.0 for m in per_domain.values()),
            }
        )

    admissible = [r for r in rows if r["admissible"]]
    meeting = [
        r for r in admissible if r["min_sequence_balanced_recall"] >= min_sequence_recall
    ]

    if meeting:
        chosen = max(meeting, key=lambda r: r["threshold"])
        status = "primary_rule_met"
        rule = (
            f"primary: largest threshold within the {max_false_fire_rate:.0%} false-fire "
            f"budget with sequence-balanced recall >= {min_sequence_recall:.0%} on both "
            "domains"
        )
        unmet = None
    elif admissible:
        chosen = max(admissible, key=lambda r: r["mean_frame_f2"])
        status = "recall_floor_infeasible"
        rule = (
            f"fallback: no threshold inside the {max_false_fire_rate:.0%} false-fire "
            f"budget reached the {min_sequence_recall:.0%} sequence-balanced recall "
            "floor on both domains; maximised mean frame F2 within the budget. The "
            "primary rule is NOT satisfied"
        )
        unmet = {
            domain: chosen["per_domain"][domain]["sequence_balanced_recall"]
            for domain in scores_by_domain
            if chosen["per_domain"][domain]["sequence_balanced_recall"]
            < min_sequence_recall
        }
    else:
        # Pathological: the largest observed score fires on at most a handful of frames,
        # so something is admissible unless the model scores every negative above every
        # candidate. Kept as a branch because "cannot happen" is how a rule acquires a
        # silent default.
        chosen = max(rows, key=lambda r: r["threshold"])
        status = "fire_budget_infeasible"
        rule = (
            f"infeasible: no threshold meets the {max_false_fire_rate:.0%} false-fire "
            "budget on both domains. This model has no deployable operating point"
        )
        unmet = {
            domain: chosen["per_domain"][domain]["false_fire_rate"]
            for domain in scores_by_domain
            if chosen["per_domain"][domain]["false_fire_rate"] > max_false_fire_rate
        }

    curve = trade_off_curve(rows, curve_points)
    if not any(point["threshold"] == chosen["threshold"] for point in curve):
        curve = sorted(
            curve + trade_off_curve([chosen], 1), key=lambda p: p["threshold"]
        )

    return {
        "threshold": chosen["threshold"],
        "status": status,
        # The one boolean C3 and the report must read. `rule` is prose for a human, and
        # a caller that greps it for "primary" would call the fallback a pass.
        "primary_rule_met": status == "primary_rule_met",
        "rule": rule,
        "min_sequence_recall_required": min_sequence_recall,
        "max_false_fire_rate_allowed": max_false_fire_rate,
        "unmet_constraint": unmet,
        "chosen_is_trivial": chosen["trivial"],
        "chosen_is_admissible": chosen["admissible"],
        "per_domain": chosen["per_domain"],
        "candidates_searched": len(candidates),
        "admissible_candidates": len(admissible),
        "non_trivial_candidates": sum(1 for r in rows if not r["trivial"]),
        # DESIGN §6.3 step 5: a constrained result must be readable as a position on a
        # curve, not inferable from a verdict.
        "recall_false_fire_curve": curve,
    }


def bootstrap_sequence_clusters(
    scores: np.ndarray,
    present: np.ndarray,
    seq_ids: list[str],
    threshold: float,
    metric: str = "frame_f2",
    replicates: int = 1000,
    seed: int = 42,
) -> dict:
    """95% interval by resampling whole `seq_id` clusters.

    Clusters, not frames. Twenty near-duplicate burst frames carry roughly one
    sequence's worth of information; resampling them independently would report an
    interval several times narrower than the evidence supports.
    """
    rng = np.random.default_rng(seed)
    by_sequence: dict[str, list[int]] = defaultdict(list)
    for index, seq_id in enumerate(seq_ids):
        by_sequence[seq_id].append(index)
    sequences = list(by_sequence)

    values = []
    for _ in range(replicates):
        drawn = rng.choice(len(sequences), size=len(sequences), replace=True)
        indices = [i for d in drawn for i in by_sequence[sequences[d]]]
        sample = target_presence_metrics(
            scores[indices], present[indices], [seq_ids[i] for i in indices], threshold
        )
        values.append(sample[metric])

    return {
        "metric": metric,
        "point_estimate": target_presence_metrics(scores, present, seq_ids, threshold)[metric],
        "ci95_low": float(np.percentile(values, 2.5)),
        "ci95_high": float(np.percentile(values, 97.5)),
        "replicates": replicates,
        "resampled": "seq_id clusters, not frames",
    }


def selection_score(cis: dict, trans: dict, macro_f1: float) -> dict:
    """DESIGN §7.2's checkpoint selection score.

    Mean bobcat F2 across the two validation domains, sequence-balanced recall as first
    tie-break, support-aware macro F1 as second. Explicitly *not* overall accuracy:
    `empty` dominates the corpus, so accuracy would select the model that is best at
    predicting nothing happened.
    """
    return {
        "primary": (cis["frame_f2"] + trans["frame_f2"]) / 2,
        "tiebreak_1_sequence_balanced_recall": (
            cis["sequence_balanced_recall"] + trans["sequence_balanced_recall"]
        )
        / 2,
        "tiebreak_2_macro_f1": macro_f1,
    }


# The order DESIGN §7.2 ranks on: primary first, then each tie-break in turn.
SELECTION_ORDER = (
    "primary",
    "tiebreak_1_sequence_balanced_recall",
    "tiebreak_2_macro_f1",
)


def selection_key(score: dict) -> tuple[float, ...]:
    """`selection_score`'s dict as the vector the rule compares."""
    return tuple(float(score[key]) for key in SELECTION_ORDER)


def is_better_checkpoint(candidate: dict, incumbent: dict | None) -> bool:
    """Does `candidate` win under the whole of DESIGN §7.2, tie-breaks included?

    Comparing `primary` alone silently discards the two tie-breaks the rule declares,
    and the values are not floats drawn from a continuum: F2, sequence-balanced recall
    and macro F1 are all ratios of finite frame and sequence counts, so exact equality
    is an ordinary event rather than a measure-zero curiosity. Two epochs that tie on
    F2 and differ on recall are precisely what the tie-breaks exist to separate.

    Tuple comparison *is* the lexicographic rule — `(0.5, 0.9) > (0.5, 0.8)` — so the
    order lives in `SELECTION_ORDER` and nowhere else.

    Strict `>` is the final tie-break: a candidate equal on all three does not displace
    the incumbent, and since epochs are offered in ascending order that keeps the
    earliest. Earliest, because among indistinguishable checkpoints the one that
    reached the score with less training is the one to defend.
    """
    if incumbent is None:
        return True
    return selection_key(candidate) > selection_key(incumbent)
