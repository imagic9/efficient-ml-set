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


def select_threshold(
    scores_by_domain: dict[str, tuple[np.ndarray, np.ndarray, list[str]]],
    min_sequence_recall: float = 0.90,
) -> dict:
    """DESIGN §6.3's primary threshold rule.

    1. search all unique observed scores;
    2. choose the LARGEST **non-trivial** threshold whose sequence-balanced recall is
       >= 90% on BOTH cis-val-clean and trans-val;
    3. if none exists, maximise mean frame-level F2 and record which constraint failed.

    Largest, not best: among thresholds that meet the recall floor, the highest one fires
    least often, and every unnecessary fire is a wasted frame. The floor is the
    requirement; F2 does not get to trade it away.

    **"Non-trivial" is load-bearing and easy to drop.** The candidates are the observed
    scores, so the smallest of them fires on every frame and scores 100% recall by
    construction — the floor is therefore *always* satisfiable, and without this
    qualifier step 3 would be unreachable dead code and the rule could return a trigger
    that photographs everything. A threshold that fires on every frame in a domain
    discriminates nothing, so it does not count as meeting the constraint.

    Both domains separately, because trans-val holds far more bobcats than
    cis-val-clean and a pooled constraint would let trans-val's 793 hide a cis failure.
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
        rows.append(
            {
                "threshold": float(threshold),
                "per_domain": per_domain,
                "min_sequence_balanced_recall": min(
                    m["sequence_balanced_recall"] for m in per_domain.values()
                ),
                "mean_frame_f2": float(
                    np.mean([m["frame_f2"] for m in per_domain.values()])
                ),
                # Trivial = fires on every frame of some domain. Such a "trigger"
                # separates nothing; it is not an operating point.
                "trivial": any(m["fire_rate"] >= 1.0 for m in per_domain.values()),
            }
        )

    meeting = [
        r
        for r in rows
        if r["min_sequence_balanced_recall"] >= min_sequence_recall and not r["trivial"]
    ]
    if meeting:
        chosen = max(meeting, key=lambda r: r["threshold"])
        rule = (
            "primary: largest non-trivial threshold with sequence-balanced recall "
            f">= {min_sequence_recall:.0%} on both domains"
        )
        unmet = None
    else:
        # Non-trivial candidates only: falling back to a fire-on-everything threshold
        # because it maximises F2 would be worse than the rule it replaced.
        non_trivial = [r for r in rows if not r["trivial"]] or rows
        chosen = max(non_trivial, key=lambda r: r["mean_frame_f2"])
        rule = (
            f"fallback: no non-trivial threshold met the {min_sequence_recall:.0%} "
            "sequence-balanced recall floor on both domains; maximised mean frame F2"
        )
        unmet = {
            domain: chosen["per_domain"][domain]["sequence_balanced_recall"]
            for domain in scores_by_domain
            if chosen["per_domain"][domain]["sequence_balanced_recall"] < min_sequence_recall
        }

    return {
        "threshold": chosen["threshold"],
        "rule": rule,
        "min_sequence_recall_required": min_sequence_recall,
        "unmet_constraint": unmet,
        "chosen_is_trivial": chosen["trivial"],
        "per_domain": chosen["per_domain"],
        "candidates_searched": len(candidates),
        "non_trivial_candidates": sum(1 for r in rows if not r["trivial"]),
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
