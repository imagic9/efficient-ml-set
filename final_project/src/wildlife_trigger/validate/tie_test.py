#!/usr/bin/env python3
"""Is the gap between two arms real, or is it the validation sample?

PLAN C1a: *"prefer 256x192 when statistically tied"*. That rule needs something to
decide *tied* with, and a pair of point estimates cannot — 0.4280 against 0.4275 and
0.4280 against 0.2000 are both "higher".

So this resamples the validation set and asks how often the ranking survives.

**The resampling unit is the sequence.** CCT frames arrive in bursts: a camera fires a
burst of frames seconds apart at one animal at one location. Frames inside a sequence
are near-duplicates, so treating them as independent would count bobcat's 315 sequences
as 937 observations and shrink the interval by roughly the square root of that ratio —
producing a confident-looking interval that is mostly an artefact of burst photography.

**The comparison is paired.** Both arms are resampled on the *same* sequences in each
replicate, so the shared difficulty of a given draw cancels. Unpaired intervals on each
arm separately would be wider than the difference deserves, and overlapping intervals
would then be misread as a tie — a real difference could be thrown away.

**Why not `metrics.bootstrap_sequence_clusters`.** That resamples the same way, but it
reports one arm's interval on one split, and two such intervals cannot answer this
question: unpaired intervals on each arm are wide enough to overlap even when every
single paired draw ranks the arms the same way, and the overlap would be read as a tie.
The pairing is the point, so the difference is what gets resampled.

**What this does not cover.** This is sampling uncertainty — *would this ranking hold on
other cameras?* It says nothing about training noise: each arm is one seed, and the
selection score's own epoch-to-epoch swing is large (arm 2's trans F2 moved 0.1251 →
0.2684 → 0.1310 on consecutive epochs). A tie here means the validation data cannot
separate the arms; it does not mean a rerun would land in the same place. Separating
that needs seeds, which PLAN C1a deliberately does not spend.

Usage:
    python -m wildlife_trigger.validate.tie_test \
        --a results/ablations/c1a_empty5k_16out_256x192/predictions.npz \
        --b results/ablations/c1a_empty5k_16out_224x224/predictions.npz \
        --output results/ablations/input_tie_test.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..metrics import PRIMARY_METRIC, target_presence_metrics
from ..metrics import selection_score as design_selection_score

TARGET_CLASS = "bobcat"
SPLITS = ("cis_val_clean", "trans_val")
DEFAULT_REPLICATES = 10_000
DEFAULT_THRESHOLD = 0.5


def load(path: Path) -> dict:
    raw = np.load(path, allow_pickle=False)
    class_names = [str(n) for n in raw["class_names"]]

    # Resolve the target column by NAME, per arm. DESIGN §5.2's no-empty arm drops
    # `empty` from the head, which shifts every index after it — a hardcoded bobcat
    # index would silently read a different class's score for that arm.
    target = class_names.index(TARGET_CLASS)

    data = {"run_name": str(raw["run_name"]), "class_names": class_names}
    for split in SPLITS:
        data[split] = {
            "scores": raw[f"{split}/probabilities"][:, target],
            "present": raw[f"{split}/present"][:, target],
            "seq_ids": [str(s) for s in raw[f"{split}/seq_ids"]],
            "image_ids": [str(s) for s in raw[f"{split}/image_ids"]],
        }
    return data


def sequence_groups(seq_ids: list[str]) -> list[np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, seq in enumerate(seq_ids):
        groups[seq].append(index)
    return [np.array(v) for v in groups.values()]


def metrics_of(split: dict, rows: np.ndarray, threshold: float) -> dict:
    seq_ids = [split["seq_ids"][i] for i in rows]
    return target_presence_metrics(
        split["scores"][rows], split["present"][rows], seq_ids, threshold
    )


def selection_score(arm: dict, draws: dict[str, np.ndarray], threshold: float) -> float:
    """DESIGN §7.2's score, taken from `metrics.selection_score` rather than restated.

    Restating it as `mean(cis_f2, trans_f2)` here would be correct today and wrong the
    day DESIGN's score changes, in a tool whose entire output is a comparison of that
    score — and the day came: the primary moved to AP under the 2026-07-16 amendment
    and back under its pre-registered verdict, and both times this function changed
    what it feeds in, not what it computes. `macro_f1` feeds only the second tie-break,
    which this never reads.
    """
    cis, trans = (metrics_of(arm[s], draws[s], threshold) for s in SPLITS)
    return float(design_selection_score(cis, trans, macro_f1=0.0)["primary"])


def check_pairable(a: dict, b: dict) -> None:
    """Refuse to pair two arms that did not see the same frames.

    Pairing subtracts one arm's score from the other's on a shared draw. If the arms
    were evaluated on different frames, that subtraction is meaningless and the
    interval it produces would be authoritative-looking nonsense.
    """
    for split in SPLITS:
        if a[split]["image_ids"] != b[split]["image_ids"]:
            raise RuntimeError(
                f"{split}: the two runs were evaluated on different frames (or in a "
                f"different order), so they cannot be paired. "
                f"{a['run_name']} has {len(a[split]['image_ids'])} frames, "
                f"{b['run_name']} has {len(b[split]['image_ids'])}."
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, type=Path, help="baseline predictions.npz")
    parser.add_argument("--b", required=True, type=Path, help="challenger predictions.npz")
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    a, b = load(args.a), load(args.b)
    check_pairable(a, b)

    groups = {s: sequence_groups(a[s]["seq_ids"]) for s in SPLITS}
    rng = np.random.default_rng(args.seed)

    observed_a = selection_score(a, {s: np.arange(len(a[s]["seq_ids"])) for s in SPLITS}, args.threshold)
    observed_b = selection_score(b, {s: np.arange(len(b[s]["seq_ids"])) for s in SPLITS}, args.threshold)

    differences, scores_a, scores_b = [], [], []
    for _ in range(args.replicates):
        # One draw of sequences, used for BOTH arms. This is the pairing.
        draws = {}
        for split in SPLITS:
            picked = rng.integers(0, len(groups[split]), len(groups[split]))
            draws[split] = np.concatenate([groups[split][i] for i in picked])
        score_a = selection_score(a, draws, args.threshold)
        score_b = selection_score(b, draws, args.threshold)
        scores_a.append(score_a)
        scores_b.append(score_b)
        differences.append(score_b - score_a)

    differences = np.array(differences)
    low, high = np.percentile(differences, [2.5, 97.5])
    tied = bool(low <= 0.0 <= high)

    result = {
        "arm_a": a["run_name"],
        "arm_b": b["run_name"],
        "score": PRIMARY_METRIC,
        "threshold": args.threshold,
        "replicates": args.replicates,
        "seed": args.seed,
        "resampling_unit": "sequence",
        "sequences": {s: len(groups[s]) for s in SPLITS},
        "observed": {
            a["run_name"]: round(observed_a, 4),
            b["run_name"]: round(observed_b, 4),
            "difference_b_minus_a": round(observed_b - observed_a, 4),
        },
        "difference_ci95": [round(float(low), 4), round(float(high), 4)],
        "difference_mean": round(float(differences.mean()), 4),
        "probability_b_beats_a": round(float((differences > 0).mean()), 4),
        "arm_a_ci95": [round(float(x), 4) for x in np.percentile(scores_a, [2.5, 97.5])],
        "arm_b_ci95": [round(float(x), 4) for x in np.percentile(scores_b, [2.5, 97.5])],
        "tied": tied,
        "interpretation": (
            "The 95% CI of the paired difference spans zero: the validation data cannot "
            "distinguish these arms. PLAN C1a's tie-break applies."
            if tied
            else f"The 95% CI of the paired difference excludes zero: "
            f"{b['run_name'] if observed_b > observed_a else a['run_name']} is ahead by "
            "more than the validation sample explains."
        ),
        "caveat": (
            "Sampling uncertainty only. Each arm is a single seed, so this does not "
            "bound training noise; a rerun could land elsewhere within it."
        ),
    }

    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
