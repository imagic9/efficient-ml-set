#!/usr/bin/env python3
"""D6 — build and freeze `benchmark_val_1000.jsonl` (DESIGN §12.2, PLAN D6).

No earlier task owned this file, yet E7 packages it and F4 runs the **mandatory**
Pi-vs-gx10 parity on it. That parity is what licenses evaluating full test
accuracy on gx10 instead of on the Pi, so the manifest is built to test exactly
that claim rather than to be a representative sample.

The load-bearing stratum is **threshold-adjacent**: frames whose M0 bobcat score
sits within `eps` of M0's operating point. A hardware numeric difference between
GB10 (SVE2) and the Pi (NEON) can only change `SHUTTER_TRIGGER` on a frame whose
score is near the threshold — so a benchmark subset that omits those frames can
pass while proving nothing. They are included in full and thereby **over-sampled
far beyond their natural frequency** (§12.2). The other strata — bobcat, empty,
multi-label, rare-animal, preprocessing-edge — give coverage "where available";
CCT-20's validation frames are geometrically uniform (all ≈1.37 aspect), so the
preprocessing-edge stratum is legitimately empty and recorded as such rather than
faked.

Built once from **M0's** operating point and scores, the manifest is fixed and
**identical for every model, M0-FP32 included** (§12.2): the point is one ordered
list of frames every runtime is measured on. Validation only — cis-test and
trans-test stay sealed (DESIGN §5.4).

Usage (gx10):
    python -m wildlife_trigger.data.benchmark_manifest \
        --m0-predictions results/optimize/m1_ptq/m0_fp32_reference/predictions.npz \
        --m0-policy artifacts/policies/bobcat_v1.json \
        --output data/manifests/benchmark_val_1000.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..runs import sha256_file

# Validation splits only. §5.4 seals the test sets; a benchmark builder that
# could be pointed at them by a flag is a leak waiting for a tired evening.
VALIDATION_SPLITS = ("cis_val_clean", "trans_val")

# Rare animal tail by validation support: bird 67, skunk 91, rodent 135 frames,
# badger 1 — plus deer/fox (0 validation positives, listed so the intent is on
# the record even though they cannot be sampled here).
RARE_CLASSES = frozenset({"bird", "skunk", "rodent", "badger", "deer", "fox"})

TOTAL = 1000

# Priority order: each frame is accounted to its FIRST matching stratum, so the
# counts partition the pool. threshold-adjacent first because it is the one that
# licenses the whole gx10-accuracy shortcut; bobcat next as the target; then the
# rarer semantic/coverage strata; empty and "other" absorb the bulk.
STRATA = (
    "threshold_adjacent",
    "bobcat",
    "multi_label",
    "rare",
    "preprocessing_edge",
    "empty",
    "other",
)

# Take-all (None) for the small, must-be-seen strata; a cap for the large ones.
# The remainder up to TOTAL is filled from "other" (and any capped overflow) by
# a seeded draw, so the total is exactly TOTAL and reproducible.
TARGETS: dict[str, int | None] = {
    "threshold_adjacent": None,  # all — the over-sampled decision-flippers
    "bobcat": 250,
    "multi_label": None,  # all — rare and semantically important
    "rare": 150,
    "preprocessing_edge": None,  # all (0 available in CCT-20 val)
    "empty": 250,
    "other": 0,  # only via fill
}


def load_pool(manifests_dir: Path) -> list[dict]:
    """Every validation frame, tagged with its source split, in a stable order."""
    pool = []
    for split in VALIDATION_SPLITS:
        for line in (manifests_dir / f"{split}.jsonl").read_text().splitlines():
            if not line:
                continue
            record = json.loads(line)
            record["source_split"] = split
            pool.append(record)
    pool.sort(key=lambda r: (r["source_split"], r["image_id"]))
    return pool


def m0_scores(predictions_npz: Path) -> dict[str, float]:
    """image_id -> M0 bobcat probability, from the deployment-regime reference.

    The M0 FP32 ONNX scored through the same ORT CPU EP the Pi runs — the scores
    that decide which frames sit near the shipped operating point.
    """
    data = np.load(predictions_npz, allow_pickle=False)
    class_names = [str(name) for name in data["class_names"]]
    column = class_names.index("bobcat")
    scores: dict[str, float] = {}
    for split in VALIDATION_SPLITS:
        probabilities = data[f"{split}/probabilities"][:, column]
        image_ids = [str(i) for i in data[f"{split}/image_ids"]]
        for image_id, probability in zip(image_ids, probabilities):
            scores[image_id] = float(probability)
    return scores


def is_preprocessing_edge(record: dict) -> bool:
    """Aspect ratio far from the letterbox target, or an unusually small frame.

    CCT-20 validation turns out geometrically uniform (≈1.37 aspect throughout),
    so this is ~always False — recorded honestly as an empty stratum rather than
    invented. Kept as real logic so a future dataset with genuine edge geometry
    populates it without a code change.
    """
    width = record.get("observed_width")
    height = record.get("observed_height")
    if not width or not height:
        return False
    aspect = width / height
    return aspect < 1.30 or aspect > 1.45 or width < 640


def assign_stratum(record: dict, score: float | None, threshold: float, eps: float) -> str:
    labels = set(record["labels"])
    if score is not None and abs(score - threshold) < eps:
        return "threshold_adjacent"
    if "bobcat" in labels:
        return "bobcat"
    if record.get("multi_class"):
        return "multi_label"
    if labels & RARE_CLASSES:
        return "rare"
    if is_preprocessing_edge(record):
        return "preprocessing_edge"
    if labels == {"empty"}:
        return "empty"
    return "other"


def select(
    pool: list[dict],
    scores: dict[str, float],
    threshold: float,
    eps: float,
    seed: int,
) -> tuple[list[dict], dict]:
    """Partition into strata, take the registered per-stratum counts, fill to TOTAL.

    Deterministic: a NumPy Generator seeded once drives every shuffle, and the
    pool arrives pre-sorted, so the same inputs always yield the same manifest.
    """
    rng = np.random.default_rng(seed)

    by_stratum: dict[str, list[dict]] = {name: [] for name in STRATA}
    for record in pool:
        score = scores.get(record["image_id"])
        record["m0_bobcat_score"] = score
        stratum = assign_stratum(record, score, threshold, eps)
        record["benchmark_stratum"] = stratum
        by_stratum[stratum].append(record)

    chosen: list[dict] = []
    leftovers: list[dict] = []
    realized: dict[str, dict] = {}
    for name in STRATA:
        members = by_stratum[name]
        indices = rng.permutation(len(members))
        target = TARGETS[name]
        take = len(members) if target is None else min(target, len(members))
        picked = [members[i] for i in indices[:take]]
        chosen.extend(picked)
        leftovers.extend(members[i] for i in indices[take:])
        realized[name] = {"available": len(members), "target": target, "taken": take}

    if len(chosen) > TOTAL:
        raise RuntimeError(
            f"the take-all strata already exceed {TOTAL} "
            f"({len(chosen)}); tighten eps or the caps"
        )

    # Fill the remainder from the unpicked pool, seeded, recording where it came
    # from so the fill is not a black box.
    remaining = TOTAL - len(chosen)
    fill_indices = rng.permutation(len(leftovers))[:remaining]
    fill = [leftovers[i] for i in fill_indices]
    chosen.extend(fill)
    fill_by_stratum: dict[str, int] = {}
    for record in fill:
        fill_by_stratum[record["benchmark_stratum"]] = (
            fill_by_stratum.get(record["benchmark_stratum"], 0) + 1
        )

    # Order the frozen manifest deterministically, independent of draw order.
    chosen.sort(key=lambda r: (r["source_split"], r["image_id"]))

    accounting = {
        "total": len(chosen),
        "requested_total": TOTAL,
        "eps": eps,
        "threshold": threshold,
        "strata_priority_partition": realized,
        "fill": {"count": len(fill), "from_strata": dict(sorted(fill_by_stratum.items()))},
        "final_stratum_counts": _count(chosen),
        "threshold_adjacent_natural_fraction": round(
            realized["threshold_adjacent"]["available"] / len(pool), 5
        ),
        "threshold_adjacent_benchmark_fraction": round(
            _count(chosen).get("threshold_adjacent", 0) / len(chosen), 5
        ),
    }
    return chosen, accounting


def _count(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record["benchmark_stratum"]] = counts.get(record["benchmark_stratum"], 0) + 1
    return dict(sorted(counts.items()))


def build(
    manifests_dir: Path,
    predictions_npz: Path,
    policy_path: Path,
    output: Path,
    eps: float = 0.1,
    seed: int = 42,
) -> dict:
    policy = json.loads(policy_path.read_text())
    (target,) = [t for t in policy["targets"] if t["class"] == "bobcat"]
    threshold = float(target["threshold"])

    pool = load_pool(manifests_dir)
    scores = m0_scores(predictions_npz)
    missing = [r["image_id"] for r in pool if r["image_id"] not in scores]
    if missing:
        raise RuntimeError(
            f"{len(missing)} validation frames have no M0 score (e.g. "
            f"{missing[:3]}); the reference npz and the manifests disagree"
        )

    chosen, accounting = select(pool, scores, threshold, eps, seed)

    output.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(record, sort_keys=True) + "\n" for record in chosen)
    output.write_text(text)

    provenance = {
        "tool": "wildlife_trigger.data.benchmark_manifest",
        "design": "12.2",
        "output": str(output),
        "output_sha256": sha256_file(output),
        "frames": len(chosen),
        "seed": seed,
        "operating_point": {
            "source": "M0 FP32 (bobcat_v1 policy)",
            "threshold": threshold,
            "policy_sha256": sha256_file(policy_path),
        },
        "m0_predictions_sha256": sha256_file(predictions_npz),
        "inputs": {
            split: sha256_file(manifests_dir / f"{split}.jsonl")
            for split in VALIDATION_SPLITS
        },
        "accounting": accounting,
        "notes": {
            "preprocessing_edge": (
                "0 available: CCT-20 validation is geometrically uniform "
                "(all ≈1.37 aspect ratio), so there are no letterbox edge cases "
                "to sample. Recorded, not faked."
            ),
            "identical_for_every_model": (
                "built from M0's operating point and scored once; every model, "
                "M0 included, is benchmarked on this same ordered list (§12.2)."
            ),
        },
    }
    provenance_path = output.with_suffix(".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument(
        "--m0-predictions",
        type=Path,
        default=Path("results/optimize/m1_ptq/m0_fp32_reference/predictions.npz"),
    )
    parser.add_argument(
        "--m0-policy", type=Path, default=Path("artifacts/policies/bobcat_v1.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/manifests/benchmark_val_1000.jsonl")
    )
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    provenance = build(
        args.manifests_dir,
        args.m0_predictions,
        args.m0_policy,
        args.output,
        eps=args.eps,
        seed=args.seed,
    )
    counts = provenance["accounting"]["final_stratum_counts"]
    print(f"wrote {args.output} ({provenance['frames']} frames, sha "
          f"{provenance['output_sha256'][:12]}…)")
    print(f"strata: {counts}")
    print(
        f"threshold-adjacent over-sample: "
        f"{provenance['accounting']['threshold_adjacent_natural_fraction']:.2%} natural "
        f"→ {provenance['accounting']['threshold_adjacent_benchmark_fraction']:.2%} of benchmark"
    )
    print(f"wrote {args.output.with_suffix('.provenance.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
