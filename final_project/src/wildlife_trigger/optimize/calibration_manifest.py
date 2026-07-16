#!/usr/bin/env python3
"""D1: the fixed 1,024-image PTQ calibration manifest (DESIGN §8.1).

Calibration data decides what PTQ produces — the activation scales *are* the
model. DESIGN §8.1 therefore fixes the data once: 1,024 images from **training
data only** (CCT-20 train plus the B2 empty supplement), stratified by class and
source, committed with a hash, and read identically by every calibration method.
This module is that one build; everything downstream refuses to re-draw it.

The registered stratification rule (results/optimize/m1_ptq/preregistration.md §6):

- **Universe**: `train.jsonl` ∪ `cct_empty_train_v1.jsonl`. Nothing else is
  reachable: the loader refuses any manifest whose filename smells of val/test,
  because a calibration set is training data by definition and a leak here would
  contaminate every quantized candidate at once.
- **Eligible**: single-class frames only (`primary_label` is not None). The
  seven multi-class train frames are ambiguous stratum members and are excluded
  from calibration exactly as B3 excludes them from cross-entropy.
- **Strata**: (source, primary_label). The supplement is its own source — it is
  not CCT-20 and never gets to look like it (the ConcatManifestDataset rule).
- **Allocation**: a floor of 8 per stratum (capped by stratum size) so rare
  classes appear at all, then the remainder proportionally by remaining stratum
  capacity, largest-remainder rounded, deterministically.
- **Draw**: numpy Generator seeded 20260716, strata visited in sorted order,
  pools sorted by image_id before sampling — same inputs, same bytes, forever.

Records are copied verbatim from their source manifests plus a `source` field
("cct20" | "empty_supplement"), which is exactly the key
`WildlifeDataset.image_path` resolves image roots by.

Usage (gx10):
    python -m wildlife_trigger.optimize.calibration_manifest \
        --train data/manifests/train.jsonl \
        --supplement data/manifests/cct_empty_train_v1.jsonl \
        --output data/manifests/calibration_1024.jsonl \
        --report results/optimize/m1_ptq/calibration_manifest.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ..data.manifests import write_jsonl
from ..runs import atomic_write_json, sha256_file

TOTAL = 1024
STRATUM_FLOOR = 8
SEED = 20260716

# The two training sources, under the names WildlifeDataset resolves roots by.
CCT20 = "cct20"
SUPPLEMENT = "empty_supplement"

# A calibration manifest is training data. These tokens in a source filename mean
# someone pointed the builder at an evaluation split, and the only correct
# response is refusal — a quantized model calibrated on validation has seen the
# data its threshold will be picked on.
FORBIDDEN_NAME_TOKENS = ("val", "test")


def load_source(path: Path, source: str) -> list[dict]:
    """Read one training manifest, tag its records with their source."""
    lowered = path.name.lower()
    for token in FORBIDDEN_NAME_TOKENS:
        if token in lowered:
            raise ValueError(
                f"{path} looks like an evaluation split ({token!r} in the name); "
                "calibration data is training data only (DESIGN §8.1)"
            )
    records = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                record["source"] = source
                records.append(record)
    if not records:
        raise ValueError(f"{path} is empty; cannot calibrate on nothing")
    return records


def eligible(records: list[dict]) -> list[dict]:
    """Single-class frames only — the same frames CE trains on."""
    return [r for r in records if r.get("primary_label") is not None]


def stratum_of(record: dict) -> tuple[str, str]:
    return (record["source"], record["primary_label"])


def allocate(sizes: dict[tuple[str, str], int], total: int, floor: int) -> dict:
    """Floor per stratum, then the remainder proportionally by capacity.

    Deterministic largest-remainder rounding, iterated because a stratum can hit
    its capacity mid-round and its unmet share must be redistributed rather than
    silently dropped — 1,024 means 1,024.
    """
    if total > sum(sizes.values()):
        raise ValueError(
            f"asked for {total} images but only {sum(sizes.values())} are eligible"
        )
    alloc = {key: min(floor, size) for key, size in sizes.items()}

    remaining = total - sum(alloc.values())
    if remaining < 0:
        raise ValueError(
            f"floors alone ({sum(alloc.values())}) exceed the target {total}; "
            "lower the floor or raise the target"
        )
    while remaining > 0:
        capacity = {k: sizes[k] - alloc[k] for k in sizes if sizes[k] > alloc[k]}
        if not capacity:
            raise ValueError("eligible images exhausted before reaching the target")
        weight_total = sum(capacity.values())
        shares = {k: remaining * cap / weight_total for k, cap in capacity.items()}
        granted = {k: min(int(shares[k]), capacity[k]) for k in capacity}
        leftover = remaining - sum(granted.values())
        # Largest fractional remainder first; the stratum key breaks exact ties so
        # the order never depends on dict insertion.
        for key in sorted(capacity, key=lambda k: (-(shares[k] - int(shares[k])), k)):
            if leftover == 0:
                break
            if granted[key] < capacity[key]:
                granted[key] += 1
                leftover -= 1
        for key, grant in granted.items():
            alloc[key] += grant
        remaining = total - sum(alloc.values())
    return alloc


def draw(
    records: list[dict], allocation: dict, seed: int = SEED
) -> list[dict]:
    """Sample each stratum without replacement, deterministically.

    One Generator, strata in sorted order, pools sorted by image_id: every part
    of the sequence of random draws is pinned, so the manifest's bytes are a
    function of (sources, rule, seed) and nothing else.
    """
    pools: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        pools.setdefault(stratum_of(record), []).append(record)

    rng = np.random.default_rng(seed)
    chosen: list[dict] = []
    for key in sorted(allocation):
        pool = sorted(pools[key], key=lambda r: r["image_id"])
        count = allocation[key]
        indices = rng.choice(len(pool), size=count, replace=False)
        chosen.extend(pool[i] for i in sorted(indices.tolist()))

    # image_id is unique across both sources (uuids), so this order is total.
    chosen.sort(key=lambda r: r["image_id"])
    if len({r["image_id"] for r in chosen}) != len(chosen):
        raise RuntimeError("duplicate image_id in the draw; the sources overlap")
    return chosen


def verify_images(records: list[dict], roots: dict[str, Path]) -> None:
    """Every referenced file must exist where the reader will look for it.

    A missing JPEG discovered here costs a message; discovered mid-quantization
    it costs the run and leaves a half-calibrated artifact on disk.
    """
    missing = []
    for record in records:
        root = roots[record["source"]]
        if record.get("relative_path"):
            path = root.parent / record["relative_path"]
        else:
            path = root / record["file_name"]
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} calibration images are missing, first: {missing[0]}"
        )


def build(
    train_manifest: Path,
    supplement_manifest: Path,
    total: int = TOTAL,
    floor: int = STRATUM_FLOOR,
    seed: int = SEED,
) -> tuple[list[dict], dict]:
    records = eligible(
        load_source(train_manifest, CCT20)
        + load_source(supplement_manifest, SUPPLEMENT)
    )
    sizes = dict(Counter(stratum_of(r) for r in records))
    allocation = allocate(sizes, total, floor)
    chosen = draw(records, allocation, seed)

    report = {
        "tool": "wildlife_trigger.optimize.calibration_manifest",
        "design": "8.1",
        "rule": (
            f"training data only; single-class frames; strata (source, class); "
            f"floor {floor} per stratum then proportional by capacity, "
            f"largest-remainder; seed {seed}"
        ),
        "seed": seed,
        "total": len(chosen),
        "sources": {
            CCT20: {"path": str(train_manifest), "sha256": sha256_file(train_manifest)},
            SUPPLEMENT: {
                "path": str(supplement_manifest),
                "sha256": sha256_file(supplement_manifest),
            },
        },
        "strata": {
            f"{source}/{label}": {
                "eligible": sizes[(source, label)],
                "drawn": allocation[(source, label)],
            }
            for source, label in sorted(sizes)
        },
        "drawn_by_source": dict(Counter(r["source"] for r in chosen)),
        "drawn_by_class": dict(
            sorted(Counter(r["primary_label"] for r in chosen).items())
        ),
    }
    return chosen, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--supplement", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--total", type=int, default=TOTAL)
    parser.add_argument("--floor", type=int, default=STRATUM_FLOOR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--images-root",
        type=Path,
        help="CCT-20 image directory; with --supplement-root, verify every file exists",
    )
    parser.add_argument("--supplement-root", type=Path)
    args = parser.parse_args()

    chosen, report = build(
        args.train, args.supplement, total=args.total, floor=args.floor, seed=args.seed
    )
    if args.images_root and args.supplement_root:
        verify_images(chosen, {CCT20: args.images_root, SUPPLEMENT: args.supplement_root})
        report["images_verified"] = True

    manifest_sha = write_jsonl(args.output, chosen)
    report["output"] = {"path": str(args.output), "sha256": manifest_sha}
    atomic_write_json(args.report, report)

    print(f"wrote {args.output} ({len(chosen)} images, sha256 {manifest_sha[:12]}…)")
    print(f"wrote {args.report}")
    for key, counts in report["strata"].items():
        print(f"  {key}: {counts['drawn']}/{counts['eligible']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
