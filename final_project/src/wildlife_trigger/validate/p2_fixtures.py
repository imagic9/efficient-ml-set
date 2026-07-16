#!/usr/bin/env python3
"""Freeze the P2 fixture set from the baseline's own validation predictions.

DESIGN §10's P2 amendment fixes the composition: **every** frame within 0.05 of
the calibrated threshold (the fire/no-fire carve-out must be exercised by real
borderline frames, not hypothesised), the top-20 bobcat-scored frames, and a
seeded stride sample across domains and predicted classes up to ~200 — so the
comparison sees confident positives, confident negatives, every class the model
actually predicts, and the region where a 1e-5 numeric wobble can legitimately
flip a decision.

Selected from the committed `predictions.npz` (the calibrated checkpoint's own
scores), so the set is reproducible from the repo without re-running inference.
Frozen with each frame's raw-file sha256: P2 must prove it compared the same
bytes, not merely the same names.

Usage:
    python -m wildlife_trigger.validate.p2_fixtures \
        --run results/training/c2/c2_m0_fp32_seed42_20260716T061203Z \
        --policy artifacts/policies/bobcat_v1.json \
        --output tests/fixtures/p2_fixtures.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..runs import resolve_run_id, sha256_file

# The registered composition (DESIGN §10 P2 amendment, 2026-07-16).
NEAR_THRESHOLD_BAND = 0.05
TOP_BOBCAT = 20
TARGET_TOTAL = 200
SEED = 42

DOMAINS = ("cis_val_clean", "trans_val")


def load_file_names(manifests_dir: Path) -> dict[str, str]:
    """image_id -> file_name across both validation manifests."""
    names: dict[str, str] = {}
    for domain in DOMAINS:
        with (manifests_dir / f"{domain}.jsonl").open() as handle:
            for line in handle:
                record = json.loads(line)
                names[record["image_id"]] = record["file_name"]
    return names


def select_fixtures(
    data: dict,
    class_names: list[str],
    threshold: float,
    rng: np.random.Generator,
) -> list[dict]:
    """The registered selection, deterministic given the npz and the seed."""
    bobcat = class_names.index("bobcat")
    frames = []
    for domain in DOMAINS:
        probabilities = data[f"{domain}/probabilities"]
        image_ids = [str(i) for i in data[f"{domain}/image_ids"]]
        top1 = probabilities.argmax(axis=1)
        for row, image_id in enumerate(image_ids):
            frames.append(
                {
                    "image_id": image_id,
                    "split": domain,
                    "seq_id": str(data[f"{domain}/seq_ids"][row]),
                    "npz_bobcat_probability": float(probabilities[row, bobcat]),
                    "npz_top1": class_names[top1[row]],
                }
            )

    chosen: dict[str, dict] = {}

    def take(frame: dict, reason: str) -> None:
        # First reason wins: a frame that is both near-threshold and top-bobcat
        # is one fixture, and the record says which rule brought it in.
        if frame["image_id"] not in chosen:
            chosen[frame["image_id"]] = {**frame, "reason": reason}

    for frame in frames:
        if abs(frame["npz_bobcat_probability"] - threshold) < NEAR_THRESHOLD_BAND:
            take(frame, "near_threshold")

    by_score = sorted(frames, key=lambda f: -f["npz_bobcat_probability"])
    for frame in by_score[:TOP_BOBCAT]:
        take(frame, "top_bobcat")

    # Stride sample: round-robin over (domain, predicted class) groups so every
    # class the model actually predicts is represented, rare ones first-class.
    groups: dict[tuple[str, str], list[dict]] = {}
    for frame in frames:
        if frame["image_id"] in chosen:
            continue
        groups.setdefault((frame["split"], frame["npz_top1"]), []).append(frame)
    group_keys = sorted(groups)
    for key in group_keys:
        rng.shuffle(groups[key])
    while len(chosen) < TARGET_TOTAL and any(groups[k] for k in group_keys):
        for key in group_keys:
            if len(chosen) >= TARGET_TOTAL:
                break
            if groups[key]:
                take(groups[key].pop(), "stride")

    return sorted(chosen.values(), key=lambda f: f["image_id"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path,
                        help="the calibrated policy; its bobcat threshold anchors the band")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--images-dir", type=Path,
                        help="raw image root; default: the run config's images_dir")
    args = parser.parse_args()

    history = json.loads((args.run / "history.json").read_text())
    data = np.load(args.run / "predictions.npz", allow_pickle=False)
    class_names = [str(name) for name in data["class_names"]]
    if class_names != history["class_names"]:
        raise RuntimeError("predictions.npz and history.json disagree on class order")

    policy = json.loads(args.policy.read_text())
    (target,) = [t for t in policy["targets"] if t["class"] == "bobcat"]
    threshold = float(target["threshold"])

    rng = np.random.default_rng(SEED)
    fixtures = select_fixtures(data, class_names, threshold, rng)

    images_dir = args.images_dir or Path(history["config"]["images_dir"])
    file_names = load_file_names(Path(history["config"]["manifests_dir"]))
    for fixture in fixtures:
        fixture["file_name"] = file_names[fixture["image_id"]]
        fixture["sha256"] = sha256_file(images_dir / fixture["file_name"])

    payload = {
        "purpose": "P2 FP32 parity fixtures (DESIGN 10 amendment, 2026-07-16)",
        "run_id": resolve_run_id(args.run, history["run_name"]),
        "threshold": threshold,
        "near_threshold_band": NEAR_THRESHOLD_BAND,
        "top_bobcat": TOP_BOBCAT,
        "target_total": TARGET_TOTAL,
        "seed": SEED,
        "predictions_npz_sha256": sha256_file(args.run / "predictions.npz"),
        "counts": {
            reason: sum(1 for f in fixtures if f["reason"] == reason)
            for reason in ("near_threshold", "top_bobcat", "stride")
        },
        "fixtures": fixtures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"{len(fixtures)} fixtures: {payload['counts']}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
