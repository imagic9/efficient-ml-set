#!/usr/bin/env python3
"""C0 — freeze the golden preprocessing fixtures (PLAN C0, DESIGN §5.5).

P1 compares the Python and C++ preprocessors, and that comparison is only meaningful
against a fixed set of images chosen for the cases where they could disagree. Picking
20 frames at random would mostly sample the dominant 1024x747 landscape frame and prove
that the two agree on the easy case.

So the selection is deliberately adversarial to the letterbox:

  - **the dominant frame** (1024x747) — the case that must be right;
  - **every other observed geometry** — B0 measured several; each exercises different
    resize and pad arithmetic;
  - **the most portrait and most landscape frames** — the extremes of pad_left vs pad_top;
  - **near-square frames** — where `min()` picks a different axis than usual;
  - **odd dimensions** — where the integer pad `(target - resized) // 2` is asymmetric
    and a one-pixel disagreement would show up;
  - **an IR/greyscale-looking frame and a bobcat frame** — real content, so the fixture
    set is not all one camera pointed at a wall.

Only the raw image hashes and their metadata are frozen here. **Tensor shapes stay
provisional until C1a selects the input contract** (PLAN C0): freezing golden tensors at
256x192 before that decision would either bless the answer or have to be redone.

Usage:
    python -m wildlife_trigger.validate.golden_fixtures --manifests-dir data/manifests \
        --images-dir data/raw/extracted/eccv_18_all_images_sm \
        --output tests/fixtures/golden_raw.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

MIN_FIXTURES = 20
TARGET_CLASS = "bobcat"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(records: list[dict], images_dir: Path) -> list[dict]:
    """Choose fixtures by the cases they exercise, deterministically."""
    usable = [
        r for r in records if r.get("observed_width") and r.get("observed_height")
    ]
    usable.sort(key=lambda r: r["image_id"])  # deterministic before any selection

    by_geometry: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for record in usable:
        by_geometry[(record["observed_width"], record["observed_height"])].append(record)

    chosen: dict[str, dict] = {}

    def take(record: dict, reason: str) -> None:
        if record["image_id"] not in chosen:
            chosen[record["image_id"]] = {**record, "fixture_reason": reason}

    # One of every observed geometry: each is a distinct resize/pad computation.
    for geometry, group in sorted(by_geometry.items(), key=lambda kv: -len(kv[1])):
        take(group[0], f"observed geometry {geometry[0]}x{geometry[1]} "
                       f"({len(group)} frames in the corpus)")

    aspect = lambda r: r["observed_width"] / r["observed_height"]

    # The extremes of the letterbox: most landscape pads top/bottom, most portrait pads
    # left/right, near-square is where min() switches axis.
    take(max(usable, key=aspect), "widest observed aspect ratio: maximum vertical pad")
    take(min(usable, key=aspect), "narrowest observed aspect ratio: maximum horizontal pad")
    take(min(usable, key=lambda r: abs(aspect(r) - 1.0)), "closest to square")

    # Odd dimensions: the integer pad `(target - resized) // 2` is asymmetric, and a
    # one-pixel disagreement between implementations lands exactly here.
    odd = [r for r in usable if r["observed_width"] % 2 or r["observed_height"] % 2]
    if odd:
        take(odd[0], "odd dimension: asymmetric integer padding")

    # Real content rather than only geometry.
    for record in usable:
        if TARGET_CLASS in record["labels"]:
            take(record, f"{TARGET_CLASS} present: the target class")
            break
    for record in usable:
        if record["labels"] == ["empty"]:
            take(record, "empty frame: the dominant real-world input")
            break

    # Fill to the floor with a deterministic spread across the corpus rather than the
    # first N, which would all come from one camera.
    if len(chosen) < MIN_FIXTURES:
        stride = max(1, len(usable) // MIN_FIXTURES)
        for record in usable[::stride]:
            take(record, "spread across the corpus to reach the fixture floor")
            if len(chosen) >= MIN_FIXTURES:
                break

    return sorted(chosen.values(), key=lambda r: r["image_id"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests-dir", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["cis_val_clean", "trans_val"],
        help="Validation only. DESIGN §5.4 seals the test splits, and fixtures drawn "
        "from them would put test frames in the repository.",
    )
    args = parser.parse_args()

    records = []
    for split in args.splits:
        for line in (args.manifests_dir / f"{split}.jsonl").read_text().splitlines():
            record = json.loads(line)
            record["split"] = split
            records.append(record)

    fixtures = select(records, args.images_dir)

    entries = []
    for record in fixtures:
        path = args.images_dir / record["file_name"]
        entries.append(
            {
                "image_id": record["image_id"],
                "file_name": record["file_name"],
                "split": record["split"],
                "labels": record["labels"],
                "observed_width": record["observed_width"],
                "observed_height": record["observed_height"],
                "aspect_ratio": round(
                    record["observed_width"] / record["observed_height"], 5
                ),
                "sha256": sha256_file(path),
                "reason": record["fixture_reason"],
            }
        )

    document = {
        "task": "C0",
        "count": len(entries),
        "selected_from": args.splits,
        "why_not_test": (
            "DESIGN §5.4 seals cis-test and trans-test. Fixtures drawn from them would "
            "commit test frames to the repository and be consulted throughout "
            "development."
        ),
        "tensor_shapes": (
            "PROVISIONAL — not frozen here. PLAN C0 defers golden tensors until C1a "
            "selects the input contract (256x192 vs 224x224). Freezing them now would "
            "either pre-empt that decision or have to be redone."
        ),
        "fixtures": entries,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")

    print(f"selected {len(entries)} golden fixtures -> {args.output}")
    for entry in entries:
        print(
            f"  {entry['observed_width']:5d}x{entry['observed_height']:<5d} "
            f"{','.join(entry['labels']):<16s} {entry['reason']}"
        )

    if len(entries) < MIN_FIXTURES:
        print(f"\nFAILED: {len(entries)} fixtures, PLAN C0 requires at least {MIN_FIXTURES}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
