#!/usr/bin/env python3
"""B2 — build `cct_empty_train_v1`, the empty training supplement.

CCT-20's train split contains no `empty` frames, while validation and test are full
of them. A trigger trained without them has never seen "nothing is here" and will
fire on empty frames — which is the product's dominant real-world input.

## The confound this module exists to avoid

Per-image CCT downloads are served at **original resolution** (~2048x1494) while every
CCT-20 split is capped at **1024 px**. Skip the downsize and `empty` becomes the only
training class carrying double resolution and a second JPEG generation — a feature
perfectly correlated with the label.

DESIGN §5.2 is worth restating because the failure mode is so quiet: validation and
test contain *only* `_sm` frames, so the shortcut is **absent at evaluation time**. A
model that learned "2048-px artifacts mean empty" cannot recognise `empty` where it is
measured, the bobcat false-fire rate is inflated exactly there, and the `A-empty-5k`
ablation reads as "the supplement barely helped" — misattributing the cause to the
location-disjoint rule. Nothing crashes. The headline number is just wrong.

So every selected image is downsized to max 1024 px with a recorded filter and quality,
and both checksums are stored. `shortcut_probe.py` then tries to tell the two pools
apart; near-chance means the confound is closed.

## The confound that remains, and cannot be removed

Rule 3 makes the supplement location-disjoint from all 20 CCT-20 locations, so the
model sees `empty` only on unfamiliar backgrounds — a second feature correlated with
the same label. It is unavoidable: within the 10 cis locations, every full-CCT `empty`
frame is already spent in cis-val and cis-test, so a background-matched supplement does
not exist. DESIGN requires the report to state this rather than fix it.

Usage:
    python -m wildlife_trigger.data.empty_supplement --select ...   # choose + manifest
    python -m wildlife_trigger.data.empty_supplement --download ... # fetch + downsize
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

# LILA per-image base, verified 2026-07-15. Images are served at original resolution.
#
# LILA publishes the same files on GCP, AWS and Azure. Measured from gx10, fetching 48
# images concurrently:
#
#   azure, 12 workers :   40 img/min   (and 4 of 48 requests failed)
#   gcp,   12 workers :  740 img/min   (0 failures)
#   gcp,   48 workers : 2144 img/min   (0 failures)
#
# Azure throttles hard enough that the retry loop absorbed it into ~20 s per image and
# the full 5,000 would have taken over two hours. GCP is the same bytes 18x faster, and
# it is where the 6.5 GB `_sm` archive already comes from. Mirrors change; re-measure
# rather than inherit this.
LILA_IMAGE_BASE = "https://storage.googleapis.com/public-datasets-lila/caltech-unzipped/cct_images"
LILA_IMAGE_MIRRORS = {
    "gcp": LILA_IMAGE_BASE,
    "azure": "https://lilawildlife.blob.core.windows.net/lila-wildlife/caltech-unzipped/cct_images",
    "aws": "https://us-west-2.opendata.source.coop/agentmorris/lila-wildlife/caltech-unzipped/cct_images",
}

SUPPLEMENT_SIZE = 5000
SELECTION_SEED = 42  # DESIGN §5.2 rule 5, fixed.

# DESIGN §5.2 step 7: match the CCT-20 `_sm` archive exactly. B0 measured the archive's
# max long side at 1024 across all 57,864 frames, so this is the real cap, not a guess.
MAX_LONG_SIDE = 1024

# Recorded in the data config because they are part of the contract: a different filter
# or quality produces a different `empty` distribution and a different confound.
RESAMPLE_FILTER = "LANCZOS"
JPEG_QUALITY = 90


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalise_location(value) -> str:
    """Compare locations as strings, always.

    **This is not defensive tidiness; it is the rule.** The two metadata files disagree
    about the type: full CCT stores `location` as a string (`"26"`), CCT-20 stores it as
    an integer (`38`). So `image["location"] in cct20_locations` is `"26" in {26, 38}`,
    which is False for *every* image — rule 3 would be silently disabled and the
    supplement would draw `empty` frames from the very cameras it must avoid.

    Measured on this data (2026-07-15): the raw comparison finds 0 overlapping images,
    the normalised comparison finds 32,255. Nothing would have crashed.
    """
    return str(value)


def load_cct20_identity(manifests_dir: Path) -> dict[str, set]:
    """Everything CCT-20 already uses: locations, image ids, sequence ids.

    All five splits, not just train: rule 4 forbids a candidate that appears anywhere
    in CCT-20, and an image reused from cis-test would be a test-set leak into
    training — the single worst outcome available here.
    """
    locations: set = set()
    image_ids: set = set()
    seq_ids: set = set()

    for split in ("train", "cis_val", "cis_test", "trans_val", "trans_test"):
        for line in (manifests_dir / f"{split}.jsonl").read_text().splitlines():
            record = json.loads(line)
            locations.add(normalise_location(record["location"]))
            image_ids.add(str(record["image_id"]))
            seq_ids.add(str(record["seq_id"]))

    return {"locations": locations, "image_ids": image_ids, "seq_ids": seq_ids}


def select(full_cct: Path, manifests_dir: Path, output: Path) -> dict:
    """Apply DESIGN §5.2 rules 1-5 and emit the selection manifest."""
    cct20 = load_cct20_identity(manifests_dir)
    document = json.loads(full_cct.read_text())

    categories = {c["id"]: c["name"] for c in document["categories"]}
    empty_ids = {i for i, name in categories.items() if name == "empty"}
    if not empty_ids:
        raise RuntimeError("full CCT metadata declares no 'empty' category")

    # Rule 2: label must be exactly `empty`. An image with an empty annotation AND an
    # animal annotation is not an empty frame.
    labels: dict[str, set] = defaultdict(set)
    for annotation in document["annotations"]:
        labels[annotation["image_id"]].add(annotation["category_id"])

    candidates = []
    rejected = Counter()
    for image in document["images"]:
        image_labels = labels.get(image["id"], set())
        # Rule 2: exactly `empty`. An image carrying both an empty annotation and an
        # animal annotation is not an empty frame.
        if not image_labels or not image_labels.issubset(empty_ids):
            rejected["not_exactly_empty"] += 1
            continue
        if normalise_location(image.get("location")) in cct20["locations"]:
            rejected["cct20_location"] += 1
            continue
        if str(image["id"]) in cct20["image_ids"]:
            rejected["cct20_image_id"] += 1
            continue
        if str(image.get("seq_id")) in cct20["seq_ids"]:
            rejected["cct20_seq_id"] += 1
            continue
        candidates.append(image)

    # Rule 3 must actually have removed something. CCT-20's 20 locations hold tens of
    # thousands of full-CCT empty frames, so a zero here does not mean "clean data" --
    # it means the comparison silently matched nothing, which is precisely the type
    # mismatch normalise_location exists to prevent.
    if rejected["cct20_location"] == 0:
        raise RuntimeError(
            "rule 3 rejected zero candidates for being in a CCT-20 location. CCT-20's "
            "locations contain many full-CCT empty frames, so this means the location "
            "comparison matched nothing -- almost certainly a type mismatch between "
            "the two metadata files -- and the supplement would be drawn from the very "
            "cameras it must avoid."
        )

    # Rule 5: stratified across locations and sequences so one camera cannot dominate.
    # Round-robin over locations, and within a location over sequences, taking one frame
    # at a time. A plain random sample of 5,000 would follow the location distribution,
    # and CCT locations are wildly unbalanced — the supplement would then teach "empty
    # looks like location 122".
    rng = random.Random(SELECTION_SEED)

    by_location: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for image in candidates:
        by_location[image["location"]][image["seq_id"]].append(image)

    for location in by_location:
        for seq_id in by_location[location]:
            by_location[location][seq_id].sort(key=lambda i: i["id"])
            rng.shuffle(by_location[location][seq_id])

    location_order = sorted(by_location)
    rng.shuffle(location_order)
    sequence_cursors = {
        location: sorted(by_location[location], key=str) for location in location_order
    }
    for location in location_order:
        rng.shuffle(sequence_cursors[location])

    selected: list[dict] = []
    exhausted: set = set()
    while len(selected) < SUPPLEMENT_SIZE and len(exhausted) < len(location_order):
        for location in location_order:
            if location in exhausted or len(selected) >= SUPPLEMENT_SIZE:
                continue
            took = False
            for seq_id in sequence_cursors[location]:
                pool = by_location[location][seq_id]
                if pool:
                    selected.append(pool.pop())
                    took = True
                    break
            if not took:
                exhausted.add(location)

    records = [
        {
            "image_id": image["id"],
            "file_name": image["file_name"],
            "labels": ["empty"],
            "primary_label": "empty",
            "multi_class": False,
            "location": image["location"],
            "seq_id": image["seq_id"],
            "frame_num": image.get("frame_num"),
            "date_captured": image.get("date_captured"),
            "declared_width": image.get("width"),
            "declared_height": image.get("height"),
            "source_url": f"{LILA_IMAGE_BASE}/{image['file_name']}",
        }
        for image in selected
    ]
    records.sort(key=lambda r: r["image_id"])

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
    )

    per_location = Counter(r["location"] for r in records)
    report = {
        "task": "B2-select",
        "seed": SELECTION_SEED,
        "requested": SUPPLEMENT_SIZE,
        "selected": len(records),
        "candidates_available": len(candidates),
        "rejected": dict(rejected),
        "locations_used": len(per_location),
        "images_per_location": {
            str(k): v for k, v in per_location.most_common()
        },
        "max_location_share": round(max(per_location.values()) / len(records), 4)
        if records
        else 0.0,
        "sequences_used": len({r["seq_id"] for r in records}),
        "disjointness": {
            "locations_overlapping_cct20": 0,
            "image_ids_overlapping_cct20": 0,
            "seq_ids_overlapping_cct20": 0,
        },
        "manifest": str(output),
        "manifest_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    return report


def verify_disjoint(manifest: Path, manifests_dir: Path) -> dict:
    """Rule 4/9, re-checked against the written manifest rather than the intent.

    Separate from `select` on purpose: this reads what was actually saved. A selection
    bug and a verification that trusts the same in-memory objects would agree with each
    other.
    """
    cct20 = load_cct20_identity(manifests_dir)
    records = [json.loads(l) for l in manifest.read_text().splitlines()]

    # Normalised on both sides, for the same reason select() is: an unnormalised
    # verification would report "disjoint" for exactly the data that is not.
    location_leaks = sorted(
        {normalise_location(r["location"]) for r in records} & cct20["locations"]
    )
    id_leaks = sorted({str(r["image_id"]) for r in records} & cct20["image_ids"])
    seq_leaks = sorted({str(r["seq_id"]) for r in records} & cct20["seq_ids"])

    return {
        "images": len(records),
        "location_leaks": location_leaks,
        "image_id_leaks": id_leaks[:10],
        "seq_id_leaks": seq_leaks[:10],
        "disjoint": not (location_leaks or id_leaks or seq_leaks),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-cct", required=True, type=Path)
    parser.add_argument("--manifests-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    report = select(args.full_cct, args.manifests_dir, args.output)
    verification = verify_disjoint(args.output, args.manifests_dir)
    report["verification"] = verification

    print(f"candidates: {report['candidates_available']}  rejected: {report['rejected']}")
    print(
        f"selected  : {report['selected']} across {report['locations_used']} locations, "
        f"{report['sequences_used']} sequences"
    )
    print(f"max single-location share: {report['max_location_share']:.1%}")
    print(f"disjoint from CCT-20: {verification['disjoint']}")
    if not verification["disjoint"]:
        print(f"  LEAKS: {verification}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.report}")

    ok = verification["disjoint"] and report["selected"] == SUPPLEMENT_SIZE
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
