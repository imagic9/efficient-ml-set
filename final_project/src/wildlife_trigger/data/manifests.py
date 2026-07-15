#!/usr/bin/env python3
"""B1 — freeze the class order and build the official split manifests.

Two things happen here that the rest of the project depends on and cannot revisit.

**The class order is frozen.** CCT-20's category IDs are sparse (1, 3, 5, 6, ... 99),
so "the 16 classes" is not an order until someone chooses one. Every threshold in
every policy is bound to an integer index, so this choice is permanent in the sense
that changing it silently rebinds every calibrated threshold to a different animal.
The order is ascending category ID: derived from the dataset, deterministic, and
traceable to the source rather than to a preference.

**The train/cis-val overlap is fingerprinted and then removed.** The official CCT-20
cis-val split shares `seq_id` values with train. Burst frames within one sequence are
near-duplicates, so a model that memorised a training frame scores on its twin at
validation. DESIGN pins the overlap to exact numbers — 224 sequences, 270 cis-val
images, 10 bobcat images — and reconciling against them is how we learn that LILA
republished the metadata rather than discovering it as an inexplicably good result.

`cis_val_clean.jsonl` is what calibration and model selection use. The dirty split is
kept, because "we removed leakage" is a claim that needs both sides to be checkable.

Usage:
    python -m wildlife_trigger.data.manifests --annotations-dir A --dimensions D \
        --output-dir data/manifests --classes-config configs/data/classes.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

SPLITS = ("train", "cis_val", "cis_test", "trans_val", "trans_test")

# DESIGN §5.1. Reconciled against, never assumed: the ECCV paper says 57,868 and the
# downloadable JSONs sum to 57,864, so the file contents are executable truth.
EXPECTED_COUNTS = {
    "train": 13553,
    "cis_val": 3484,
    "cis_test": 15827,
    "trans_val": 1725,
    "trans_test": 23275,
}

# DESIGN §5.3's measured leakage fingerprint.
EXPECTED_OVERLAP = {"sequences": 224, "cis_val_images": 270, "bobcat_images": 10}
EXPECTED_CLEAN = {"images": 3214, "bobcat_images": 144}

# DESIGN §4: `car` and `empty` are model classes but never selectable wildlife
# targets. The remaining 14 are.
NON_ANIMAL_CLASSES = frozenset({"car", "empty"})
EXPECTED_CLASS_COUNT = 16
EXPECTED_ANIMAL_COUNT = 14

TARGET_CLASS = "bobcat"


def load_split(annotations_dir: Path, split: str) -> dict:
    return json.loads((annotations_dir / f"{split}_annotations.json").read_text())


def freeze_class_order(annotations_dir: Path) -> list[dict]:
    """Collect categories across every split and order them by ascending ID.

    Every split is read rather than just train: a class absent from train but present
    in test would otherwise silently shift every later index.
    """
    categories: dict[int, str] = {}
    for split in SPLITS:
        for category in load_split(annotations_dir, split)["categories"]:
            existing = categories.get(category["id"])
            if existing is not None and existing != category["name"]:
                raise RuntimeError(
                    f"category id {category['id']} is {existing!r} in one split and "
                    f"{category['name']!r} in {split}. The splits disagree about what "
                    "a class means; nothing downstream can be trusted."
                )
            categories[category["id"]] = category["name"]

    ordered = [
        {
            "index": index,
            "category_id": category_id,
            "name": categories[category_id],
            "selectable_target": categories[category_id] not in NON_ANIMAL_CLASSES,
        }
        for index, category_id in enumerate(sorted(categories))
    ]

    if len(ordered) != EXPECTED_CLASS_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_CLASS_COUNT} classes, found {len(ordered)}: "
            f"{[c['name'] for c in ordered]}"
        )
    animals = [c for c in ordered if c["selectable_target"]]
    if len(animals) != EXPECTED_ANIMAL_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_ANIMAL_COUNT} animal classes, found {len(animals)}"
        )
    missing = NON_ANIMAL_CLASSES - {c["name"] for c in ordered}
    if missing:
        raise RuntimeError(f"expected non-animal classes are absent: {missing}")
    return ordered


def build_records(
    document: dict, id_to_name: dict[int, str], dimensions: dict[str, list[int]]
) -> list[dict]:
    """One record per image, carrying the COMPLETE label set.

    `labels` is the full set because DESIGN evaluates target *presence*: a frame
    holding a bobcat and a coyote is a true bobcat frame regardless of which the
    model calls top-1. `primary_label` exists only when the image carries exactly one
    distinct class, and is None otherwise — that is the signal B3 uses to exclude
    multi-class frames from cross-entropy without discarding them from evaluation.
    """
    labels: dict[str, set[str]] = {}
    for annotation in document["annotations"]:
        name = id_to_name[annotation["category_id"]]
        labels.setdefault(annotation["image_id"], set()).add(name)

    records = []
    for image in document["images"]:
        image_labels = sorted(labels.get(image["id"], set()))
        observed = dimensions.get(image["id"])
        records.append(
            {
                "image_id": image["id"],
                "file_name": image["file_name"],
                "labels": image_labels,
                "primary_label": image_labels[0] if len(image_labels) == 1 else None,
                "multi_class": len(image_labels) > 1,
                "location": image["location"],
                "seq_id": image["seq_id"],
                "frame_num": image.get("frame_num"),
                "seq_num_frames": image.get("seq_num_frames"),
                "date_captured": image.get("date_captured"),
                "rights_holder": image.get("rights_holder"),
                # Both geometries, because they differ and both matter: the declared
                # one is the original camera frame, the observed one is what we decode.
                "declared_width": image["width"],
                "declared_height": image["height"],
                "observed_width": observed[0] if observed else None,
                "observed_height": observed[1] if observed else None,
            }
        )

    # Sort by image_id: a manifest whose order depends on JSON iteration is not
    # reproducible, and every hash downstream would move with it.
    records.sort(key=lambda record: record["image_id"])
    return records


def write_jsonl(path: Path, records: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def summarise(records: list[dict]) -> dict:
    label_counts: Counter = Counter()
    for record in records:
        for label in record["labels"]:
            label_counts[label] += 1
    return {
        "images": len(records),
        "unlabelled_images": sum(1 for r in records if not r["labels"]),
        "multi_class_images": sum(1 for r in records if r["multi_class"]),
        "sequences": len({r["seq_id"] for r in records}),
        "locations": sorted({r["location"] for r in records}),
        "label_counts": dict(label_counts.most_common()),
        f"{TARGET_CLASS}_images": label_counts[TARGET_CLASS],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-dir", required=True, type=Path)
    parser.add_argument("--dimensions", type=Path, help="B0's per-image index.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--classes-config", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    dimensions: dict[str, list[int]] = (
        json.loads(args.dimensions.read_text()) if args.dimensions else {}
    )

    classes = freeze_class_order(args.annotations_dir)
    id_to_name = {c["category_id"]: c["name"] for c in classes}
    names = [c["name"] for c in classes]
    print(f"frozen class order ({len(names)}): {names}")

    # YAML by hand: the only structure needed is a list of scalars, and a dependency
    # for that is not worth the reader's attention.
    lines = [
        "# The frozen 16-class order for CCT-20 (PLAN B1). GENERATED — do not edit.",
        "#",
        "# Ordered by ascending CCT category ID: derived from the dataset, so it is",
        "# deterministic and traceable to the source rather than to a preference.",
        "#",
        "# Changing this order silently rebinds every calibrated threshold to a",
        "# different animal. It is frozen for the life of the project.",
        "#",
        "# `car` and `empty` are model classes but never selectable policy targets",
        "# (DESIGN §4).",
        "",
        "classes:",
    ]
    for entry in classes:
        flag = "true " if entry["selectable_target"] else "false"
        lines.append(
            f"  - {{index: {entry['index']:2d}, category_id: {entry['category_id']:2d}, "
            f"name: {entry['name']:<9} selectable_target: {flag}}}"
        )
    args.classes_config.parent.mkdir(parents=True, exist_ok=True)
    args.classes_config.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.classes_config}")

    per_split: dict[str, dict] = {}
    records_by_split: dict[str, list[dict]] = {}
    for split in SPLITS:
        document = load_split(args.annotations_dir, split)
        records = build_records(document, id_to_name, dimensions)
        records_by_split[split] = records
        digest = write_jsonl(args.output_dir / f"{split}.jsonl", records)
        info = summarise(records)
        info["sha256"] = digest
        info["expected_images"] = EXPECTED_COUNTS[split]
        info["count_reconciles"] = info["images"] == EXPECTED_COUNTS[split]
        per_split[split] = info
        print(
            f"  {split:10s} {info['images']:6d} images "
            f"({'ok' if info['count_reconciles'] else 'MISMATCH'})  "
            f"{info['sequences']:5d} seqs  {info['multi_class_images']:3d} multi-class  "
            f"{info[TARGET_CLASS + '_images']:4d} {TARGET_CLASS}"
        )

    # --- the leakage fingerprint ---
    train_sequences = {r["seq_id"] for r in records_by_split["train"]}
    cis_val = records_by_split["cis_val"]
    overlapping = [r for r in cis_val if r["seq_id"] in train_sequences]
    overlap = {
        "sequences": len({r["seq_id"] for r in overlapping}),
        "cis_val_images": len(overlapping),
        "bobcat_images": sum(1 for r in overlapping if TARGET_CLASS in r["labels"]),
    }
    overlap_matches = overlap == EXPECTED_OVERLAP

    clean_records = [r for r in cis_val if r["seq_id"] not in train_sequences]
    clean_digest = write_jsonl(args.output_dir / "cis_val_clean.jsonl", clean_records)
    clean_summary = summarise(clean_records)
    clean = {
        "images": clean_summary["images"],
        "bobcat_images": clean_summary[f"{TARGET_CLASS}_images"],
    }
    clean_matches = clean == EXPECTED_CLEAN

    print(f"\ntrain/cis-val overlap: {overlap}")
    print(f"  expected             : {EXPECTED_OVERLAP}  -> "
          f"{'ok' if overlap_matches else 'MISMATCH'}")
    print(f"cis_val_clean         : {clean}")
    print(f"  expected             : {EXPECTED_CLEAN}  -> "
          f"{'ok' if clean_matches else 'MISMATCH'}")

    report = {
        "task": "B1",
        "classes": classes,
        "class_order_rule": "ascending CCT category ID",
        "splits": per_split,
        "total_images": sum(i["images"] for i in per_split.values()),
        "leakage": {
            "why": (
                "Official cis-val shares seq_id values with train. Burst frames in a "
                "sequence are near-duplicates, so a memorised training frame scores "
                "on its twin at validation. cis_val_clean removes every "
                "train-overlapping sequence and is what calibration and model "
                "selection use."
            ),
            "observed_overlap": overlap,
            "expected_overlap": EXPECTED_OVERLAP,
            "overlap_reconciles": overlap_matches,
            "cis_val_clean": clean,
            "expected_cis_val_clean": EXPECTED_CLEAN,
            "clean_reconciles": clean_matches,
            "cis_val_clean_sha256": clean_digest,
        },
        "all_counts_reconcile": all(i["count_reconciles"] for i in per_split.values())
        and overlap_matches
        and clean_matches,
    }

    report_path = args.report or (args.output_dir / "manifest_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {report_path}")

    if not report["all_counts_reconcile"]:
        print(
            "\nFAILED: a count does not reconcile with DESIGN §5.1/§5.3. Either LILA "
            "republished the metadata or this code is wrong. Do not proceed by "
            "adjusting the expected numbers — find out which."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
