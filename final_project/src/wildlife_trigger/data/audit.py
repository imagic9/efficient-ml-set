#!/usr/bin/env python3
"""B4 / Gate B — every assertion in DESIGN §5.3, as a gate that stops the project.

DESIGN §5.3 ends with: *If any assertion fails, stop. Do not train around a split
problem.* This module is that sentence, executable.

It also carries §5.3's own warning, which matters more than any single check here:

> These counts are fingerprints of a specific upstream download, not invariants of the
> universe. If one fails, first check the recorded source hashes from section 5.1: a
> hash change means LILA republished the metadata and the expected numbers must be
> re-derived and re-reviewed. **Never edit an expected number to make a failing
> assertion pass.**

So a failure here prints the source hashes alongside the mismatch — because the first
question is always "did the data change?", and the answer is a file we already recorded.

Usage:
    python -m wildlife_trigger.data.audit --manifests-dir data/manifests \
        --report results/data_audit/gate_b.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

SPLITS = ("train", "cis_val", "cis_test", "trans_val", "trans_test")

EXPECTED_COUNTS = {
    "train": 13553,
    "cis_val": 3484,
    "cis_test": 15827,
    "trans_val": 1725,
    "trans_test": 23275,
}
EXPECTED_MULTI_LABEL = {
    "train": 7,
    "cis_val": 0,
    "cis_test": 1,
    "trans_val": 61,
    "trans_test": 9,
}
EXPECTED_OVERLAP = {"sequences": 224, "cis_val_images": 270, "bobcat_images": 10}
EXPECTED_CLEAN = {"images": 3214, "bobcat_images": 144}

NON_ANIMAL = frozenset({"car", "empty"})
EXPECTED_CLASSES = 16
EXPECTED_ANIMALS = 14

# DESIGN §4: no defensible operating point exists for these on validation.
EXPECTED_NO_SUPPORT = frozenset({"deer", "fox"})
EXPECTED_SINGLE_IMAGE_SUPPORT = "badger"

SUPPLEMENT_MAX_LONG_SIDE = 1024
SHORTCUT_BLOCKING_ACCURACY = 0.75


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests-dir", required=True, type=Path)
    parser.add_argument("--classes-config", required=True, type=Path)
    parser.add_argument("--supplement", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--shortcut-probe", required=True, type=Path)
    parser.add_argument("--dimension-report", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--supplement-dir", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--check-paths",
        type=int,
        default=500,
        help="Verify this many manifest paths per split exist. 0 checks all 57,864; "
        "the default samples, because a full stat() sweep is slow and B0 already "
        "opened every file to measure it.",
    )
    args = parser.parse_args()

    checks: dict[str, dict] = {}

    def record(name: str, passed: bool, detail: object) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}

    splits = {name: load(args.manifests_dir / f"{name}.jsonl") for name in SPLITS}
    clean = load(args.manifests_dir / "cis_val_clean.jsonl")
    supplement = load(args.supplement)

    import yaml

    classes = yaml.safe_load(args.classes_config.read_text())["classes"]
    class_names = [c["name"] for c in sorted(classes, key=lambda c: c["index"])]
    animals = [c["name"] for c in classes if c["selectable_target"]]

    # --- counts and categories ---
    for name in SPLITS:
        record(
            f"count_{name}",
            len(splits[name]) == EXPECTED_COUNTS[name],
            {"observed": len(splits[name]), "expected": EXPECTED_COUNTS[name]},
        )

    record("category_count_is_16", len(class_names) == EXPECTED_CLASSES, class_names)
    record("animal_count_is_14", len(animals) == EXPECTED_ANIMALS, len(animals))
    record(
        "non_animal_classes_are_car_and_empty",
        set(class_names) - set(animals) == NON_ANIMAL,
        sorted(set(class_names) - set(animals)),
    )

    observed_labels = {
        label for records in splits.values() for r in records for label in r["labels"]
    }
    record(
        "labels_are_within_the_frozen_class_set",
        observed_labels <= set(class_names),
        sorted(observed_labels - set(class_names)),
    )

    # --- identity ---
    for name in SPLITS:
        ids = [r["image_id"] for r in splits[name]]
        record(f"unique_ids_{name}", len(ids) == len(set(ids)), len(ids) - len(set(ids)))

    cross_split_dupes = {}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1 :]:
            shared = {r["image_id"] for r in splits[a]} & {r["image_id"] for r in splits[b]}
            if shared:
                cross_split_dupes[f"{a}|{b}"] = len(shared)
    record("no_image_in_two_splits", not cross_split_dupes, cross_split_dupes)

    # --- the known overlap, and every other intersection being zero ---
    train_seqs = {r["seq_id"] for r in splits["train"]}
    overlapping = [r for r in splits["cis_val"] if r["seq_id"] in train_seqs]
    overlap = {
        "sequences": len({r["seq_id"] for r in overlapping}),
        "cis_val_images": len(overlapping),
        "bobcat_images": sum(1 for r in overlapping if "bobcat" in r["labels"]),
    }
    record("known_train_cis_val_overlap", overlap == EXPECTED_OVERLAP,
           {"observed": overlap, "expected": EXPECTED_OVERLAP})

    clean_stats = {
        "images": len(clean),
        "bobcat_images": sum(1 for r in clean if "bobcat" in r["labels"]),
    }
    record("cis_val_clean_fingerprint", clean_stats == EXPECTED_CLEAN,
           {"observed": clean_stats, "expected": EXPECTED_CLEAN})
    record(
        "cis_val_clean_shares_no_sequence_with_train",
        not ({r["seq_id"] for r in clean} & train_seqs),
        len({r["seq_id"] for r in clean} & train_seqs),
    )

    # Every intersection that DESIGN says must be empty. The train/cis-val one is the
    # documented exception and is checked above.
    for a, b in (
        ("train", "cis_test"),
        ("train", "trans_val"),
        ("train", "trans_test"),
        ("cis_val", "cis_test"),
    ):
        shared = {r["seq_id"] for r in splits[a]} & {r["seq_id"] for r in splits[b]}
        record(f"no_sequence_overlap_{a}_{b}", not shared, len(shared))

    # --- locations: the cis/trans premise ---
    train_locations = {r["location"] for r in splits["train"]}
    for other in ("trans_val", "trans_test"):
        shared = train_locations & {r["location"] for r in splits[other]}
        record(f"train_locations_disjoint_from_{other}", not shared, sorted(shared))

    # --- the supplement ---
    cct20_ids = {r["image_id"] for records in splits.values() for r in records}
    cct20_seqs = {r["seq_id"] for records in splits.values() for r in records}
    cct20_locations = {str(r["location"]) for records in splits.values() for r in records}

    record("supplement_size_is_5000", len(supplement) == 5000, len(supplement))
    record(
        "supplement_ids_disjoint",
        not ({str(r["image_id"]) for r in supplement} & cct20_ids),
        len({str(r["image_id"]) for r in supplement} & cct20_ids),
    )
    record(
        "supplement_sequences_disjoint",
        not ({str(r["seq_id"]) for r in supplement} & cct20_seqs),
        len({str(r["seq_id"]) for r in supplement} & cct20_seqs),
    )
    record(
        "supplement_locations_disjoint",
        not ({str(r["location"]) for r in supplement} & cct20_locations),
        sorted({str(r["location"]) for r in supplement} & cct20_locations),
    )
    record(
        "supplement_is_all_empty",
        all(r["labels"] == ["empty"] for r in supplement),
        Counter(tuple(r["labels"]) for r in supplement).most_common(3),
    )

    long_sides = [
        max(r["downsized_width"], r["downsized_height"])
        for r in supplement
        if r.get("downsized_width")
    ]
    record(
        "supplement_downsized_to_1024",
        bool(long_sides) and max(long_sides) <= SUPPLEMENT_MAX_LONG_SIDE,
        {"max_long_side": max(long_sides) if long_sides else None},
    )
    record(
        "supplement_records_both_geometries_and_checksums",
        all(
            r.get("original_width")
            and r.get("downsized_width")
            and r.get("original_sha256")
            and r.get("downsized_sha256")
            for r in supplement
        ),
        "DESIGN §5.2 step 8",
    )

    # The supplement's decoded geometry must look like the `_sm` splits, not like the
    # 2048-px originals it was downloaded from. This is the confound, measured.
    dimension_report = json.loads(args.dimension_report.read_text())
    supplement_sizes = Counter(
        (r["downsized_width"], r["downsized_height"]) for r in supplement
    )
    dominant_supplement, dominant_count = supplement_sizes.most_common(1)[0]
    record(
        "supplement_geometry_consistent_with_sm_splits",
        f"{dominant_supplement[0]}x{dominant_supplement[1]}" == dimension_report["dominant"],
        {
            "supplement_dominant": f"{dominant_supplement[0]}x{dominant_supplement[1]}",
            "supplement_dominant_share": round(dominant_count / len(supplement), 4),
            "cct20_dominant": dimension_report["dominant"],
        },
    )

    probe = json.loads(args.shortcut_probe.read_text())
    record(
        "shortcut_probe_near_chance",
        probe["held_out_accuracy"] < SHORTCUT_BLOCKING_ACCURACY,
        {"accuracy": probe["held_out_accuracy"], "verdict": probe["verdict"]},
    )

    # --- labels and multi-label semantics ---
    for name in SPLITS:
        observed = sum(1 for r in splits[name] if len(r["labels"]) > 1)
        record(
            f"multi_label_count_{name}",
            observed == EXPECTED_MULTI_LABEL[name],
            {"observed": observed, "expected": EXPECTED_MULTI_LABEL[name]},
        )

    record(
        "every_record_carries_an_ordered_label_set",
        all(
            isinstance(r["labels"], list) and r["labels"] == sorted(r["labels"])
            for records in splits.values()
            for r in records
        ),
        "labels must be a sorted list on every record",
    )
    record(
        "primary_label_is_none_exactly_when_multi_class",
        all(
            (r["primary_label"] is None) == (len(r["labels"]) != 1)
            for records in splits.values()
            for r in records
        ),
        "cross-entropy excludes these; target-presence still counts them",
    )

    # --- validation support: which targets can exist at all ---
    validation = clean + splits["trans_val"]
    support = {
        name: {
            "images": sum(1 for r in validation if name in r["labels"]),
            "sequences": len({r["seq_id"] for r in validation if name in r["labels"]}),
        }
        for name in animals
    }
    zero_support = {n for n, s in support.items() if s["images"] == 0}
    record(
        "zero_validation_support_is_exactly_deer_and_fox",
        zero_support == EXPECTED_NO_SUPPORT,
        sorted(zero_support),
    )
    record(
        "badger_support_is_one_image_one_sequence",
        support[EXPECTED_SINGLE_IMAGE_SUPPORT] == {"images": 1, "sequences": 1},
        support[EXPECTED_SINGLE_IMAGE_SUPPORT],
    )

    # --- paths ---
    missing = []
    for name in SPLITS:
        sample = splits[name] if args.check_paths == 0 else splits[name][: args.check_paths]
        for r in sample:
            if not (args.images_dir / r["file_name"]).exists():
                missing.append(f"{name}:{r['file_name']}")
    for r in supplement[: args.check_paths or len(supplement)]:
        if not (args.supplement_dir / r["file_name"]).exists():
            missing.append(f"supplement:{r['file_name']}")
    record("manifest_paths_exist", not missing, missing[:5])

    passed = all(c["passed"] for c in checks.values())
    sources = json.loads(args.source_manifest.read_text())

    report = {
        "gate": "B",
        "passed": passed,
        "checks": checks,
        "failed_checks": sorted(k for k, v in checks.items() if not v["passed"]),
        "validation_support": support,
        "class_distribution": {
            name: Counter(
                label for r in splits[name] for label in r["labels"]
            ).most_common()
            for name in SPLITS
        },
        "location_distribution": {
            name: Counter(str(r["location"]) for r in splits[name]).most_common()
            for name in SPLITS
        },
        "source_hashes": {s["name"]: s["sha256"] for s in sources["sources"]},
    }

    for name, check in checks.items():
        print(f"    {'PASS' if check['passed'] else 'FAIL'}  {name}")
    print()
    print(f"GATE B {'PASSED' if passed else 'FAILED'}")
    if not passed:
        print(f"  failed: {', '.join(report['failed_checks'])}")
        print()
        print("  DESIGN §5.3: these counts fingerprint a specific upstream download.")
        print("  Check the recorded source hashes FIRST — a change means LILA")
        print("  republished and the expected numbers must be re-derived and reviewed.")
        print("  Never edit an expected number to make an assertion pass.")
        for name, digest in report["source_hashes"].items():
            print(f"    {name}: {digest}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(f"  wrote {args.report}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
