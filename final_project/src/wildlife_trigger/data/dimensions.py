#!/usr/bin/env python3
"""B0 — measure the observed image-dimension distribution of every split.

PLAN B0 is emphatic: *neither may be inherited from the paper or from DESIGN*. Two
load-bearing arguments rest on these numbers and both would fail quietly if the
assumed frame were wrong:

  - **the input-shape choice.** DESIGN §5.5 prefers 256x192 over 224x224 because the
    dominant frame's aspect ratio wastes 27% of a square letterbox on grey bars. That
    is an argument about the aspect ratio of the frames we actually decode.
  - **the reduced-decode alignment.** `1024 / 4 = 256` exactly, so libjpeg's 1/4 scaled
    decode emits the network input with no resize step. That only holds if the long
    side really is 1024.

The annotation JSON cannot answer this. It records the **original** geometry —
2048x1494 — while the `_sm` archive this pipeline consumes is capped at 1024 px per
side. Reading dimensions from the JSON would confirm DESIGN's aspect-ratio claim while
being wrong about every absolute number, and the reduced-decode alignment would be
checked against a frame we never decode. So this opens the actual files.

Only the JPEG header is read (Pillow's `Image.open` is lazy; `.size` does not decode
pixels), which makes 57,864 files a matter of seconds rather than minutes.

The per-image index it writes is reused by B1, so the manifests carry both geometries
without a second pass over the archive.

Usage:
    python -m wildlife_trigger.data.dimensions --images-dir data/raw/extracted \
        --annotations-dir data/raw/extracted/eccv_18_annotation_files \
        --output data/manifests/dimensions.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

SPLITS = ("train", "cis_val", "cis_test", "trans_val", "trans_test")

# DESIGN §5.5's claims, checked rather than trusted. The `_sm` archive caps the long
# side at 1024; the dominant original frame is 2048x1494, so the dominant decoded frame
# should be 1024x747.
EXPECTED_SM_LONG_SIDE = 1024
EXPECTED_DOMINANT_SM = (1024, 747)
EXPECTED_DOMINANT_SHARE = 0.91


def find_image(images_dir: Path, file_name: str) -> Path | None:
    """Locate a frame in the extracted `_sm` tree.

    The archive's internal layout is its own business and has changed between LILA
    rehostings, so the direct path is tried first and a recursive search is the
    fallback rather than the default.
    """
    direct = images_dir / file_name
    if direct.exists():
        return direct
    matches = list(images_dir.rglob(file_name))
    return matches[0] if matches else None


def scan_split(
    split: str, annotations_dir: Path, images_dir: Path
) -> tuple[dict[str, list[int]], dict]:
    """Return {image_id: [width, height]} observed, plus a report for this split."""
    document = json.loads((annotations_dir / f"{split}_annotations.json").read_text())

    observed: dict[str, list[int]] = {}
    declared: dict[str, list[int]] = {}
    observed_counter: Counter = Counter()
    declared_counter: Counter = Counter()
    missing: list[str] = []

    for image in document["images"]:
        declared_size = (image["width"], image["height"])
        declared[image["id"]] = list(declared_size)
        declared_counter[declared_size] += 1

        path = find_image(images_dir, image["file_name"])
        if path is None:
            missing.append(image["file_name"])
            continue
        try:
            with Image.open(path) as handle:
                size = handle.size  # (width, height), header only
        except Exception as exc:  # a truncated frame is a finding, not a crash
            missing.append(f"{image['file_name']}: {type(exc).__name__}: {exc}")
            continue

        observed[image["id"]] = list(size)
        observed_counter[size] += 1

    total = sum(observed_counter.values())
    dominant, dominant_count = (
        observed_counter.most_common(1)[0] if observed_counter else ((0, 0), 0)
    )

    report = {
        "split": split,
        "images_in_annotations": len(document["images"]),
        "images_measured": total,
        "missing_or_unreadable": len(missing),
        "missing_examples": missing[:5],
        "observed_sm": {
            "distribution": {
                f"{w}x{h}": n for (w, h), n in observed_counter.most_common(12)
            },
            "distinct_sizes": len(observed_counter),
            "dominant": f"{dominant[0]}x{dominant[1]}",
            "dominant_share": round(dominant_count / total, 4) if total else 0.0,
            "max_long_side": max(
                (max(w, h) for (w, h) in observed_counter), default=0
            ),
        },
        "declared_original": {
            "distribution": {
                f"{w}x{h}": n for (w, h), n in declared_counter.most_common(6)
            },
            "dominant": (
                f"{declared_counter.most_common(1)[0][0][0]}x"
                f"{declared_counter.most_common(1)[0][0][1]}"
                if declared_counter
                else "n/a"
            ),
        },
    }
    return observed, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--annotations-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    index: dict[str, list[int]] = {}
    reports = []
    for split in SPLITS:
        print(f"scanning {split} ...", flush=True)
        observed, report = scan_split(split, args.annotations_dir, args.images_dir)
        index.update(observed)
        reports.append(report)
        print(
            f"  {report['images_measured']:6d} measured  "
            f"dominant {report['observed_sm']['dominant']} "
            f"({report['observed_sm']['dominant_share']:.1%})  "
            f"declared original {report['declared_original']['dominant']}  "
            f"missing {report['missing_or_unreadable']}"
        )

    # The whole-corpus view: what the pipeline actually decodes.
    everything: Counter = Counter(tuple(size) for size in index.values())
    total = sum(everything.values())
    dominant, dominant_count = everything.most_common(1)[0]
    dominant_share = dominant_count / total
    max_long_side = max(max(w, h) for (w, h) in everything)

    checks = {
        "sm_long_side_capped_at_1024": max_long_side <= EXPECTED_SM_LONG_SIDE,
        "dominant_matches_design_5_5": list(dominant) == list(EXPECTED_DOMINANT_SM),
        "dominant_share_at_least_design_claim": dominant_share
        >= EXPECTED_DOMINANT_SHARE - 0.02,
        "every_image_measured": all(r["missing_or_unreadable"] == 0 for r in reports),
    }

    summary = {
        "task": "B0",
        "measured_from": "the extracted _sm JPEG headers, not the annotation JSON",
        "why": (
            "The annotation JSON records the ORIGINAL geometry (2048x1494). The _sm "
            "archive is capped at 1024 px per side, so the JSON cannot answer what "
            "this pipeline decodes. DESIGN §5.5's input-shape argument and the "
            "reduced-decode alignment both rest on the decoded frame."
        ),
        "total_images_measured": total,
        "distinct_sizes": len(everything),
        "dominant": f"{dominant[0]}x{dominant[1]}",
        "dominant_share": round(dominant_share, 4),
        "max_long_side": max_long_side,
        "top_sizes": {f"{w}x{h}": n for (w, h), n in everything.most_common(12)},
        "design_5_5_expectations": {
            "dominant_sm_frame": f"{EXPECTED_DOMINANT_SM[0]}x{EXPECTED_DOMINANT_SM[1]}",
            "dominant_share_claim": EXPECTED_DOMINANT_SHARE,
            "sm_long_side_cap": EXPECTED_SM_LONG_SIDE,
        },
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "per_split": reports,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(index, sort_keys=True) + "\n")
    print(f"\nwrote per-image dimension index: {args.output} ({len(index)} images)")

    report_path = args.report or args.output.with_name("dimension_report.json")
    report_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\ncorpus: {total} images, dominant {summary['dominant']} "
          f"({dominant_share:.1%}), max long side {max_long_side}")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"wrote {report_path}")

    return 0 if summary["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
