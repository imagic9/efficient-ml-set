#!/usr/bin/env python3
"""E6 reduced-decode accuracy / decision-drift check (PLAN E6, DESIGN §11).

The optimization matrix offers a cheap decode: `IMREAD_REDUCED_COLOR_2/4` lets
libjpeg emit the frame at 1/2 or 1/4 each side straight from the DCT coefficients,
which is far cheaper than a full decode plus resize. But it is **not preprocessing
parity** — it changes the pixels the letterbox resamples, so it changes the tensor
and can change the model's score. DESIGN §11 is explicit: reduced decode is kept
**only if** the validation bobcat metrics hold within a predeclared tolerance.

This turns that rule into a gate. For each model we run the C++ dataset runner over
the same benchmark manifest three times — full, half, quarter decode — and compare
each reduced variant against the model's own **full-decode** run (the certified
pipeline), plus against the ground-truth labels:

  - the safety-critical direction is **lost detections**: a frame the full pipeline
    fires on that a reduced decode misses. Missing the target animal is the product's
    core failure mode, so the tolerance here is **zero** — one lost true bobcat
    detection rejects the variant;
  - **new false fires** (a non-bobcat frame the reduced decode fires on that full
    does not) waste the shutter and battery but do not miss the animal, so a small
    predeclared fraction is tolerated;
  - **absolute** recall / precision against the labels are reported for full and each
    variant, so the reader sees the ground-truth picture, not only the drift;
  - **score deltas** vs full are reported as diagnostics.

A variant that passes is a *candidate* the report may choose to ship; the gate only
certifies that it did not silently degrade detection. Latency is NOT judged here —
the decode saving is measured (diagnostic, off-Pi) by the optimization matrix; only a
Pi measures a latency that counts (DESIGN §12.4).

Usage (gx10, after scripts/run_e6_decode_drift.sh has produced the JSONL):
    python -m wildlife_trigger.validate.decode_drift \\
        --dir results/e6/decode_drift \\
        --models M0:0.5381 M2:0.6504 M4:0.3730 \\
        --manifest data/manifests/benchmark_val_1000.jsonl \\
        --variants half quarter --target bobcat \\
        --output results/e6/decode_drift.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..runs import atomic_write_json

# Predeclared tolerances (DESIGN §11: the metrics must hold within a *predeclared*
# tolerance, fixed here before the numbers are seen, not fitted to them afterwards).
MAX_LOST_TRUE_DETECTIONS = 0        # a missed bobcat is the core failure; none allowed
MAX_NEW_FALSE_FIRE_FRAC = 0.01      # ≤1% of frames may become new false fires


def read_predictions(path: Path) -> dict[str, dict]:
    """The scored rows of a run-dataset JSONL, keyed by image_id. Refuses a file
    that did not finish: a missing footer means partial coverage, and a drift
    conclusion drawn from a truncated run proves nothing."""
    lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if not lines or lines[0].get("kind") != "run_dataset_header":
        raise RuntimeError(f"{path} does not start with a run_dataset_header")
    if lines[-1].get("kind") != "run_dataset_footer":
        raise RuntimeError(f"{path} has no footer; the run did not finish")
    return {
        r["image_id"]: r
        for r in lines[1:-1]
        if "image_id" in r and "target_scores" in r and not r.get("skipped")
    }


def confusion(fire: np.ndarray, present: np.ndarray) -> dict:
    tp = int((fire & present).sum())
    fp = int((fire & ~present).sum())
    fn = int((~fire & present).sum())
    tn = int((~fire & ~present).sum())
    recall = tp / (tp + fn) if (tp + fn) else None
    precision = tp / (tp + fp) if (tp + fp) else None
    return {
        "true_fire": tp, "false_fire": fp, "missed": fn, "true_quiet": tn,
        "recall": recall, "precision": precision,
    }


def compare_model(model_id: str, full_path: Path, variant_paths: dict[str, Path],
                  threshold: float, manifest_labels: dict[str, list], target: str) -> dict:
    full = read_predictions(full_path)
    ids = sorted(full)
    present = np.array([target in manifest_labels[i] for i in ids])
    full_fire = np.array([bool(full[i]["shutter_trigger"]) for i in ids])
    full_score = np.array([full[i]["target_scores"][target] for i in ids], dtype=float)

    variants = []
    model_failures: list[str] = []
    for decode, vpath in variant_paths.items():
        var = read_predictions(vpath)
        failures: list[str] = []

        missing = [i for i in ids if i not in var]
        if missing:
            failures.append(f"{len(missing)} frames present in full but absent in {decode}")
        common = [i for i in ids if i in var]
        idx = np.array([i in var for i in ids])

        var_fire = np.array([bool(var[i]["shutter_trigger"]) for i in common])
        var_score = np.array([var[i]["target_scores"][target] for i in common], dtype=float)
        cf = full_fire[idx]
        cs = full_score[idx]
        cp = present[idx]

        # Drift vs the model's own full-decode run (the certified pipeline).
        lost = cf & cp & ~var_fire            # full fired on a real bobcat; variant missed
        new_false = ~cf & ~cp & var_fire      # variant fires on a non-bobcat full let pass
        flips = cf != var_fire
        lost_ids = [common[k] for k in np.flatnonzero(lost)]
        new_false_frac = float(new_false.sum()) / len(common) if common else 0.0

        gap = np.abs(var_score - cs)
        bob = cp
        score_gap = {
            "max": float(gap.max()) if gap.size else 0.0,
            "mean": float(gap.mean()) if gap.size else 0.0,
            "p95": float(np.percentile(gap, 95)) if gap.size else 0.0,
            "max_on_bobcat": float(gap[bob].max()) if bob.any() else 0.0,
        }

        if len(lost_ids) > MAX_LOST_TRUE_DETECTIONS:
            failures.append(
                f"{len(lost_ids)} true bobcat detection(s) lost vs full decode "
                f"(tolerance {MAX_LOST_TRUE_DETECTIONS}): {lost_ids[:8]}"
            )
        if new_false_frac > MAX_NEW_FALSE_FIRE_FRAC:
            failures.append(
                f"new false fires {new_false_frac:.3%} exceed the "
                f"{MAX_NEW_FALSE_FIRE_FRAC:.0%} tolerance ({int(new_false.sum())} frames)"
            )

        variants.append({
            "decode": decode,
            "frames": len(common),
            "confusion_full": confusion(cf, cp),
            "confusion_variant": confusion(var_fire, cp),
            "decision_flips": int(flips.sum()),
            "lost_true_detections": len(lost_ids),
            "lost_detection_ids": lost_ids[:16],
            "new_false_fires": int(new_false.sum()),
            "new_false_fire_frac": new_false_frac,
            "score_gap_vs_full": score_gap,
            "keep": not failures,
            "failures": failures,
        })
        model_failures.extend(f"[{decode}] {m}" for m in failures)

    return {
        "model": model_id,
        "threshold": threshold,
        "frames": len(ids),
        "bobcat_frames": int(present.sum()),
        "confusion_full": confusion(full_fire, present),
        "variants": variants,
        "passed": not model_failures,
        "failures": model_failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, type=Path,
                        help="holds {model}_{full,half,quarter}.jsonl from run-dataset")
    parser.add_argument("--models", nargs="+", required=True,
                        help="model_id:threshold, one per shortlisted model")
    parser.add_argument("--variants", nargs="+", default=["half", "quarter"])
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--target", default="bobcat")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest_labels = {}
    with args.manifest.open() as handle:
        for line in handle:
            if line.strip():
                r = json.loads(line)
                manifest_labels[r["image_id"]] = r["labels"]

    results = []
    for spec in args.models:
        model_id, threshold = spec.split(":")
        results.append(compare_model(
            model_id,
            args.dir / f"{model_id}_full.jsonl",
            {v: args.dir / f"{model_id}_{v}.jsonl" for v in args.variants},
            float(threshold), manifest_labels, args.target,
        ))

    passed = all(r["passed"] for r in results)
    report = {
        "gate": "E6 reduced-decode accuracy / decision-drift (DESIGN §11, PLAN E6)",
        "target": args.target,
        "manifest": str(args.manifest),
        "tolerances": {
            "max_lost_true_detections": MAX_LOST_TRUE_DETECTIONS,
            "max_new_false_fire_frac": MAX_NEW_FALSE_FIRE_FRAC,
            "note": "reduced decode is NOT parity; kept only if bobcat metrics hold. "
                    "Latency is not judged here (DESIGN §12.4).",
        },
        "models": results,
        "verdict": {"passed": passed, "failed": [r["model"] for r in results if not r["passed"]]},
    }
    atomic_write_json(args.output, report)

    for r in results:
        cf = r["confusion_full"]
        print(f"{r['model']}: {r['frames']} frames, {r['bobcat_frames']} bobcat; "
              f"full recall={cf['recall']} precision={cf['precision']}")
        for v in r["variants"]:
            cv = v["confusion_variant"]
            print(f"    {v['decode']:>7}: {'KEEP' if v['keep'] else 'REJECT'} — "
                  f"recall={cv['recall']} precision={cv['precision']}, "
                  f"flips={v['decision_flips']}, lost={v['lost_true_detections']}, "
                  f"new_false={v['new_false_fires']} ({v['new_false_fire_frac']:.2%}), "
                  f"score Δ max={v['score_gap_vs_full']['max']:.3f}")
            for f in v["failures"]:
                print(f"        FAIL: {f}")
    print(f"E6 decode-drift {'PASSED' if passed else 'FAILED'}; wrote {args.output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
