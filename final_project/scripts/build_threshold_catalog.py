#!/usr/bin/env python3
"""F3 threshold catalog — the 14 animal-class status entries for the final model (M2).

DESIGN §4/§6.3 & PLAN F3: after the final model is selected, record a status entry for
every animal class — a numeric threshold for the selectable targets, null for the three
with insufficient validation support (badger/deer/fox). The threshold is the frozen §6.3
rule verbatim via `metrics.select_threshold`, reusing the D-phase loader
(`optimize.calibrate_candidate.load_candidate`) READ-ONLY: unlike the CLI this writes NO
`calibration.json` and does not touch the frozen bobcat policy. Also emits the multi-target
example policy `bobcat_coyote_v1` (DESIGN §4 generic-target feature), bound by hash to M2.

Real numbers only, generated from the M2 candidate's ORT validation scores — never
hand-typed (PLAN §1). Test labels stay sealed (load_candidate can only reach validation).

Usage:  python3 scripts/build_threshold_catalog.py
"""
import hashlib
import json
from pathlib import Path

from wildlife_trigger import metrics
from wildlife_trigger.optimize.calibrate_candidate import load_candidate
from wildlife_trigger.policy import (
    ANIMAL_CLASSES,
    NO_THRESHOLD_CLASSES,
    build_policy,
)

CAND = Path("results/optimize/m2_qat/lr5e-5")
ARTIFACTS = Path("artifacts")
OUT = Path("results/f3")
OUT.mkdir(parents=True, exist_ok=True)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class_map = json.loads((ARTIFACTS / "class_map.json").read_text())
class_map_sha256 = sha256_file(ARTIFACTS / "class_map.json")
model_path = CAND / "model.onnx"
model_sha256 = sha256_file(model_path)

KEEP = ("sequence_balanced_recall", "false_fire_rate", "fire_rate", "frame_recall", "frame_f2")

entries = []
thresholds = {}
for cls in ANIMAL_CLASSES:
    if cls in NO_THRESHOLD_CLASSES:
        entries.append({
            "target": cls,
            "selectable": False,
            "threshold": None,
            "status": "unavailable_insufficient_validation_support",
            "note": "DESIGN §4: no defensible operating point (badger 1 val image; deer/fox 0).",
        })
        continue
    scores_by_domain, _ = load_candidate(CAND, cls)
    sel = metrics.select_threshold(scores_by_domain)
    thresholds[cls] = sel["threshold"]
    entries.append({
        "target": cls,
        "selectable": True,
        "threshold": sel["threshold"],
        "status": sel["status"],
        "per_domain": {
            d: {k: round(float(m[k]), 6) for k in KEEP if k in m}
            for d, m in sel["per_domain"].items()
        },
    })

catalog = {
    "kind": "threshold_catalog",
    "schema_version": 1,
    "design": "DESIGN §4/§6.3, PLAN F3",
    "final_model": "M2 (int8_qat)",
    "candidate": str(CAND),
    "model_sha256": model_sha256,
    "class_map_sha256": class_map_sha256,
    "rule": "frozen §6.3: largest threshold inside the 5% per-domain false-fire budget "
            "meeting the 90% sequence-balanced recall floor on both domains; status recorded "
            "verbatim. recall_floor_infeasible ships an operating point and is NOT a pass.",
    "counts": {
        "animal_classes": len(ANIMAL_CLASSES),
        "selectable": sum(1 for e in entries if e["selectable"]),
        "unavailable": sum(1 for e in entries if not e["selectable"]),
    },
    "entries": entries,
}
cat_path = OUT / "threshold_catalog.json"
cat_path.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")
print(f"wrote {cat_path}: {catalog['counts']}")

# --- multi-target example: bobcat + coyote, bound to M2 (DESIGN §4) ---
bobcat_policy = json.loads((ARTIFACTS / "policies" / "bobcat_m2_qat_lr5e-5_v1.json").read_text())
bobcat_threshold = next(t["threshold"] for t in bobcat_policy["targets"] if t["class"] == "bobcat")
targets = [
    {"class": "bobcat", "threshold": bobcat_threshold},
    {"class": "coyote", "threshold": thresholds["coyote"]},
]
policy = build_policy(
    policy_id="bobcat_coyote_v1",
    targets=targets,
    class_map=class_map,
    class_map_sha256=class_map_sha256,
    model_sha256=model_sha256,
    metadata={
        "note": "DESIGN §4 generic multi-target example on the final model M2; both targets "
                "recall_floor_infeasible individually (see threshold_catalog.json). mode 'any' "
                "fires if EITHER target clears its own threshold.",
        "calibration": "results/f3/threshold_catalog.json",
        "status_bobcat": "recall_floor_infeasible",
        "status_coyote": "recall_floor_infeasible",
    },
)
pol_path = ARTIFACTS / "policies" / "bobcat_coyote_v1.json"
pol_path.write_text(json.dumps(policy, indent=2) + "\n")
print(f"wrote {pol_path}: targets bobcat@{bobcat_threshold:.4f} + coyote@{thresholds['coyote']:.4f}")

# --- human summary ---
print("\n== threshold catalog (final model M2) ==")
for e in entries:
    if e["selectable"]:
        cis = e["per_domain"].get("cis_val_clean", {})
        tr = e["per_domain"].get("trans_val", {})
        print(f"  {e['target']:12s} thr={e['threshold']:.4f} {e['status']:24s} "
              f"cisFF={cis.get('false_fire_rate', float('nan')):.4f} "
              f"trFF={tr.get('false_fire_rate', float('nan')):.4f}")
    else:
        print(f"  {e['target']:12s} thr=NULL   {e['status']}")
