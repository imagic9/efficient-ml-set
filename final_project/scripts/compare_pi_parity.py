#!/usr/bin/env python3
"""F4 Pi-vs-gx10 parity comparison (PLAN F4, DESIGN §12.4).

The frozen deployment ran run-dataset on the SAME bundled parity slice on the real Pi
(pi_parity_<M>.jsonl) and on the gx10 reference in the target container (gx10_parity_<M>.jsonl).
This diffs them frame by frame: bobcat score delta and shutter decision agreement, per model.
Records score deltas, not only decision agreement (§12.4), so drift that has not yet crossed a
threshold stays visible. Expect the FP32 baseline (M0) to be the one that could move (float
accumulation order differs between the gx10 X925 SVE2/i8mm path and the Pi A76 NEON path); the
INT8 winner (M2) accumulates exactly in int32 and should be bit-identical.

Usage:  python3 scripts/compare_pi_parity.py [results/f4]
"""
import json
import sys
from pathlib import Path

from wildlife_trigger.validate.p4_dataset_parity import load_cpp_jsonl

F4 = Path(sys.argv[1] if len(sys.argv) > 1 else "results/f4")
TARGET = "bobcat"


def rows_by_id(path):
    _, rows, _ = load_cpp_jsonl(path)
    return {r["image_id"]: r for r in rows if not r.get("skipped")}


summary = {"kind": "f4_pi_vs_gx10_parity", "schema_version": 1,
           "design": "PLAN F4 / DESIGN §12.4", "target": TARGET, "models": {}}
overall_ok = True

for M in ("M0", "M2"):
    pi = rows_by_id(F4 / f"pi_parity_{M}.jsonl")
    gx = rows_by_id(F4 / f"gx10_parity_{M}.jsonl")
    ids = sorted(set(pi) & set(gx))
    assert ids, f"no common frames for {M}"
    max_delta = 0.0
    sum_delta = 0.0
    decision_disagree = 0
    worst = None
    for iid in ids:
        ps = float(pi[iid]["target_scores"][TARGET])
        gs = float(gx[iid]["target_scores"][TARGET])
        d = abs(ps - gs)
        sum_delta += d
        if d > max_delta:
            max_delta, worst = d, iid
        if bool(pi[iid]["shutter_trigger"]) != bool(gx[iid]["shutter_trigger"]):
            decision_disagree += 1
    entry = {
        "frames": len(ids),
        "max_score_delta": max_delta,
        "mean_score_delta": sum_delta / len(ids),
        "worst_frame": worst,
        "decision_disagreements": decision_disagree,
        "pi_fired": sum(1 for i in ids if pi[i]["shutter_trigger"]),
        "gx10_fired": sum(1 for i in ids if gx[i]["shutter_trigger"]),
        "verdict": "PASS" if decision_disagree == 0 else "DISAGREE",
    }
    summary["models"][M] = entry
    overall_ok = overall_ok and decision_disagree == 0
    print(f"  {M}: frames={entry['frames']} maxΔ={max_delta:.3e} meanΔ={entry['mean_score_delta']:.3e} "
          f"decisions_disagree={decision_disagree} pi_fired={entry['pi_fired']} "
          f"gx10_fired={entry['gx10_fired']} -> {entry['verdict']}")

summary["overall_verdict"] = "PASS" if overall_ok else "DISAGREE — stop before claiming Pi equivalence (§12.4)"
out = F4 / "parity_comparison.json"
out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(f"wrote {out}: {summary['overall_verdict']}")
sys.exit(0 if overall_ok else 1)
