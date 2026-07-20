#!/usr/bin/env python3
"""F5 reproducibility — compare the unchanged frozen Pi repeat against F4 (PLAN F5).

The F5 run re-executed the exact frozen Pi benchmark + parity slice on the real CM5 with no
change (deploy/pi/run_f4_pi_benchmark.sh ... f5). This compares F5 vs F4 latency per model and
re-checks the F5 parity slice against the SAME frozen gx10 reference (results/f4/gx10_parity_*),
which is deterministic. Investigates only measurement spread; no tuning. Emits
results/f5/f5_reproducibility.json. All numbers generated.

Usage:  python3 scripts/summarize_f5.py
"""
import json
import statistics
from pathlib import Path

from wildlife_trigger.validate.p4_dataset_parity import load_cpp_jsonl

F4 = Path("results/f4")
F5 = Path("results/f5")
MODELS = ["M0", "M2", "M4"]
TARGET = "bobcat"


def mean_p50(root: Path, m: str):
    vals = []
    for r in (1, 2, 3):
        p = root / f"bench_{m}_rep{r}.json"
        if p.exists():
            vals.append(json.loads(p.read_text())["stages_ms"]["end_to_end"]["p50"])
    return (statistics.mean(vals), min(vals), max(vals), len(vals)) if vals else None


f4_sum = json.loads((F4 / "f4_summary.json").read_text())
latency = {}
for m in MODELS:
    f5 = mean_p50(F5, m)
    if not f5:
        continue
    f4_mean = f4_sum["pi_latency"].get(m, {}).get("e2e_ms_p50_mean")
    f5_mean = round(f5[0], 3)
    latency[m] = {
        "f4_mean_ms": f4_mean,
        "f5_mean_ms": f5_mean,
        "f5_min_ms": round(f5[1], 3),
        "f5_max_ms": round(f5[2], 3),
        "delta_ms": round(f5_mean - f4_mean, 3) if f4_mean else None,
        "delta_pct": round(100 * (f5_mean - f4_mean) / f4_mean, 2) if f4_mean else None,
    }

# parity re-check: F5 Pi predictions vs the frozen gx10 reference (deterministic)
def rows_by_id(path):
    _, rows, _ = load_cpp_jsonl(path)
    return {r["image_id"]: r for r in rows if not r.get("skipped")}


parity = {}
for m in ("M0", "M2"):
    pi = rows_by_id(F5 / f"pi_parity_{m}.jsonl")
    gx = rows_by_id(F4 / f"gx10_parity_{m}.jsonl")
    ids = sorted(set(pi) & set(gx))
    maxd = max(abs(float(pi[i]["target_scores"][TARGET]) - float(gx[i]["target_scores"][TARGET])) for i in ids)
    dis = sum(1 for i in ids if bool(pi[i]["shutter_trigger"]) != bool(gx[i]["shutter_trigger"]))
    parity[m] = {"frames": len(ids), "max_score_delta": maxd, "decision_disagreements": dis,
                 "verdict": "PASS" if dis == 0 else "DISAGREE"}

repro_ok = all(v["verdict"] == "PASS" for v in parity.values()) and \
           all(abs(v["delta_pct"]) <= 5.0 for v in latency.values() if v["delta_pct"] is not None)

summary = {
    "kind": "f5_reproducibility",
    "schema_version": 1,
    "design": "PLAN F5",
    "note": "Unchanged frozen Pi repeat vs F4; latency spread only, no tuning. Parity re-checked "
            "against the deterministic frozen gx10 reference.",
    "latency_f4_vs_f5": latency,
    "parity_f5_vs_gx10_reference": parity,
    "reproducible": repro_ok,
}
out = F5 / "f5_reproducibility.json"
out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(f"wrote {out}")
print("\n== latency reproducibility (Pi p50, F4 vs F5) ==")
for m, v in latency.items():
    print(f"  {m}: F4 {v['f4_mean_ms']:.2f}ms  F5 {v['f5_mean_ms']:.2f}ms  Δ {v['delta_ms']:+.2f}ms ({v['delta_pct']:+.1f}%)")
print("== parity (F5 Pi vs frozen gx10 ref) ==")
for m, v in parity.items():
    print(f"  {m}: maxΔ={v['max_score_delta']:.2e} disagree={v['decision_disagreements']} -> {v['verdict']}")
print(f"\nREPRODUCIBLE: {repro_ok}")
