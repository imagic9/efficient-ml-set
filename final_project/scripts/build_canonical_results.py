#!/usr/bin/env python3
"""G1 — the canonical results table (PLAN G1, DESIGN §17).

One machine-readable index every downstream artifact (G2 notebook, G3 report, G4 slides)
reads instead of re-deriving numbers. Consolidates, from committed raw result files only:
- the optimization ladder M0-M4 (validation operating point, size, MACs) from comparison.jsonl;
- the real-Pi frozen benchmark + frozen full-test + reproducibility + parity (results/f{2,4,5});
- seed variability for M0 (C5 seeds 17/42/73) and the final M2 (F3 seeds 17/42/73);
- model cards.
Missing/unavailable fields are recorded explicitly, never invented (PLAN §1, DESIGN §9.2).

Usage:  python3 scripts/build_canonical_results.py
"""
import glob
import json
import statistics
from pathlib import Path

R = Path("results")
OUT = Path("results/analysis"); OUT.mkdir(parents=True, exist_ok=True)


def jload(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


# --- optimization ladder (validation, deployment ORT) ---
ladder = {}
for line in (R / "model_selection/comparison.jsonl").read_text().splitlines():
    d = json.loads(line)
    op = d.get("operating_point", {})
    pd = op.get("per_domain", {})
    v = d.get("validation_at_0p5", {})
    ladder[d["model_id"]] = {
        "kind": d["kind"],
        "params": d.get("params"),
        "macs": d.get("macs"),
        "bytes": d["model"]["bytes"],
        "mb": round(d["model"]["bytes"] / 1e6, 2),
        "onnx_sha256": d["model"]["sha256"],
        "threshold": op.get("threshold"),
        "status": op.get("status"),
        "selection_score_seed42": v.get("selection_score"),
        "cis_f2_at_0p5": v.get("cis_f2"),
        "trans_f2_at_0p5": v.get("trans_f2"),
        "operating_point": {
            dom: {k: pd[dom].get(k) for k in
                  ("event_capture_rate", "false_fire_rate", "frame_f2",
                   "sequence_balanced_recall", "frame_recall")}
            for dom in pd
        },
    }

# --- seed variability (primary = mean bobcat F2 @0.5) ---
def primary_of(summary_path):
    s = jload(summary_path)
    if not s:
        return None
    bss = s.get("best_selection_score", {})
    return bss.get("primary") if isinstance(bss, dict) else s.get("best_score")


def seed_block(seed42, glob17, glob73):
    vals = {"seed42": seed42}
    for tag, pat in (("seed17", glob17), ("seed73", glob73)):
        hits = sorted(glob.glob(pat))
        vals[tag] = round(primary_of(hits[0]), 6) if hits else None
    present = [x for x in vals.values() if isinstance(x, (int, float))]
    return {
        "primary_metric": "mean_bobcat_frame_f2_at_0.5",
        "per_seed": vals,
        "mean": round(statistics.mean(present), 6) if present else None,
        "std": round(statistics.pstdev(present), 6) if len(present) > 1 else None,
    }

seeds = {
    "M0_fp32": seed_block(
        ladder["M0"]["selection_score_seed42"],
        "results/training/c5/c5_m0_fp32_seed17_*/run_summary.json",
        "results/training/c5/c5_m0_fp32_seed73_*/run_summary.json"),
    "M2_int8_qat": seed_block(
        ladder["M2"]["selection_score_seed42"],
        "results/optimize/m2_qat_seed17/runs/d2/*/run_summary.json",
        "results/optimize/m2_qat_seed73/runs/d2/*/run_summary.json"),
}

# --- Pi + frozen test + reproducibility + parity ---
f4 = jload(R / "f4/f4_summary.json") or {}
f5 = jload(R / "f5/f5_reproducibility.json") or {}
parity = jload(R / "f4/parity_comparison.json") or {}
frozen = {m: jload(R / f"f4/frozen_test_{m}.json") for m in ("M0", "M2")}

canonical = {
    "kind": "canonical_results",
    "schema_version": 1,
    "design": "PLAN G1 / DESIGN §17",
    "final_model": "M2 (int8_qat)",
    "baseline": "M0 (fp32_baseline)",
    "sources": {
        "ladder": "results/model_selection/comparison.jsonl",
        "pi_latency": "results/f4/f4_summary.json",
        "frozen_test": "results/f4/frozen_test_{M0,M2}.json",
        "reproducibility": "results/f5/f5_reproducibility.json",
        "parity": "results/f4/parity_comparison.json",
        "seeds": "results/training/c5 (M0), results/optimize/m2_qat_seed{17,73} (M2)",
        "model_cards": "artifacts/model_cards/*.md",
    },
    "optimization_ladder_validation": ladder,
    "seed_variability": seeds,
    "pi_latency_frozen": f4.get("pi_latency"),
    "pi_speedups": f4.get("pi_speedups"),
    "frozen_full_test": {
        m: {"frozen_threshold": frozen[m]["frozen_threshold"],
            "per_domain": {dom: {k: round(v[k], 4) for k in
                                 ("frame_f2", "sequence_balanced_recall", "event_capture_rate",
                                  "false_fire_rate", "frame_recall", "frame_precision",
                                  "positive_frames", "frames_scored") if k in v}
                           for dom, v in frozen[m]["per_domain"].items()}}
        for m in ("M0", "M2") if frozen[m]
    },
    "reproducibility_f4_vs_f5": f5.get("latency_f4_vs_f5"),
    "pi_vs_gx10_parity": {"verdict": parity.get("overall_verdict"), "detail": parity.get("models")},
    "unavailable": {
        "pi_latency_M1_M3": "not measured on the Pi — M1/M3 were dropped from the shortlist "
                            "pre-Pi (dominated); only M0/M2/M4 were carried to the CM5.",
        "catalog_null_targets": ["badger", "deer", "fox"],
        "catalog_note": "no defensible operating point (badger 1 val image; deer/fox 0). "
                        "See results/f3/threshold_catalog.json.",
        "cpu_affinity": "not exposed in the benchmark environment (recorded not-measured at E6).",
        "energy_power": "not measured — no power instrumentation on the rented Pi (DESIGN §13 scope).",
    },
    "headline": "M0 FP32 49.06 ms / 20.4 FPS -> M2 INT8 QAT 21.61 ms / 46.3 FPS = 2.27x on the "
                "real Raspberry Pi CM5, 3.5x smaller (8.95 -> 2.54 MB), Pi<->gx10 parity "
                "bit-identical, accuracy-equivalent (M2 better in-distribution).",
}

out = OUT / "canonical_results.json"
out.write_text(json.dumps(canonical, indent=2, sort_keys=True) + "\n")
print(f"wrote {out}")
print("\n== seed variability (mean bobcat F2 @0.5) ==")
for m, b in seeds.items():
    print(f"  {m:14s} {b['per_seed']}  mean={b['mean']} std={b['std']}")
print("\n== ladder (validation) ==")
for m, d in ladder.items():
    print(f"  {m:3s} {d['kind']:16s} score={d['selection_score_seed42']:.4f} "
          f"{d['mb']:.2f}MB MACs={d['macs']/1e6:.0f}M {d['status']}")
print(f"\n== pi speedups: {canonical['pi_speedups']} ==")
print(f"== parity: {canonical['pi_vs_gx10_parity']['verdict']} ==")
