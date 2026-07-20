#!/usr/bin/env python3
"""Consolidate the F4 evidence into one machine-readable summary.

Reads the real-Pi benchmark reps (bench_<M>_rep*.json), the frozen full-test metrics
(frozen_test_<M>.json), and the Pi-vs-gx10 parity verdict (parity_comparison.json) under
results/f4/, and emits results/f4/f4_summary.json plus a human table. Derived numbers
(rep means/spreads, speedups) are COMPUTED here, never hand-typed (PLAN §1).

Latencies are real Pi results (Raspberry Pi CM5; DESIGN §12.4). Test accuracy is gx10 C++
evaluation of the frozen artifacts, shown to transfer to the Pi bit-identically by the parity
verdict.

Usage:  python3 scripts/summarize_f4.py [results/f4]
"""
import json
import statistics
import sys
from pathlib import Path

F4 = Path(sys.argv[1] if len(sys.argv) > 1 else "results/f4")
MODELS = ["M0", "M2", "M4"]
KIND = {"M0": "fp32_baseline", "M2": "int8_qat (FINAL)", "M4": "pruned_qat_int8"}


def load(name):
    p = F4 / name
    return json.loads(p.read_text()) if p.exists() else None


# --- Pi latency: 3 separate-process reps per model, frozen threads=1 config ---
pi_latency = {}
for m in MODELS:
    reps = [load(f"bench_{m}_rep{r}.json") for r in (1, 2, 3)]
    reps = [d for d in reps if d]
    if not reps:
        continue
    p50s = [d["stages_ms"]["end_to_end"]["p50"] for d in reps]
    fps = [d["fps"]["end_to_end_from_p50"] for d in reps]
    s = reps[0]["stages_ms"]
    pi_latency[m] = {
        "kind": KIND[m],
        "reps": len(reps),
        "e2e_ms_p50_mean": round(statistics.mean(p50s), 3),
        "e2e_ms_p50_min": round(min(p50s), 3),
        "e2e_ms_p50_max": round(max(p50s), 3),
        "e2e_ms_p50_spread": round(max(p50s) - min(p50s), 3),
        "fps_mean": round(statistics.mean(fps), 2),
        "stages_ms_p50_rep1": {k: round(s[k]["p50"], 3) for k in ("decode", "preprocess", "inference", "policy")},
        "peak_rss_mb": round(reps[0]["system"]["peak_rss_kib"] / 1024, 1),
        "iterations_each": reps[0]["measured_iterations"],
    }

speedups = {}
if "M0" in pi_latency:
    base = pi_latency["M0"]["e2e_ms_p50_mean"]
    for m in ("M2", "M4"):
        if m in pi_latency:
            speedups[f"M0_to_{m}"] = round(base / pi_latency[m]["e2e_ms_p50_mean"], 3)

# --- frozen full-test accuracy (gx10 C++, frozen threshold) ---
frozen_test = {}
for m in ("M0", "M2"):
    d = load(f"frozen_test_{m}.json")
    if d:
        frozen_test[m] = {
            "frozen_threshold": d["frozen_threshold"],
            "per_domain": {
                dom: {k: round(v[k], 4) for k in
                      ("frame_f2", "sequence_balanced_recall", "event_capture_rate",
                       "false_fire_rate", "frame_recall", "frame_precision")
                      if k in v}
                for dom, v in d["per_domain"].items()
            },
            "positives": {dom: v.get("positive_frames") for dom, v in d["per_domain"].items()},
        }

parity = load("parity_comparison.json")

summary = {
    "kind": "f4_summary",
    "schema_version": 1,
    "host_latency": "Raspberry Pi CM5 (BCM2712 / Cortex-A76 @ 2.4 GHz, 8 GB), Ubuntu 24.04, performance governor",
    "host_accuracy": "gx10 C++/ORT of the frozen bundle artifacts (accuracy only; §12.4)",
    "frozen_config": "threads=1, full decode, ORT_ENABLE_ALL, arena on, fused, 256x192",
    "pi_latency": pi_latency,
    "pi_speedups": speedups,
    "frozen_test_accuracy": frozen_test,
    "pi_vs_gx10_parity": parity.get("overall_verdict") if parity else None,
    "parity_detail": parity.get("models") if parity else None,
}
out = F4 / "f4_summary.json"
out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(f"wrote {out}")

print("\n== Pi latency (frozen config, 3 reps, threads=1) ==")
for m in MODELS:
    if m in pi_latency:
        p = pi_latency[m]
        print(f"  {m:3s} {p['kind']:22s} p50={p['e2e_ms_p50_mean']:6.2f}ms "
              f"(±{p['e2e_ms_p50_spread']:.2f}) {p['fps_mean']:6.2f}FPS  RSS={p['peak_rss_mb']}MB")
for k, v in speedups.items():
    print(f"  speedup {k}: {v}x")
if frozen_test:
    print("\n== frozen full-test accuracy (gx10 C++, frozen threshold) ==")
    for m, d in frozen_test.items():
        for dom, v in d["per_domain"].items():
            print(f"  {m} {dom:10s} F2={v.get('frame_f2'):.4f} seqRecall={v.get('sequence_balanced_recall'):.4f} "
                  f"eventCapture={v.get('event_capture_rate'):.4f} falseFire={v.get('false_fire_rate'):.4f}")
if parity:
    print(f"\n== parity: {parity['overall_verdict']} ==")
