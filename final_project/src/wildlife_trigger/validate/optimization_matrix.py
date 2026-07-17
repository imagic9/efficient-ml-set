#!/usr/bin/env python3
"""E6 inference-pipeline optimization matrix — collation (PLAN E6, DESIGN §11/§12).

Reads the per-cell `benchmark` outputs (one JSON per pipeline configuration) and
collates them into one table, each row differing from the baseline in exactly **one
factor** — that single-factor discipline is what makes a latency delta attributable
to the knob rather than to two knobs at once, so the collator *asserts* it.

The baseline is the shipping configuration: fused preprocessing, full decode,
ORT_ENABLE_ALL, CPU arena on, one intra-op thread. Every other cell changes one of:
preprocessing (reference), decode (half / quarter), graph level (extended), threads
(2 / 4), arena (off).

**These are gx10 latencies — diagnostic only, never a Pi result (DESIGN §12.4).**
The collator refuses any cell whose `performance_targets.measured_on_pi` is not
false, so a number from this table can never be mistaken for a Pi verdict. The matrix
tells us which knobs are *worth carrying to the Pi*, where the measurement that counts
is taken; it does not itself decide latency.

Usage (gx10, after scripts/run_e6_optimization_matrix.sh):
    python -m wildlife_trigger.validate.optimization_matrix \\
        --dir results/e6/optimization_matrix \\
        --output results/e6/optimization_matrix.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..runs import atomic_write_json

BASELINE = {
    "preprocess": "fused",
    "decode_reduction": 1,
    "graph_optimization": "all",
    "cpu_arena": "on",
    "intra_op_threads": 1,
}


def load_cell(path: Path) -> dict:
    d = json.loads(path.read_text())
    failures: list[str] = []
    if d.get("schema_version") != 1:
        failures.append("schema_version != 1")
    pt = d.get("performance_targets", {})
    if pt.get("measured_on_pi") is not False:
        failures.append("measured_on_pi must be present and false (this is not a Pi run)")
    for stage, v in d.get("stages_ms", {}).items():
        if not (v["min"] <= v["p50"] <= v["p95"] <= v["p99"] <= v["max"]):
            failures.append(f"{stage}: percentiles not ordered")
    return {"path": path, "doc": d, "failures": failures}


def summarise(d: dict) -> dict:
    s = d["stages_ms"]
    return {
        "config": d["pipeline_config"],
        "measured_iterations": d["measured_iterations"],
        "decode_p50_ms": s["decode"]["p50"],
        "preprocess_p50_ms": s["preprocess"]["p50"],
        "inference_p50_ms": s["inference"]["p50"],
        "end_to_end_p50_ms": s["end_to_end"]["p50"],
        "end_to_end_p95_ms": s["end_to_end"]["p95"],
        "end_to_end_fps": d["fps"]["end_to_end_from_p50"],
    }


def differing_factors(config: dict) -> list[str]:
    return [k for k, v in BASELINE.items() if config.get(k) != v]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    cells = [load_cell(p) for p in sorted(args.dir.glob("bench_*.json"))]
    if not cells:
        raise RuntimeError(f"no bench_*.json cells in {args.dir}")

    failures: list[str] = []
    for c in cells:
        for f in c["failures"]:
            failures.append(f"{c['path'].name}: {f}")

    summaries = {c["path"].stem: summarise(c["doc"]) for c in cells}

    # Exactly one baseline cell; every other cell must differ in exactly one factor.
    baselines = [name for name, s in summaries.items() if not differing_factors(s["config"])]
    if len(baselines) != 1:
        failures.append(f"expected exactly one baseline cell, found {baselines}")
    base_name = baselines[0] if baselines else None
    base = summaries.get(base_name)

    model_shas = {c["doc"].get("model_sha256") for c in cells}
    if len(model_shas) != 1:
        failures.append(f"cells span multiple models {model_shas}; hold the model constant")

    rows = []
    for name, s in summaries.items():
        if name == base_name:
            continue
        factors = differing_factors(s["config"])
        if len(factors) != 1:
            failures.append(
                f"{name}: differs from baseline in {len(factors)} factors {factors}; "
                "the matrix changes one factor at a time"
            )
        factor = factors[0] if len(factors) == 1 else "+".join(factors)
        row = {
            "cell": name,
            "factor": factor,
            "value": s["config"].get(factor) if len(factors) == 1 else s["config"],
            **s,
        }
        if base:
            row["end_to_end_p50_delta_ms"] = s["end_to_end_p50_ms"] - base["end_to_end_p50_ms"]
            row["speedup_vs_baseline"] = (
                base["end_to_end_p50_ms"] / s["end_to_end_p50_ms"]
                if s["end_to_end_p50_ms"] else None
            )
        rows.append(row)

    rows.sort(key=lambda r: (r["factor"], str(r["value"])))
    passed = not failures
    report = {
        "experiment": "E6 inference-pipeline optimization matrix (one factor at a time)",
        "provenance": "gx10 latencies — DIAGNOSTIC ONLY, never a Pi result (DESIGN §12.4). "
                      "The matrix ranks which knobs to carry to the Pi; Phase F measures "
                      "the latency that counts.",
        "model_sha256": next(iter(model_shas)) if len(model_shas) == 1 else None,
        "baseline": {"cell": base_name, **base} if base else None,
        "rows": rows,
        "verdict": {"passed": passed, "failures": failures},
    }
    atomic_write_json(args.output, report)

    if base:
        print(f"baseline ({base_name}): end-to-end p50={base['end_to_end_p50_ms']:.2f}ms "
              f"p95={base['end_to_end_p95_ms']:.2f}ms ({base['end_to_end_fps']:.1f} FPS) "
              f"[decode {base['decode_p50_ms']:.2f} / inf {base['inference_p50_ms']:.2f}]")
    for r in rows:
        sp = r.get("speedup_vs_baseline")
        tail = f" (Δ{r.get('end_to_end_p50_delta_ms', 0):+.2f}ms, {sp:.2f}x)" if sp else ""
        print(f"  {r['factor']}={r['value']}: p50={r['end_to_end_p50_ms']:.2f}ms{tail}")
    for f in failures:
        print(f"  FAIL: {f}")
    print(f"E6 optimization matrix {'OK' if passed else 'FAILED'} (diagnostic); wrote {args.output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
