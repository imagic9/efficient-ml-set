#!/usr/bin/env python3
"""Gate E — full ARM64 dry run of the deployment bundle (PLAN E8).

E8 runs the EXACT commands a Pi operator will run — `install.sh`, `run_benchmark.sh`,
`run_demo.sh` — unattended, in a clean target-compatible container, and this parses the
machine-readable outputs the way the reporting code will on the real Pi. It is the last
gate before the rental: if the one-command benchmark does not produce a well-formed
matrix here, it will not on Day 1 either.

Checks:
  - the run completed unattended (the driver's exit code, passed in);
  - `benchmark_matrix.json` exists and names the **baseline** M0 — an optimized model is
    never reported without the thing it is measured against;
  - every model in the matrix has a `benchmark_<MODEL>.json` with `schema_version:1`,
    ordered percentiles, and `performance_targets.measured_on_pi == false` (this is a
    container on gx10, never a Pi verdict);
  - the demo produced predictions.

Gate E passes iff all hold. gx10 dry-run latency is diagnostic only (DESIGN §12.4);
Phase F takes the measurement that counts.

Usage (gx10, driven by scripts/run_e8_dry_run.sh).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..runs import atomic_write_json


def check_benchmark(path: Path) -> tuple[bool, dict]:
    problems: list[str] = []
    if not path.exists():
        return False, {"path": str(path), "error": "missing"}
    d = json.loads(path.read_text())
    if d.get("schema_version") != 1:
        problems.append("schema_version != 1")
    stages = d.get("stages_ms", {})
    for stage, v in stages.items():
        if not (v["min"] <= v["p50"] <= v["p95"] <= v["p99"] <= v["max"]):
            problems.append(f"{stage}: percentiles not ordered")
    pt = d.get("performance_targets", {})
    if pt.get("measured_on_pi") is not False:
        problems.append("measured_on_pi must be false (this is not a Pi)")
    summary = {
        "path": str(path),
        "end_to_end_p50_ms": stages.get("end_to_end", {}).get("p50"),
        "end_to_end_p95_ms": stages.get("end_to_end", {}).get("p95"),
        "measured_iterations": d.get("measured_iterations"),
        "problems": problems,
    }
    return not problems, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--dry-run-rc", required=True, type=int,
                        help="exit code of the unattended install+benchmark+demo run")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    failures: list[str] = []
    if args.dry_run_rc != 0:
        failures.append(f"the unattended dry run exited {args.dry_run_rc}")

    matrix_path = args.bundle / "benchmark_matrix.json"
    models: list[str] = []
    baseline = None
    if not matrix_path.exists():
        failures.append("benchmark_matrix.json missing — one-command benchmark did not run")
    else:
        matrix = json.loads(matrix_path.read_text())
        baseline = matrix.get("baseline")
        models = matrix.get("models", [])
        if baseline != "M0":
            failures.append(f"baseline is {baseline}, expected M0")
        if baseline not in models:
            failures.append("baseline M0 not in the measurement matrix")

    benchmarks = {}
    for m in models:
        ok, summary = check_benchmark(args.bundle / f"benchmark_{m}.json")
        benchmarks[m] = summary
        if not ok:
            failures.append(f"{m} benchmark invalid: {summary.get('problems') or summary.get('error')}")

    demo_preds = args.bundle / "demo_predictions.jsonl"
    demo_ok = demo_preds.exists() and demo_preds.stat().st_size > 0
    if not demo_ok:
        failures.append("demo produced no predictions")

    # The fail-closed preflight (issue #77) ran inside install.sh and must have left a
    # machine-readable environment.json with a passing verdict. On gx10 the host is a
    # dev host (is_pi5_a76=false) — recorded, never a Pi verdict.
    env_path = args.bundle / "environment.json"
    environment = {}
    if not env_path.exists():
        failures.append("install.sh left no environment.json (F1 preflight record)")
    else:
        env = json.loads(env_path.read_text())
        environment = {
            "preflight_passed": env.get("preflight", {}).get("passed"),
            "is_pi5_a76": env.get("cpu", {}).get("is_pi5_a76"),
            "os_codename": env.get("os", {}).get("codename"),
            "opencv_runtime_installed": env.get("opencv_runtime_installed"),
        }
        if environment["preflight_passed"] is not True:
            failures.append("environment.json preflight.passed is not true")

    passed = not failures
    report = {
        "gate": "E — deployment bundle + one-command benchmark work end to end (PLAN E8)",
        "unattended_exit_code": args.dry_run_rc,
        "baseline_in_matrix": baseline == "M0" and baseline in models,
        "models_benchmarked": models,
        "benchmarks": benchmarks,
        "demo_produced_predictions": demo_ok,
        "f1_preflight": environment,
        "provenance": "gx10 dry-run latency is diagnostic only; a latency is a Pi "
                      "result only when measured on a Pi (DESIGN §12.4). Phase F is "
                      "still mandatory.",
        "verdict": {"passed": passed, "failures": failures},
    }
    atomic_write_json(args.output, report)

    print(f"unattended run exit: {args.dry_run_rc}")
    print(f"baseline M0 in matrix: {report['baseline_in_matrix']}")
    for m, s in benchmarks.items():
        if "error" in s:
            print(f"    {m}: {s['error']}")
        else:
            print(f"    {m}: end-to-end p50={s['end_to_end_p50_ms']}ms "
                  f"p95={s['end_to_end_p95_ms']}ms ({s['measured_iterations']} iters)"
                  f"{' PROBLEMS ' + str(s['problems']) if s['problems'] else ''}")
    print(f"demo predictions: {demo_ok}")
    if environment:
        print(f"F1 preflight: passed={environment['preflight_passed']}, "
              f"is_pi5_a76={environment['is_pi5_a76']} (dev host on gx10), "
              f"os={environment['os_codename']}, opencv={environment['opencv_runtime_installed']}")
    for f in failures:
        print(f"    FAIL: {f}")
    print(f"\nGATE E {'PASSED' if passed else 'FAILED'}; wrote {args.output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
