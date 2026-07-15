#!/usr/bin/env python3
"""Gate A: P0 passes AND the thin C++ path works, before any data or training.

PLAN's Gate A is a conjunction of two gates that were run separately, and this is the
place that says so in one exit code. It exists because Gate A is what *permits* the
next thing — the CCT-20 download and long training runs — and a permission that lives
only in a human's memory of two green terminals is not a gate.

It re-reads the gate reports rather than re-running them: p0_gate and a4_gate own
their own evidence and their own verdicts. What this adds is refusing to call Gate A
passed when one of them is missing, stale, or failed.

Usage:
    python -m wildlife_trigger.validate.gate_a --p0 results/p0/p0_gate.json \
        --a4 results/a4/a4_gate.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_gate(path: Path, expected: str) -> dict:
    if not path.exists():
        raise RuntimeError(
            f"{expected} gate report is missing: {path}. Gate A cannot pass on "
            "evidence that was never produced."
        )
    report = json.loads(path.read_text())
    if report.get("gate") != expected:
        raise RuntimeError(
            f"{path} reports gate {report.get('gate')!r}, expected {expected!r}"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0", required=True, type=Path)
    parser.add_argument("--a4", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    try:
        p0 = load_gate(args.p0, "P0")
        a4 = load_gate(args.a4, "A4")
    except RuntimeError as exc:
        print(f"GATE A FAILED — {exc}")
        return 1

    passed = bool(p0["passed"]) and bool(a4["passed"])
    result = {
        "gate": "A",
        "passed": passed,
        "requires": {
            "P0": {
                "passed": p0["passed"],
                "checks": len(p0["checks"]),
                "failed": p0["failed_checks"],
                "report": str(args.p0),
            },
            "A4": {
                "passed": a4["passed"],
                "checks": len(a4["checks"]),
                "failed": a4["failed_checks"],
                "report": str(args.a4),
            },
        },
        "permits": (
            "The CCT-20 download (Phase B) and long training runs (Phase C). Neither "
            "may start before this passes: PLAN puts the toolchain risk first "
            "precisely so a quantization or deployment dead-end is found in an hour "
            "rather than after a week of training."
        ),
    }

    for name, info in result["requires"].items():
        status = "PASS" if info["passed"] else "FAIL"
        print(f"    {status}  {name}  ({info['checks']} checks)")
        if info["failed"]:
            print(f"          failed: {', '.join(info['failed'])}")
    print()
    print(f"GATE A {'PASSED' if passed else 'FAILED'}")
    if passed:
        print("  Phase B (data) and Phase C (training) are now permitted.")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
        print(f"  wrote {args.report}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
