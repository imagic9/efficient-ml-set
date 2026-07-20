#!/usr/bin/env python3
"""E7/F1 fail-closed host preflight gate (issue #77).

Reads the raw scenario outputs `scripts/run_e7_preflight.sh` produced (one `.rc`,
`.out`, `.err` per scenario) and asserts the preflight's contract:

  - the success path passes and, on a dev host (gx10), records `is_pi5_a76=0` — proof
    the gate accepts an ISA-compatible host for the E8 dry run while still flagging that
    it is not a literal Pi 5 (never confused with a Pi result, DESIGN §12.4);
  - a simulated Pi 5 (A76 cpuinfo) passes AND records `is_pi5_a76=1`;
  - a Pi 4 (A72 cpuinfo, no `asimddp`), a wrong-OS host (not Ubuntu 24.04), and a
    non-aarch64 host each REFUSE with a non-zero exit and an actionable, on-topic reason
    — before any change.

This proves issue #77's Definition of Done without a physical Pi 4 or wrong-OS host.

Usage (driven by scripts/run_e7_preflight.sh):
    python -m wildlife_trigger.validate.preflight_check --raw-dir R --output O
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..runs import atomic_write_json


def read(raw: Path, name: str, ext: str) -> str:
    p = raw / f"{name}.{ext}"
    return p.read_text() if p.exists() else ""


def rc(raw: Path, name: str) -> int:
    text = read(raw, name, "rc").strip()
    return int(text) if text else -1


def fact(blob: str, key: str) -> str | None:
    for line in blob.splitlines():
        if line.startswith(key + "="):
            return line.split("=", 1)[1].strip().strip("'")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw = args.raw_dir

    r1_pi5 = fact(read(raw, "R1", "out"), "WT_IS_PI5_A76")
    r1_asimddp = fact(read(raw, "R1", "out"), "WT_HAS_ASIMDDP")
    r2_pi5 = fact(read(raw, "R2", "out"), "WT_IS_PI5_A76")
    r3_err = read(raw, "R3", "err")
    r4_err = read(raw, "R4", "err")
    r5_err = read(raw, "R5", "err")

    checks = []

    def chk(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    chk("R1 success on the real host", rc(raw, "R1") == 0,
        f"exit {rc(raw, 'R1')}, asimddp={r1_asimddp}")
    chk("R1 records a dev host (is_pi5_a76=0)", r1_pi5 == "0", f"is_pi5_a76={r1_pi5}")
    chk("R2 simulated Pi 5 passes and is flagged (is_pi5_a76=1)",
        rc(raw, "R2") == 0 and r2_pi5 == "1", f"exit {rc(raw, 'R2')}, is_pi5_a76={r2_pi5}")
    chk("R3 Pi 4 (no asimddp) refused with reason",
        rc(raw, "R3") != 0 and "asimddp" in r3_err, f"exit {rc(raw, 'R3')}")
    chk("R4 wrong-OS refused with reason",
        rc(raw, "R4") != 0 and "Ubuntu 24.04" in r4_err, f"exit {rc(raw, 'R4')}")
    chk("R5 non-aarch64 refused with reason",
        rc(raw, "R5") != 0 and "aarch64" in r5_err, f"exit {rc(raw, 'R5')}")

    passed = all(c["passed"] for c in checks)
    report = {
        "gate": "E7/F1 fail-closed host preflight (issue #77)",
        "exit_codes": {n: rc(raw, n) for n in ("R1", "R2", "R3", "R4", "R5")},
        "is_pi5_a76": {"R1_real_host": r1_pi5, "R2_pi5_sim": r2_pi5},
        "checks": checks,
        "note": "The gate accepts any asimddp-capable aarch64 Ubuntu 24.04 host (so the "
                "E8 dry run runs on gx10) and records whether it is a literal Pi 5; it "
                "refuses Pi 4 / wrong-OS / non-aarch64 before any mutation.",
        "verdict": {"passed": passed, "failures": [c["check"] for c in checks if not c["passed"]]},
    }
    atomic_write_json(args.output, report)

    for c in checks:
        print(f"    {'PASS' if c['passed'] else 'FAIL'}  {c['check']}  ({c['detail']})")
    print(f"E7/F1 preflight {'PASSED' if passed else 'FAILED'}; wrote {args.output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
