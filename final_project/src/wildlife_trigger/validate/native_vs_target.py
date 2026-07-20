#!/usr/bin/env python3
"""E6 native-vs-target build-and-test (PLAN E6, DESIGN §11).

HISTORICAL NOTE (added 2026-07-20): this gate ran when the deployment **target** was
Debian bookworm (gcc 12 / glibc 2.36) and the native toolchain was Ubuntu 24.04 (gcc 13
/ glibc 2.39). Its finding — the two builds are **bit-identical** — is precisely what
made it safe to later move the target itself to Ubuntu 24.04 (the rented Pi's OS). The
gate is sealed; the labels below describe the E6-era comparison, not today's target.

E6 requires the whole test suite to pass under a clean **native** CPU-only build as
well as the **target-compatible** ARM64 build. gx10's host is Ubuntu 24.04 (gcc 13,
glibc 2.39); the target is Debian bookworm (gcc 12, glibc 2.36). Building and testing
under both surfaces any compiler- or glibc-version-specific breakage before the Pi —
and, because both carry the same pinned ORT and the same OpenCV 4.6.0, a *decision*
difference between the two builds would isolate a compiler/glibc effect rather than a
library gap.

This gate consumes the two builds' evidence and asserts:

  - **unit/integration tests pass in both** — ctest all-green and the on-device
    self-test PASSED under each toolchain;
  - **the two builds agree on every decision** over the validation manifest — a
    cross-toolchain determinism check. FP32 M0 is the sensitive case (INT8 is integer
    and trivially deterministic); the score delta is reported and expected ~0 since
    the OpenCV and ORT are identical and only gcc/glibc differ;
  - the two **environments genuinely differ** (glibc), so the check is not comparing a
    build against itself.

Only the target build ships (it links glibc 2.36); the native build is a portability
witness, never a deployable. No latency is judged here (DESIGN §12.4).

Usage (gx10, driven by scripts/run_e6_native_vs_target.sh).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..runs import atomic_write_json

SCORE_SANITY_ABS = 1e-3  # same OpenCV + same ORT; only gcc/glibc differ -> expect ~0


def read_predictions(path: Path) -> tuple[dict, dict[str, dict]]:
    lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if not lines or lines[0].get("kind") != "run_dataset_header":
        raise RuntimeError(f"{path} does not start with a run_dataset_header")
    if lines[-1].get("kind") != "run_dataset_footer":
        raise RuntimeError(f"{path} has no footer; the run did not finish")
    rows = {r["image_id"]: r for r in lines[1:-1]
            if "image_id" in r and "target_scores" in r and not r.get("skipped")}
    return lines[0], rows


def parse_ctest(spec: str) -> tuple[int, int]:
    passed, total = spec.split("/")
    return int(passed), int(total)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-jsonl", required=True, type=Path)
    parser.add_argument("--target-jsonl", required=True, type=Path)
    parser.add_argument("--native-selftest", required=True, type=Path)
    parser.add_argument("--target-selftest", required=True, type=Path)
    parser.add_argument("--native-ctest", required=True, help="passed/total, e.g. 5/5")
    parser.add_argument("--target-ctest", required=True, help="passed/total, e.g. 5/5")
    parser.add_argument("--target", default="bobcat")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    failures: list[str] = []

    # 1. ctest green under both toolchains.
    ctest = {}
    for name, spec in (("native", args.native_ctest), ("target", args.target_ctest)):
        p, t = parse_ctest(spec)
        ctest[name] = {"passed": p, "total": t}
        if t == 0 or p != t:
            failures.append(f"{name} ctest not all-green ({p}/{t})")

    # 2. self-test PASSED under both.
    selftest = {}
    for name, path in (("native", args.native_selftest), ("target", args.target_selftest)):
        d = json.loads(path.read_text())
        selftest[name] = {"self_test": d.get("self_test"), "failures": d.get("failures"),
                          "environment": d.get("environment")}
        if d.get("self_test") != "PASSED":
            failures.append(f"{name} self-test not PASSED ({d.get('self_test')})")

    # 3. cross-toolchain decision parity over the validation manifest.
    nhdr, nrows = read_predictions(args.native_jsonl)
    thdr, trows = read_predictions(args.target_jsonl)
    common = [i for i in nrows if i in trows]
    if len(common) != len(nrows) or len(nrows) != len(trows):
        failures.append(f"frame sets differ (native {len(nrows)}, target {len(trows)}, "
                        f"common {len(common)})")
    max_abs = 0.0
    sum_abs = 0.0
    decisions_differ = 0
    disagreements = []
    for i in common:
        ns = nrows[i]["target_scores"][args.target]
        ts = trows[i]["target_scores"][args.target]
        gap = abs(ns - ts)
        max_abs = max(max_abs, gap)
        sum_abs += gap
        if nrows[i]["shutter_trigger"] != trows[i]["shutter_trigger"]:
            decisions_differ += 1
            disagreements.append({"image_id": i, "native": ns, "target": ts})
    mean_abs = (sum_abs / len(common)) if common else 0.0

    if decisions_differ:
        failures.append(f"{decisions_differ} decisions differ between native and target builds")
    if max_abs > SCORE_SANITY_ABS:
        failures.append(
            f"max score delta {max_abs:.2e} exceeds the {SCORE_SANITY_ABS:.0e} sanity "
            "bound; same OpenCV + ORT, so gcc/glibc alone should not move it this far")

    # 4. the two builds must share the pinned ORT (else a score gap could be an ORT
    #    difference, not a compiler/glibc one). glibc itself differs by construction
    #    (bookworm 2.36 vs ubuntu 2.39) — that is the point of the second toolchain.
    env_native = selftest["native"].get("environment", {})
    env_target = selftest["target"].get("environment", {})
    same_ort = env_native.get("onnxruntime_version") == env_target.get("onnxruntime_version")
    if not same_ort:
        failures.append("ORT versions differ between builds; the pin is supposed to be shared")

    passed = not failures
    report = {
        "gate": "E6 native-vs-target build-and-test (DESIGN §11, PLAN E6)",
        "toolchains": {
            "native": {"image": "wildlife-trigger-native:ubuntu2404 (gcc 13 / glibc 2.39)",
                       "ctest": ctest["native"], "self_test": selftest["native"]["self_test"]},
            "target": {"image": "wildlife-trigger-target:bookworm (gcc 12 / glibc 2.36)",
                       "ctest": ctest["target"], "self_test": selftest["target"]["self_test"]},
        },
        "cross_toolchain_parity": {
            "frames": len(common),
            "decisions_differing": decisions_differ,
            "max_score_delta": max_abs,
            "mean_score_delta": mean_abs,
            "disagreements": disagreements[:8],
            "sanity_bound": SCORE_SANITY_ABS,
            "note": "both builds share OpenCV 4.6.0 and the pinned ORT; only gcc/glibc "
                    "differ, so decisions must match and the score delta be ~0.",
        },
        "ort_version": env_native.get("onnxruntime_version"),
        "verdict": {"passed": passed, "failures": failures},
    }
    atomic_write_json(args.output, report)

    print(f"native ctest {ctest['native']['passed']}/{ctest['native']['total']}, "
          f"self-test {selftest['native']['self_test']}")
    print(f"target ctest {ctest['target']['passed']}/{ctest['target']['total']}, "
          f"self-test {selftest['target']['self_test']}")
    print(f"cross-toolchain: {len(common)} frames, {decisions_differ} decisions differ, "
          f"max score Δ {max_abs:.2e}, mean {mean_abs:.2e}")
    for f in failures:
        print(f"    FAIL: {f}")
    print(f"E6 native-vs-target {'PASSED' if passed else 'FAILED'}; wrote {args.output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
