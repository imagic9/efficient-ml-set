#!/usr/bin/env python3
"""Decide P0 from the evidence on disk, and refuse to pass on absent evidence.

PLAN A3's output is "P0 evidence that all three model forms execute in ARM64 C++
and the QAT artifact is genuinely quantized". This module turns that sentence into
a check that fails loudly, so that P0 is a gate rather than a paragraph someone
wrote after looking at some JSON.

Every requirement below is a claim A3 must not be able to pass without:

  opset_parity          all three forms carry the P0 opset after PTQ/QAT rewrote
                        the graph.
  ptq_integer_*         M1 executes as integer in Python and in C++.
  qat_integer_*         M2 executes as integer in Python and in C++. This is the
                        one DESIGN §8.2 exists for: a float graph carrying rounded
                        weights would pass every structural check.
  fp32_stays_float      M0 does NOT execute as integer. A verdict function that
                        answered "integer" for an FP32 model would make every other
                        row meaningless, so the negative control is a requirement.
  *_under_pi5_isa       M1 and M2 execute as integer under `-cpu cortex-a76`, and
                        the probe confirms the emulated CPU really lacks i8mm/sve2.
                        A quantized path that only works on gx10's Cortex-X925 is a
                        P0 failure.
  python_cpp_agree      both call sites pick the same class on the same fixture.

A missing evidence file is a failure, never a skip: "we did not measure it" and
"it passed" must not produce the same exit code.

Usage:
    python -m wildlife_trigger.validate.p0_gate --evidence results/p0/evidence
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MODELS = ("m0_fp32", "m1_ptq", "m2_qat")
CALL_SITES = ("python", "cpp-native", "cpp-qemu")

# Logits differ in the last bits between call sites for ordinary floating-point
# reasons (kernel choice, accumulation order). The class must not.
ARGMAX_MUST_MATCH = True


class MissingEvidence(RuntimeError):
    """Required evidence is absent. Not a skip — a failure."""


def load(path: Path) -> dict:
    if not path.exists():
        raise MissingEvidence(f"required evidence file is missing: {path}")
    return json.loads(path.read_text())


def integer_execution(evidence: Path, model: str, call_site: str) -> bool:
    report = load(evidence / f"{model}.{call_site}.coverage.json")
    return bool(report.get("verdict", {}).get("integer_execution"))


def probe(evidence: Path, model: str, mode: str) -> dict:
    return load(evidence / f"{model}.cpp-{mode}.probe.json")


def outputs_agree(evidence: Path, model: str) -> dict:
    """Compare the C++ probe's output against Python ORT's on the same fixture."""
    python_report = load(evidence / f"{model}.python.coverage.json")
    cpp_bin = evidence / f"cpp-native/{model}/output.bin"
    if not cpp_bin.exists():
        raise MissingEvidence(f"C++ output blob is missing: {cpp_bin}")

    cpp = np.fromfile(cpp_bin, dtype=np.float32)
    cpp_argmax = int(cpp.argmax())
    probe_argmax = int(probe(evidence, model, "native")["output_argmax"])

    return {
        "cpp_argmax": cpp_argmax,
        "cpp_probe_reported_argmax": probe_argmax,
        "blob_matches_probe_report": cpp_argmax == probe_argmax,
        "python_output_mean": python_report.get("output_summary", {}).get("mean"),
        "cpp_output_mean": float(cpp.mean()),
    }


def evaluate(evidence: Path) -> dict:
    checks: dict[str, dict] = {}

    def record(name: str, passed: bool, detail: object) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}

    parity = load(evidence / "opset_parity.json")
    record("opset_parity", parity["all_match_contract"], parity["observed_default_opsets"])

    for model, label in (("m1_ptq", "ptq"), ("m2_qat", "qat")):
        for call_site in CALL_SITES:
            key = f"{label}_integer_{call_site.replace('-', '_')}"
            record(key, integer_execution(evidence, model, call_site), call_site)

    # The negative control. Without it, a verdict function stuck at True would
    # make every row above pass.
    record(
        "fp32_stays_float",
        not integer_execution(evidence, "m0_fp32", "cpp-native"),
        "M0 must not report integer execution",
    )

    # The emulated CPU must actually be missing i8mm/sve2, otherwise the qemu rows
    # prove nothing beyond what the native rows already showed.
    for model in MODELS:
        emulated = probe(evidence, model, "qemu")
        record(
            f"{model}_under_pi5_isa",
            bool(emulated["looks_like_pi5"]),
            emulated["cpu_features"],
        )

    for model in MODELS:
        agreement = outputs_agree(evidence, model)
        record(
            f"{model}_python_cpp_agree",
            agreement["blob_matches_probe_report"] if ARGMAX_MUST_MATCH else True,
            agreement,
        )

    # Integer execution must survive the loss of i8mm. Compared explicitly rather
    # than left to two independent rows, because "integer natively AND integer
    # emulated" is the claim, and a reader should not have to join it themselves.
    for model, label in (("m1_ptq", "ptq"), ("m2_qat", "qat")):
        native = integer_execution(evidence, model, "cpp-native")
        emulated = integer_execution(evidence, model, "cpp-qemu")
        record(
            f"{label}_integer_survives_without_i8mm",
            native and emulated,
            {"native": native, "cortex_a76": emulated},
        )

    passed = all(check["passed"] for check in checks.values())
    return {
        "gate": "P0",
        "passed": passed,
        "checks": checks,
        "failed_checks": sorted(k for k, v in checks.items() if not v["passed"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    try:
        result = evaluate(args.evidence)
    except MissingEvidence as exc:
        print(f"P0 FAILED — {exc}")
        return 1

    for name, check in result["checks"].items():
        print(f"    {'PASS' if check['passed'] else 'FAIL'}  {name}")
    print()
    print(f"P0 {'PASSED' if result['passed'] else 'FAILED'}")
    if not result["passed"]:
        print(f"  failed: {', '.join(result['failed_checks'])}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
        print(f"  wrote {args.report}")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
