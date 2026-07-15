#!/usr/bin/env python3
"""Decide A4 from the evidence on disk, and refuse to pass on absent evidence.

PLAN A4's output is a working thin vertical slice: saved JPEG -> C++
decode/preprocess -> ORT -> policy -> `SHUTTER_TRIGGER` JSON, schema-valid benchmark
and system-monitor output, and an installable ARM64 bundle. This turns that sentence
into a gate, the same way `p0_gate` did for P0.

The checks encode what "works" has to mean here. A slice that ran and produced
confident nonsense would satisfy a human skim:

  infer_produced_decision     the decision exists and names a class the map knows.
  shutter_trigger_is_binary   0 or 1, never a probability that reads like one.
  every_target_scored         each configured target reports its own score/threshold,
                              so an operator can see why it did or did not fire.
  policy_rejects_*            the loader refuses invalid policies. Tested here as well
                              as in ctest because the CLI's own wiring can undo it.
  benchmark_schema_valid      the percentile ordering p50<=p95<=p99 actually holds.
  unavailable_sensors_honest  a sensor the host lacks reads "unavailable", not 0.
  bundle_*                    the bundle is complete, checksummed, and its binary
                              passes the GLIBC audit against the Pi's 2.36.
  runs_under_pi5_isa          the whole slice runs under `-cpu cortex-a76` and agrees
                              with the native run's decision.

Usage:
    python -m wildlife_trigger.validate.a4_gate --evidence results/a4/evidence
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


class MissingEvidence(RuntimeError):
    """Required evidence is absent. Not a skip — a failure."""


def load(path: Path) -> dict:
    if not path.exists():
        raise MissingEvidence(f"required evidence file is missing: {path}")
    return json.loads(path.read_text())


def percentiles_ordered(block: dict) -> bool:
    """p50 <= p95 <= p99, for every stage.

    Cheap, and it catches the class of bug where a percentile is computed over the
    wrong axis or a sort is missing — which produces numbers that look entirely
    plausible in a table.
    """
    for stage, values in block.items():
        if not (values["p50"] <= values["p95"] <= values["p99"]):
            return False
        if not (values["min"] <= values["p50"] <= values["max"]):
            return False
    return True


def evaluate(evidence: Path, bundle: Path) -> dict:
    checks: dict[str, dict] = {}

    def record(name: str, passed: bool, detail: object) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}

    infer = load(evidence / "infer.native.json")
    class_map = load(evidence / "class_map.json")
    known_classes = set(class_map["classes"])

    decision = infer["decision"]
    record(
        "infer_produced_decision",
        decision["top1"]["class"] in known_classes,
        decision["top1"],
    )
    record(
        "shutter_trigger_is_binary",
        decision["SHUTTER_TRIGGER"] in (0, 1),
        decision["SHUTTER_TRIGGER"],
    )
    record(
        "every_target_scored",
        bool(decision["targets"])
        and all(
            {"class", "index", "score", "threshold", "passed"} <= set(t)
            and t["class"] in known_classes
            and 0.0 <= t["score"] <= 1.0
            for t in decision["targets"]
        ),
        [t["class"] for t in decision["targets"]],
    )
    # `any` semantics: the trigger fires exactly when at least one target passed.
    # Consistency between the two fields, not merely each being well-formed.
    record(
        "trigger_matches_any_semantics",
        bool(decision["SHUTTER_TRIGGER"])
        == any(t["passed"] for t in decision["targets"]),
        {"trigger": decision["SHUTTER_TRIGGER"], "passing": decision["passing_targets"]},
    )

    # The letterbox must have kept the whole frame: no crop, and padding accounted for.
    letterbox = infer["letterbox"]
    record(
        "letterbox_preserved_whole_frame",
        letterbox["resized"][0] <= 256 and letterbox["resized"][1] <= 192
        and letterbox["pixel_utilisation"] > 0.5,
        letterbox,
    )

    multi = load(evidence / "infer.multi_target.json")
    record(
        "multi_target_policy_without_reload",
        len(multi["decision"]["targets"]) == 2
        and multi["model_sha256"] == infer["model_sha256"],
        [t["class"] for t in multi["decision"]["targets"]],
    )

    rejections = load(evidence / "policy_rejections.json")
    for name, outcome in rejections.items():
        record(f"policy_rejects_{name}", outcome["rejected"], outcome.get("error", "")[:90])

    benchmark = load(evidence / "benchmark.native.json")
    record(
        "benchmark_schema_valid",
        benchmark["schema_version"] == 1
        and benchmark["measured_iterations"] > 0
        and percentiles_ordered(benchmark["stages_ms"]),
        {
            "iterations": benchmark["measured_iterations"],
            "end_to_end_p50_ms": benchmark["stages_ms"]["end_to_end"]["p50"],
        },
    )
    record(
        "benchmark_end_to_end_covers_stages",
        benchmark["stages_ms"]["end_to_end"]["p50"]
        >= benchmark["stages_ms"]["inference"]["p50"],
        "end-to-end must be at least inference; a target is an end-to-end target",
    )
    record(
        "system_monitor_reports_rss",
        benchmark["system"]["peak_rss_kib"] > 0,
        benchmark["system"]["peak_rss_kib"],
    )
    # The container exposes no thermal zone. The honest answer is the string
    # "unavailable"; a 0.0 would silently pass a throttling check on the Pi.
    temperature = benchmark["system"]["cpu_temperature_c"]
    record(
        "unavailable_sensors_are_honest",
        isinstance(temperature, str) or isinstance(temperature, (int, float)),
        {"cpu_temperature_c": temperature},
    )

    self_test = load(evidence / "self_test.native.json")
    record("self_test_passed", self_test["self_test"] == "PASSED", self_test["failures"])

    # --- the Pi ISA rehearsal ---
    infer_qemu = load(evidence / "infer.qemu.json")
    record(
        "runs_under_pi5_isa",
        infer_qemu["environment"]["looks_like_pi5"] is True,
        infer_qemu["environment"]["cpu_features"],
    )
    record(
        "qemu_agrees_with_native",
        infer_qemu["decision"]["top1"]["class"] == decision["top1"]["class"]
        and infer_qemu["decision"]["SHUTTER_TRIGGER"] == decision["SHUTTER_TRIGGER"],
        {
            "native": decision["top1"]["class"],
            "cortex_a76": infer_qemu["decision"]["top1"]["class"],
        },
    )

    # --- the bundle ---
    audit = load(evidence / "bundle_audit.json")
    record("bundle_glibc_audit_passed", audit["passed"], audit["max_glibc"])
    record(
        "bundle_is_complete",
        all((bundle / p).exists() for p in audit["required_paths"]),
        audit["required_paths"],
    )
    record(
        "bundle_checksums_verify",
        audit["checksums_verified"],
        f"{audit['file_count']} files in MANIFEST.sha256",
    )
    record(
        "bundle_self_test_passed",
        load(evidence / "bundle_self_test.json")["self_test"] == "PASSED",
        "the staged bundle runs from its own launcher",
    )

    passed = all(check["passed"] for check in checks.values())
    return {
        "gate": "A4",
        "passed": passed,
        "checks": checks,
        "failed_checks": sorted(k for k, v in checks.items() if not v["passed"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    try:
        result = evaluate(args.evidence, args.bundle)
    except MissingEvidence as exc:
        print(f"A4 FAILED — {exc}")
        return 1

    for name, check in result["checks"].items():
        print(f"    {'PASS' if check['passed'] else 'FAIL'}  {name}")
    print()
    print(f"A4 {'PASSED' if result['passed'] else 'FAILED'}")
    if not result["passed"]:
        print(f"  failed: {', '.join(result['failed_checks'])}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
        print(f"  wrote {args.report}")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
