#!/usr/bin/env python3
"""Decide E1 from the evidence on disk, and refuse to pass on absent evidence.

PLAN E1 hardens the A4 smoke slice into the real C++ application/library **using
M0** — the actual FP32 baseline, not the synthetic smoke network A4 ran on. This
gate turns that sentence into checks, the same way `a4_gate` did for A4 and
`p0_gate` for P0.

Where A4 proved the pipeline runs at all (on a deterministic smoke model whose every
number it declared synthetic), E1 proves the *foundation* is sound against the real
deployable baseline: the model contract holds on M0, the decision is self-consistent,
the Pi's ISA agrees with native under QEMU, the machine-readable schemas are stamped
and ordered, the dataset runner reproduces M0's own operating point at corpus scale,
and the logging convention (DESIGN §11 component 7) behaves under its threshold.

The checks encode what "the foundation works on M0" has to mean:

  m0_identity_matches_freeze   the ONNX under test is the frozen M0, by SHA-256.
  self_test_passed_on_m0       the on-host self-test (contract, softmax, scoring) passes.
  infer_decision_valid         the decision names a known class, the trigger is binary,
                               every target is scored, and it carries schema_version.
  trigger_matches_any          SHUTTER_TRIGGER == any(target passed): the two fields agree.
  letterbox_preserved_frame    no crop; padding accounted; real pixels dominate.
  runs_under_pi5_isa           M0 infer runs under `-cpu cortex-a76`...
  qemu_agrees_with_native      ...and reaches the same decision as native.
  benchmark_schema_valid       schema_version stamped, percentiles ordered p50<=p95<=p99.
  benchmark_not_a_pi_result    the output says so in words; a gx10 number is never a Pi one.
  dataset_runner_wellformed    header (schema_version + M0 hash), one line per frame, footer
                               counts reconcile.
  dataset_reproduces_m0        the C++ decisions match M0's precomputed operating point on
                               every non-threshold-adjacent frame (the boundary stratum is
                               excluded by construction — it exists to flip).
  logging_*                    the leveled convention: debug suppressed by default and
                               shown at WILDLIFE_LOG_LEVEL=debug; an error keeps its
                               "error:" prefix and non-zero exit; error level silences info.

Usage:
    python -m wildlife_trigger.validate.e1_gate \\
        --evidence results/e1/evidence --freeze results/model_selection/pre_pi_freeze.json \\
        --manifest data/manifests/benchmark_val_1000.jsonl --report results/e1/e1_gate.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


class MissingEvidence(RuntimeError):
    """Required evidence is absent. Not a skip — a failure."""


def load_json(path: Path) -> dict:
    if not path.exists():
        raise MissingEvidence(f"required evidence file is missing: {path}")
    return json.loads(path.read_text())


def load_text(path: Path) -> str:
    if not path.exists():
        raise MissingEvidence(f"required evidence file is missing: {path}")
    return path.read_text()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise MissingEvidence(f"required evidence file is missing: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def percentiles_ordered(block: dict) -> bool:
    for values in block.values():
        if not (values["p50"] <= values["p95"] <= values["p99"]):
            return False
        if not (values["min"] <= values["p50"] <= values["max"]):
            return False
    return True


def m0_from_freeze(freeze: dict) -> dict:
    for model in freeze["models"]:
        if model["model_id"] == "M0":
            return model
    raise MissingEvidence("pre_pi_freeze.json has no M0 entry")


def evaluate(evidence: Path, freeze_path: Path, manifest_path: Path) -> dict:
    checks: dict[str, dict] = {}

    def record(name: str, passed: bool, detail: object) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}

    freeze = load_json(freeze_path)
    m0 = m0_from_freeze(freeze)
    m0_sha = m0["onnx"]["sha256"]

    infer = load_json(evidence / "infer.m0.native.json")
    decision = infer["decision"]
    known_classes = {t["class"] for t in decision["targets"]} | {decision["top1"]["class"]}

    # The whole point of E1 is that this ran on the real M0. Verify by hash, not name.
    record(
        "m0_identity_matches_freeze",
        infer["model_sha256"] == m0_sha,
        {"infer": infer["model_sha256"][:16], "freeze": m0_sha[:16]},
    )

    self_test = load_json(evidence / "self_test.m0.json")
    record(
        "self_test_passed_on_m0",
        self_test["self_test"] == "PASSED" and self_test.get("schema_version") == 1,
        self_test["failures"],
    )

    record(
        "infer_decision_valid",
        infer.get("schema_version") == 1
        and decision["top1"]["class"] in known_classes
        and decision["SHUTTER_TRIGGER"] in (0, 1)
        and bool(decision["targets"])
        and all(
            {"class", "index", "score", "threshold", "passed"} <= set(t)
            and 0.0 <= t["score"] <= 1.0
            and 0.0 <= t["threshold"] <= 1.0
            for t in decision["targets"]
        ),
        {"top1": decision["top1"]["class"], "targets": [t["class"] for t in decision["targets"]]},
    )
    record(
        "trigger_matches_any",
        bool(decision["SHUTTER_TRIGGER"]) == any(t["passed"] for t in decision["targets"]),
        {"trigger": decision["SHUTTER_TRIGGER"], "passing": decision["passing_targets"]},
    )

    letterbox = infer["letterbox"]
    record(
        "letterbox_preserved_frame",
        letterbox["resized"][0] <= 256
        and letterbox["resized"][1] <= 192
        and letterbox["pixel_utilisation"] > 0.5,
        letterbox,
    )

    infer_qemu = load_json(evidence / "infer.m0.qemu.json")
    record(
        "runs_under_pi5_isa",
        infer_qemu["environment"]["looks_like_pi5"] is True,
        infer_qemu["environment"]["cpu_features"],
    )
    record(
        "qemu_agrees_with_native",
        infer_qemu["model_sha256"] == m0_sha
        and infer_qemu["decision"]["top1"]["class"] == decision["top1"]["class"]
        and infer_qemu["decision"]["SHUTTER_TRIGGER"] == decision["SHUTTER_TRIGGER"],
        {
            "native_top1": decision["top1"]["class"],
            "cortex_a76_top1": infer_qemu["decision"]["top1"]["class"],
            "native_trigger": decision["SHUTTER_TRIGGER"],
            "cortex_a76_trigger": infer_qemu["decision"]["SHUTTER_TRIGGER"],
        },
    )

    benchmark = load_json(evidence / "benchmark.m0.json")
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
        "benchmark_not_a_pi_result",
        "Pi" in benchmark.get("provenance", ""),
        "a gx10 latency is a timing-path smoke check, never a Pi result (DESIGN §12.4)",
    )

    # --- the dataset runner over a real slice of benchmark_val_1000, on M0 ---
    rows = read_jsonl(evidence / "run_dataset.m0.jsonl")
    header = next((r for r in rows if r.get("kind") == "run_dataset_header"), None)
    footer = next((r for r in rows if r.get("kind") == "run_dataset_footer"), None)
    predictions = [r for r in rows if "image_id" in r and "target_scores" in r]
    if header is None or footer is None:
        raise MissingEvidence("run_dataset.m0.jsonl missing header or footer line")

    record(
        "dataset_runner_wellformed",
        header.get("schema_version") == 1
        and header.get("model_sha256") == m0_sha
        and footer["processed"] == len(predictions)
        and footer["skipped"] == 0
        and all(
            "bobcat" in p["target_scores"]
            and 0.0 <= p["target_scores"]["bobcat"] <= 1.0
            and p["shutter_trigger"] in (0, 1)
            for p in predictions
        ),
        {"processed": footer["processed"], "fired": footer["fired"], "skipped": footer["skipped"]},
    )

    # Cross-check the C++ decisions against M0's own precomputed operating point
    # (m0_bobcat_score in the manifest). The threshold-adjacent stratum is excluded
    # from the strict requirement: it was over-sampled precisely because those frames
    # sit within eps=0.1 of the boundary and a legitimate numeric delta can flip them
    # (the reason benchmark_val_1000 exists).
    #
    # The gate is DECISION agreement, not tight probability parity. Tight numeric
    # parity is P1/P2/P4's job and already passed for M0; here the manifest scores
    # came from M0 through Python preprocessing (OpenCV 4.13) while the C++ path uses
    # the Pi's OpenCV 4.6, so a small preprocessing delta (P1 budget: 0.035 at the
    # pixel level) is expected. That delta cannot flip a non-threshold-adjacent frame,
    # which is >=0.1 from the boundary by construction. max_score_delta is reported as
    # a diagnostic — a grossly broken foundation would move it toward 1.0 and collapse
    # agreement at the same time, so agreement alone is the sufficient gate.
    threshold = next(t["threshold"] for t in decision["targets"] if t["class"] == "bobcat")
    manifest = {r["image_id"]: r for r in read_jsonl(manifest_path)}
    strict_total = 0
    strict_agree = 0
    max_score_delta = 0.0
    disagreements: list[dict] = []
    for pred in predictions:
        ref = manifest.get(pred["image_id"])
        if ref is None:
            continue
        cpp_score = pred["target_scores"]["bobcat"]
        ref_score = ref["m0_bobcat_score"]
        max_score_delta = max(max_score_delta, abs(cpp_score - ref_score))
        cpp_fire = bool(pred["shutter_trigger"])
        ref_fire = ref_score >= threshold
        if ref.get("benchmark_stratum") == "threshold_adjacent":
            continue
        strict_total += 1
        if cpp_fire == ref_fire:
            strict_agree += 1
        else:
            disagreements.append(
                {
                    "image_id": pred["image_id"],
                    "stratum": ref.get("benchmark_stratum"),
                    "cpp_score": cpp_score,
                    "m0_score": ref_score,
                }
            )
    agreement = (strict_agree / strict_total) if strict_total else 0.0
    record(
        "dataset_reproduces_m0",
        strict_total > 0 and agreement == 1.0,
        {
            "strict_frames": strict_total,
            "agreement": round(agreement, 4),
            "max_score_delta_diagnostic": max_score_delta,
            "threshold": threshold,
            "disagreements": disagreements[:5],
        },
    )

    # --- the logging convention (DESIGN §11 component 7) ---
    default_err = load_text(evidence / "log.default.stderr")
    debug_err = load_text(evidence / "log.debug.stderr")
    error_err = load_text(evidence / "log.error.stderr")
    badpolicy_err = load_text(evidence / "log.badpolicy.stderr")
    badpolicy_exit = int(load_text(evidence / "log.badpolicy.exit").strip())

    record(
        "logging_debug_suppressed_by_default",
        "debug:" not in default_err and "SHUTTER_TRIGGER=" in default_err,
        {"has_debug": "debug:" in default_err, "has_summary": "SHUTTER_TRIGGER=" in default_err},
    )
    record(
        "logging_debug_shown_when_enabled",
        "debug:" in debug_err and "model contract:" in debug_err,
        {"debug_lines": [ln for ln in debug_err.splitlines() if ln.startswith("debug:")][:3]},
    )
    record(
        "logging_error_prefixed_and_nonzero_exit",
        "error:" in badpolicy_err and badpolicy_exit != 0,
        {"exit": badpolicy_exit, "first_line": badpolicy_err.strip().splitlines()[:1]},
    )
    record(
        "logging_error_level_silences_info",
        "SHUTTER_TRIGGER=" not in error_err,
        {"stderr_len": len(error_err.strip())},
    )

    passed = all(check["passed"] for check in checks.values())
    return {
        "gate": "E1",
        "passed": passed,
        "model": "M0",
        "m0_sha256": m0_sha,
        "checks": checks,
        "failed_checks": sorted(k for k, v in checks.items() if not v["passed"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    try:
        result = evaluate(args.evidence, args.freeze, args.manifest)
    except MissingEvidence as exc:
        print(f"E1 FAILED — {exc}")
        return 1

    for name, check in result["checks"].items():
        print(f"    {'PASS' if check['passed'] else 'FAIL'}  {name}")
    print()
    print(f"E1 {'PASSED' if result['passed'] else 'FAILED'}")
    if not result["passed"]:
        print(f"  failed: {', '.join(result['failed_checks'])}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
        print(f"  wrote {args.report}")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
