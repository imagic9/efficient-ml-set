#!/usr/bin/env python3
"""Gate P3 — quantized-model validation (DESIGN §10), for one candidate.

Four checks, each of which catches a different way a quantized artifact lies:

1. **Graph and coverage** — a fresh `ort_coverage` pass on the artifact as it
   exists now: ONNX checker validity, and the integer-execution verdict from
   the session-optimized graph plus the runtime profile. A QDQ file that runs
   float convolutions fails here, whatever its filename says.
2. **Metrics reproduce** — the full validation evaluation is re-run through
   `optimize.evaluate_onnx` and must equal the recorded candidate numbers
   exactly (same runtime, same data, deterministic arithmetic — equality, not
   tolerance). A mismatch means the recorded evidence describes different
   bytes, a moved dataset, or a non-deterministic path; all three block.
3. **ORT Python vs ORT C++ on fixtures** — the same two layers C4 registered
   (logits on P1's canonical tensors via `ort_probe`; decisions through the
   real `infer` CLI), same gates: max abs <= 1e-4, identical argmax, identical
   SHUTTER_TRIGGER under the 1e-4 threshold carve-out.
4. **Binding** — every C++ infer record must name this candidate's model hash
   and this policy id: "no silent fallback to an unintended model or
   preprocessing path" is checked against what the CLI actually loaded, not
   against what the invocation intended.

On a passing verdict the report attaches itself to the policy
(`model.parity`), which is what lets `comparison --candidate` admit the row —
the same one-way door `rebind_policy` implemented for M0.

Usage (gx10, after scripts/run_d1_p3p4.sh has produced the C++ halves):
    python -m wildlife_trigger.validate.p3_quantized \
        --candidate results/optimize/m1_ptq/<method> \
        --policy artifacts/policies/bobcat_m1_ptq_<method>_v1.json \
        --p1-dir results/parity/c2_m0_fp32_seed42_20260716T061203Z/p1 \
        --cpp-logits-dir results/optimize/m1_ptq/<method>/p3/logits \
        --infer-dir results/optimize/m1_ptq/<method>/p3 \
        --output results/optimize/m1_ptq/<method>/p3_quantized.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
import onnxruntime as ort

from ..data.preprocess import PreprocessConfig
from ..policy import validate_policy, write_canonical_json
from ..runs import atomic_write_json, sha256_file
from . import ort_coverage
from .ort_cpp_parity import compare_decision_layer, compare_logits_layer

# The registered numeric gates, shared with P2/C4 (DESIGN §10).
from .ort_cpp_parity import DECISION_CARVE_OUT, LOGITS_MAX_ABS  # noqa: F401


def check_graph_and_coverage(model_path: Path, workdir: Path, label: str) -> dict:
    onnx.checker.check_model(str(model_path), full_check=True)
    report = ort_coverage.analyse(model_path, workdir, label)
    verdict = report["verdict"]
    return {
        "onnx_checker": "passed",
        "integer_execution": verdict["integer_execution"],
        "float_compute_executed": verdict["float_compute_executed"],
        "integer_kernels_executed": sorted(verdict["integer_kernels_executed"]),
        "coverage_report": report,
        "passed": verdict["integer_execution"],
    }


def check_graph_fp32_pruned(
    model_path: Path, workdir: Path, label: str, candidate: dict
) -> dict:
    """Check 1 for a pruned FP32 candidate (m3_registration.md §5).

    FP32 has no integer-coverage question — the physical one replaces it: the
    artifact's conv-shape multiset must equal the candidate's recorded pruned
    shapes, so the deployable file provably carries the cuts. And the inverse
    of the quantized gate holds: an FP32 artifact that executed an integer
    kernel somewhere would be a different candidate than the evidence
    describes.
    """
    onnx.checker.check_model(str(model_path), full_check=True)
    report = ort_coverage.analyse(model_path, workdir, label)
    verdict = report["verdict"]

    graph = onnx.load(str(model_path)).graph
    exported_shapes = sorted(
        list(initializer.dims)
        for initializer in graph.initializer
        if len(initializer.dims) == 4
    )
    recorded_shapes = sorted(
        list(shape) for shape in candidate["pruning"]["exported_conv_shapes"]
    )
    shapes_match = exported_shapes == recorded_shapes

    failures = []
    if not shapes_match:
        failures.append("exported conv-shape multiset differs from the candidate record")
    if verdict["integer_execution"]:
        failures.append("an FP32 candidate executed integer kernels")

    return {
        "onnx_checker": "passed",
        "pruned_shapes_match": shapes_match,
        "conv_weights": len(exported_shapes),
        "integer_execution": verdict["integer_execution"],
        "coverage_report": report,
        "failures": failures,
        "passed": not failures,
    }


def check_metrics_reproduce(candidate_dir: Path, workdir: Path) -> dict:
    """Re-run the evaluation; the recorded candidate numbers must be exact."""
    from ..optimize.evaluate_onnx import evaluate

    recorded = json.loads((candidate_dir / "evaluation.json").read_text())
    fresh = evaluate(
        Path(recorded["model"]["path"]),
        label=f"{recorded['label']}_p3_rerun",
        output_dir=workdir,
        intra_op_threads=recorded["regime"]["intra_op_threads"],
    )

    mismatches = []
    if fresh["selection_score"] != recorded["selection_score"]:
        mismatches.append("selection_score")
    for domain in ("cis_val_clean", "trans_val"):
        if fresh["domains"][domain]["target"] != recorded["domains"][domain]["target"]:
            mismatches.append(f"domains.{domain}.target")
    if fresh["model"]["sha256"] != recorded["model"]["sha256"]:
        mismatches.append("model.sha256")

    return {
        "recorded_primary": recorded["selection_score"]["primary"],
        "rerun_primary": fresh["selection_score"]["primary"],
        "mismatches": mismatches,
        "passed": not mismatches,
    }


def check_binding(infer_dir: Path, model_sha256: str, policy_id: str) -> dict:
    """Every infer record must name the artifact and policy under test."""
    records = sorted(infer_dir.glob("infer_*.json"))
    if not records:
        raise RuntimeError(f"no infer_*.json under {infer_dir}")
    wrong = []
    for path in records:
        record = json.loads(path.read_text())
        if record["model_sha256"] != model_sha256:
            wrong.append(f"{path.name}: model {record['model_sha256'][:12]}…")
        if record["policy_id"] != policy_id:
            wrong.append(f"{path.name}: policy {record['policy_id']}")
    return {"records": len(records), "wrong_bindings": wrong, "passed": not wrong}


def attach_parity(policy_path: Path, class_map_path: Path, report_path: Path) -> None:
    """The one-way door: a passing P3 report becomes part of the policy."""
    policy = json.loads(policy_path.read_text())
    policy["model"]["parity"] = str(report_path)
    validate_policy(
        policy,
        json.loads(class_map_path.read_text()),
        model_sha256=policy["model_sha256"],
        class_map_sha256=sha256_file(class_map_path),
    )
    write_canonical_json(policy_path, policy)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--class-map", type=Path, default=Path("artifacts/class_map.json"))
    parser.add_argument("--p1-dir", required=True, type=Path)
    parser.add_argument("--cpp-logits-dir", required=True, type=Path)
    parser.add_argument("--infer-dir", required=True, type=Path)
    parser.add_argument("--image-root", type=Path, default=Path("."))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--no-attach",
        action="store_true",
        help="do not write the passing report into the policy's model.parity",
    )
    args = parser.parse_args()

    candidate = json.loads((args.candidate / "candidate.json").read_text())
    evaluation = json.loads((args.candidate / "evaluation.json").read_text())
    model_path = Path(evaluation["model"]["path"])
    model_sha256 = sha256_file(model_path)
    if model_sha256 != candidate["model"]["sha256"]:
        raise RuntimeError(
            f"{model_path} hashes to {model_sha256[:12]}… but the candidate "
            f"records {candidate['model']['sha256'][:12]}…; P3 refuses to "
            "validate a different artifact than the evidence describes"
        )

    policy = json.loads(args.policy.read_text())
    if policy["model_sha256"] != model_sha256:
        raise RuntimeError(
            "the policy is not bound to this candidate's ONNX; P3 validates the "
            "deployable pair, not arbitrary combinations"
        )
    (target,) = [t for t in policy["targets"] if t["class"] == "bobcat"]

    label = candidate["candidate_id"]
    width, height = candidate["input"]["width"], candidate["input"]["height"]

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    config = PreprocessConfig(width=width, height=height)

    pruned_fp32 = candidate.get("kind") in ("pruned_fp32",)
    if pruned_fp32:
        print("--- P3 check 1 (fp32-pruned): graph validity + physical shapes")
        graph = check_graph_fp32_pruned(
            model_path, args.candidate / "p3_coverage", label, candidate
        )
    else:
        print("--- P3 check 1: graph validity + operator/dtype coverage")
        graph = check_graph_and_coverage(
            model_path, args.candidate / "p3_coverage", label
        )

    print("--- P3 check 2: validation metrics reproduce exactly")
    metrics_check = check_metrics_reproduce(
        args.candidate, args.candidate / "p3_reevaluation"
    )

    print("--- P3 check 3: ORT python vs C++ (logits + decision layers)")
    logits_layer = compare_logits_layer(
        session, args.p1_dir, args.cpp_logits_dir, (3, height, width)
    )
    decision_layer = compare_decision_layer(
        session,
        args.infer_dir,
        config,
        evaluation["class_names"],
        float(target["threshold"]),
        args.image_root,
    )

    print("--- P3 check 4: no silent fallback (model/policy bindings)")
    binding = check_binding(args.infer_dir, model_sha256, policy["policy_id"])

    checks = {
        "graph_and_coverage": graph,
        "metrics_reproduce": metrics_check,
        "logits_layer": {
            "fixtures": len(logits_layer),
            "worst_max_abs": max(r["logits_max_abs"] for r in logits_layer),
            "passed": all(r["passed"] for r in logits_layer),
            "results": logits_layer,
        },
        "decision_layer": {
            "fixtures": len(decision_layer),
            "worst_max_abs": max(r["logits_max_abs"] for r in decision_layer),
            "passed": all(r["passed"] for r in decision_layer),
            "results": decision_layer,
        },
        "binding": binding,
    }
    failed = [name for name, check in checks.items() if not check["passed"]]

    report = {
        "gate": (
            "P3-equivalent — pruned FP32 candidate validation (m3_registration §5)"
            if pruned_fp32
            else "P3 — quantized-model validation (DESIGN §10)"
        ),
        "candidate_id": label,
        "onnx": {"path": str(model_path), "sha256": model_sha256},
        "policy_id": policy["policy_id"],
        "ort_version_python": ort.__version__,
        "tolerances": {
            "logits_max_abs": LOGITS_MAX_ABS,
            "decision_carve_out": DECISION_CARVE_OUT,
            "metrics": "exact equality",
        },
        "checks": checks,
        "verdict": {"passed": not failed, "failed": failed},
    }
    atomic_write_json(args.output, report)
    print(f"P3 {'PASSED' if not failed else 'FAILED'} "
          f"(failed checks: {failed if failed else 'none'})")
    print(f"wrote {args.output}")

    if not failed and not args.no_attach:
        attach_parity(args.policy, args.class_map, args.output)
        print(f"attached parity to {args.policy}")

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
