#!/usr/bin/env python3
"""Apply D1's pre-registered PTQ selection rule — mechanically.

The rule lives in `results/optimize/m1_ptq/preregistration.md` (§3-§5),
committed before any candidate existed. This module is its executable form, so
the selection is a computation over evidence files rather than a judgement made
while looking at the numbers:

1. eligibility: `integer_execution == true` in the candidate's coverage verdict
   — a "quantized" model running float convolutions is mislabeled M0, not a
   candidate;
2. ranking: the frozen §7.2 selection key (mean bobcat frame F2 at 0.5, then
   sequence-balanced recall, then support-aware macro F1), final tie-break
   smaller ONNX bytes;
3. the material-drop triggers, measured against the M0 ORT reference: primary
   < 0.95x reference, or either domain's F2 down >10% relative. Triggered means
   quantization debugging is *required before the verdict is accepted* — the
   tool records the obligation, it cannot discharge it;
4. the QOperator rule: warranted only if QDQ coverage on the ARM64 host shows
   surviving float compute.

Usage (gx10):
    python -m wildlife_trigger.optimize.select_ptq \
        --root results/optimize/m1_ptq \
        --methods minmax entropy percentile
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..metrics import SELECTION_ORDER, selection_key
from ..runs import atomic_write_json

# Preregistration §4, verbatim.
PRIMARY_DROP_FACTOR = 0.95
DOMAIN_RELATIVE_DROP = 0.10

DOMAINS = ("cis_val_clean", "trans_val")


def load_candidate(root: Path, method: str) -> dict:
    directory = root / method
    candidate = json.loads((directory / "candidate.json").read_text())
    coverage = json.loads((directory / "coverage.json").read_text())
    evaluation = json.loads((directory / "evaluation.json").read_text())
    calibration = json.loads((directory / "calibration.json").read_text())

    if evaluation["model"]["sha256"] != candidate["model"]["sha256"]:
        raise RuntimeError(
            f"{directory}: candidate.json and evaluation.json describe different "
            "artifacts; refusing to rank an inconsistent directory"
        )
    return {
        "method": method,
        "candidate_id": candidate["candidate_id"],
        "directory": str(directory),
        "model_sha256": candidate["model"]["sha256"],
        "bytes": evaluation["model"]["bytes"],
        "integer_execution": coverage["verdict"]["integer_execution"],
        "float_compute_executed": coverage["verdict"]["float_compute_executed"],
        "selection_score": evaluation["selection_score"],
        "domains": {
            domain: {
                "frame_f2": evaluation["domains"][domain]["target"]["frame_f2"],
                "average_precision": evaluation["domains"][domain]["target"][
                    "average_precision"
                ],
            }
            for domain in DOMAINS
        },
        "operating_point": {
            "threshold": calibration["selection"]["threshold"],
            "status": calibration["selection"]["status"],
            "primary_rule_met": calibration["selection"]["primary_rule_met"],
        },
    }


def rank_key(entry: dict) -> tuple:
    """§7.2 key, then smaller bytes — the exact pre-registered order."""
    return (*selection_key(entry["selection_score"]), -entry["bytes"])


def material_drop(winner: dict, reference: dict) -> dict:
    """Preregistration §4: the debugging triggers, shown with their arithmetic."""
    ref_primary = reference["selection_score"]["primary"]
    primary = winner["selection_score"]["primary"]
    primary_triggered = primary < PRIMARY_DROP_FACTOR * ref_primary

    domain_checks = {}
    for domain in DOMAINS:
        ref_f2 = reference["domains"][domain]["target"]["frame_f2"]
        f2 = winner["domains"][domain]["frame_f2"]
        relative_drop = (ref_f2 - f2) / ref_f2 if ref_f2 > 0 else 0.0
        domain_checks[domain] = {
            "reference_f2": ref_f2,
            "candidate_f2": f2,
            "relative_drop": relative_drop,
            "triggered": relative_drop > DOMAIN_RELATIVE_DROP,
        }

    triggered = primary_triggered or any(c["triggered"] for c in domain_checks.values())
    return {
        "rule": (
            f"primary < {PRIMARY_DROP_FACTOR} x reference, or a domain F2 down "
            f"> {DOMAIN_RELATIVE_DROP:.0%} relative (preregistration §4)"
        ),
        "reference_primary": ref_primary,
        "candidate_primary": primary,
        "primary_ratio": primary / ref_primary if ref_primary > 0 else None,
        "primary_triggered": primary_triggered,
        "per_domain": domain_checks,
        "quantization_debugging_required": triggered,
    }


def qoperator_verdict(candidates: list[dict]) -> dict:
    """Preregistration §5: warranted only if QDQ left float compute running."""
    failing = [c["method"] for c in candidates if not c["integer_execution"]]
    return {
        "rule": (
            "QOperator is generated only if S8S8 QDQ coverage on the ARM64 host "
            "shows float Conv/Gemm/MatMul surviving optimization "
            "(preregistration §5)"
        ),
        "qdq_float_fallback_methods": failing,
        "warranted": bool(failing),
    }


def select(root: Path, methods: list[str], reference_dir: str) -> dict:
    reference = json.loads(
        (root / reference_dir / "evaluation.json").read_text()
    )
    candidates = [load_candidate(root, method) for method in methods]

    eligible = [c for c in candidates if c["integer_execution"]]
    excluded = [
        {"method": c["method"], "reason": "float compute survived optimization"}
        for c in candidates
        if not c["integer_execution"]
    ]
    if not eligible:
        return {
            "tool": "wildlife_trigger.optimize.select_ptq",
            "preregistration": "results/optimize/m1_ptq/preregistration.md",
            "verdict": "no_eligible_candidate",
            "excluded": excluded,
            "qoperator": qoperator_verdict(candidates),
            "candidates": candidates,
            "reference": {
                "label": reference["label"],
                "selection_score": reference["selection_score"],
            },
        }

    ranked = sorted(eligible, key=rank_key, reverse=True)
    winner = ranked[0]

    return {
        "tool": "wildlife_trigger.optimize.select_ptq",
        "preregistration": "results/optimize/m1_ptq/preregistration.md",
        "rule": {
            "eligibility": "integer_execution == true",
            "ranking": list(SELECTION_ORDER) + ["smaller_onnx_bytes"],
        },
        "selected": {
            "method": winner["method"],
            "candidate_id": winner["candidate_id"],
            "model_sha256": winner["model_sha256"],
            "directory": winner["directory"],
        },
        "ranking": [
            {
                "method": c["method"],
                "primary": c["selection_score"]["primary"],
                "tiebreak_1_sequence_balanced_recall": c["selection_score"][
                    "tiebreak_1_sequence_balanced_recall"
                ],
                "tiebreak_2_macro_f1": c["selection_score"]["tiebreak_2_macro_f1"],
                "bytes": c["bytes"],
                "cis_f2": c["domains"]["cis_val_clean"]["frame_f2"],
                "trans_f2": c["domains"]["trans_val"]["frame_f2"],
                "operating_point": c["operating_point"],
            }
            for c in ranked
        ],
        "excluded": excluded,
        "material_drop": material_drop(winner, reference),
        "qoperator": qoperator_verdict(candidates),
        "reference": {
            "label": reference["label"],
            "selection_score": reference["selection_score"],
            "domains": {
                domain: reference["domains"][domain]["target"]["frame_f2"]
                for domain in DOMAINS
            },
        },
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# D1 PTQ candidate selection",
        "",
        f"Rule: `{report['preregistration']}` — applied mechanically by "
        f"`{report['tool']}`.",
        "",
    ]
    if report.get("verdict") == "no_eligible_candidate":
        lines += [
            "**No eligible candidate**: every method left float compute running.",
            "",
        ]
        return "\n".join(lines)

    reference = report["reference"]
    lines += [
        f"Reference (M0 FP32 through deployment ORT): primary "
        f"{reference['selection_score']['primary']:.4f}, cis F2 "
        f"{reference['domains']['cis_val_clean']:.4f}, trans F2 "
        f"{reference['domains']['trans_val']:.4f}.",
        "",
        "| method | primary | cis F2 | trans F2 | bytes | threshold | status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for c in report["ranking"]:
        op = c["operating_point"]
        lines.append(
            f"| {c['method']} | {c['primary']:.4f} | {c['cis_f2']:.4f} "
            f"| {c['trans_f2']:.4f} | {c['bytes']:,} "
            f"| {op['threshold']:.6f} | {op['status']} |"
        )
    for exc in report["excluded"]:
        lines.append(f"| {exc['method']} | excluded: {exc['reason']} | | | | | |")

    drop = report["material_drop"]
    lines += [
        "",
        f"**Selected: {report['selected']['method']}** "
        f"(`{report['selected']['candidate_id']}`).",
        "",
        f"Material-drop check: primary ratio "
        f"{drop['primary_ratio']:.4f} vs the 0.95 line; "
        + ", ".join(
            f"{d} {checks['relative_drop']:+.1%}"
            for d, checks in drop["per_domain"].items()
        )
        + f" vs the -10% line → quantization debugging "
        f"{'REQUIRED' if drop['quantization_debugging_required'] else 'not triggered'}.",
        "",
        f"QOperator: {'warranted' if report['qoperator']['warranted'] else 'not warranted'} "
        f"({report['qoperator']['rule']}).",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/optimize/m1_ptq"))
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--reference-dir", default="m0_fp32_reference")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    report = select(args.root, args.methods, args.reference_dir)
    output = args.output or args.root / "selection.json"
    markdown = args.markdown or args.root / "selection.md"
    atomic_write_json(output, report)
    markdown.write_text(render_markdown(report))

    print(render_markdown(report))
    print(f"wrote {output}")
    print(f"wrote {markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
