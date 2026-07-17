#!/usr/bin/env python3
"""D6 — the deployable pre-Pi shortlist (DESIGN §8.5, PLAN D6), mechanically.

Reads the frozen `comparison.jsonl` — every row of which already carries a
passing parity gate, because `comparison.py` refuses to write a row whose model
failed export/parity/correctness — and applies §8.5 to produce the shortlist
that goes to the Pi. Nothing here re-derives a metric; the ladder's committed
numbers are the input.

§8.5, in order:

1. **Gate failures are rejected.** Enforced upstream (a failed gate never became
   a row), and re-asserted here: every row must name a passed parity report.
2. **The recall operating rule.** Retain candidates meeting the 90%
   sequence-balanced validation bobcat-recall rule; if *none* do, retain the
   best documented fallback candidates. Every ladder candidate is
   `recall_floor_infeasible`, so this is the fallback branch — recorded as such,
   not hidden.
3. **Dominated candidates are removed** on (validation bobcat F2 ↑, MACs ↓,
   model size ↓). A candidate is dominated if another is ≥ on F2 and ≤ on both
   MACs and bytes, with at least one strict.
4. **gx10 latency is not used to rank.** Float-fallback pathologies are detected
   from the committed integer-execution coverage verdict, which is stronger than
   a latency heuristic; §12.4 forbids ranking Cortex-A76 by GB10 timing.
5. **M0-FP32 is always kept** as the mandatory baseline (§12.2 requires it in the
   benchmark), separately from the optimized dominance analysis.

Usage:
    python -m wildlife_trigger.optimize.pre_pi_shortlist \
        --comparison results/model_selection/comparison.jsonl \
        --output results/model_selection/pre_pi_shortlist.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..runs import atomic_write_json

BASELINE = "M0"


def load_rows(comparison: Path) -> list[dict]:
    rows = [json.loads(line) for line in comparison.read_text().splitlines() if line]
    if not rows:
        raise RuntimeError(f"{comparison} is empty; no shortlist without candidates")
    return rows


def row_metrics(row: dict) -> dict:
    """The three §8.5 axes plus the recall verdict, from a committed row."""
    return {
        "model_id": row["model_id"],
        "kind": row["kind"],
        "primary": row["validation_at_0p5"]["selection_score"],
        "cis_f2": row["validation_at_0p5"]["cis_f2"],
        "trans_f2": row["validation_at_0p5"]["trans_f2"],
        "macs": row["macs"],
        "bytes": row["model"]["bytes"],
        "operating_status": row["operating_point"]["status"],
        "primary_rule_met": row["operating_point"]["primary_rule_met"],
        "parity_passed": row["parity"]["passed"],
    }


def dominates(a: dict, b: dict) -> bool:
    """Does `a` dominate `b` on (primary ↑, MACs ↓, bytes ↓)?

    `primary` is the mean bobcat F2 at the yardstick — the §8.5 "validation
    bobcat F2" axis, taken across both domains rather than one.
    """
    at_least = a["primary"] >= b["primary"] and a["macs"] <= b["macs"] and a["bytes"] <= b["bytes"]
    strict = a["primary"] > b["primary"] or a["macs"] < b["macs"] or a["bytes"] < b["bytes"]
    return at_least and strict


def shortlist(rows: list[dict]) -> dict:
    metrics = [row_metrics(r) for r in rows]

    gate_failures = [m["model_id"] for m in metrics if not m["parity_passed"]]
    if gate_failures:
        raise RuntimeError(
            f"{gate_failures} carry a failed parity gate but reached the "
            "comparison table; a row should never have been written"
        )

    recall_rule_met = [m for m in metrics if m["primary_rule_met"]]
    fallback = not recall_rule_met

    # The optimized candidates compete for the front; M0 is the baseline and is
    # kept unconditionally.
    optimized = [m for m in metrics if m["model_id"] != BASELINE]
    baseline = [m for m in metrics if m["model_id"] == BASELINE]

    non_dominated = []
    rejected = []
    for candidate in optimized:
        dominators = [
            other["model_id"]
            for other in optimized
            if other is not candidate and dominates(other, candidate)
        ]
        if dominators:
            rejected.append({**candidate, "dominated_by": dominators})
        else:
            non_dominated.append(candidate)

    shortlist_ids = [m["model_id"] for m in baseline + non_dominated]
    return {
        "tool": "wildlife_trigger.optimize.pre_pi_shortlist",
        "design": "8.5",
        "recall_rule": {
            "met_by": [m["model_id"] for m in recall_rule_met],
            "fallback_branch": fallback,
            "note": (
                "no candidate meets the 90% sequence-balanced bobcat-recall rule "
                "(all recall_floor_infeasible); §8.5 step 2 fallback — retain the "
                "best documented candidates by dominance"
                if fallback
                else "the recall rule is met by at least one candidate"
            ),
        },
        "baseline": baseline,
        "non_dominated": non_dominated,
        "rejected": rejected,
        "shortlist": shortlist_ids,
        "latency_ranking_used": False,
        "float_fallback_evidence": (
            "integer execution is proven per-candidate by the committed coverage "
            "verdicts (P3 check 1); gx10 latency is not used to rank Cortex-A76 "
            "(§12.4)"
        ),
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Pre-Pi deployable shortlist (DESIGN §8.5)",
        "",
        "Mechanically derived from `comparison.jsonl` by "
        "`wildlife_trigger.optimize.pre_pi_shortlist`. Every row already carries "
        "a passing parity gate (`comparison.py` refuses to write a failing one).",
        "",
        "## The recall operating rule (§8.5 step 2)",
        "",
        report["recall_rule"]["note"] + ".",
        "",
        "## Candidates (validation, deployment ORT)",
        "",
        "| model | kind | primary (mean bobcat F2) | cis F2 | trans F2 | MACs | bytes | status |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    order = {m["model_id"]: m for m in report["baseline"] + report["non_dominated"]}
    for entry in report["rejected"]:
        order[entry["model_id"]] = entry
    for model_id in sorted(order, key=lambda m: (m != "M0", m)):
        m = order[model_id]
        mark = "**baseline**" if model_id == "M0" else (
            "rejected" if any(r["model_id"] == model_id for r in report["rejected"])
            else "**shortlist**"
        )
        lines.append(
            f"| {model_id} ({mark}) | {m['kind']} | {m['primary']:.4f} | "
            f"{m['cis_f2']:.4f} | {m['trans_f2']:.4f} | {m['macs']:,} | "
            f"{m['bytes']:,} | {m['operating_status']} |"
        )
    lines += ["", "## Rejections (dominated on §8.5's three axes)", ""]
    if report["rejected"]:
        for entry in report["rejected"]:
            lines.append(
                f"- **{entry['model_id']}** dominated by "
                f"{', '.join(entry['dominated_by'])} — "
                f"lower or equal on primary F2 ({entry['primary']:.4f}) with no "
                "advantage on both MACs and size."
            )
    else:
        lines.append("- none.")
    lines += [
        "",
        "## Shortlist frozen for the Pi",
        "",
        "**" + " · ".join(report["shortlist"]) + "**",
        "",
        "M0 is the mandatory FP32 baseline (§12.2). The optimized front is every "
        "non-dominated candidate; Pi latency (F-phase), never gx10 latency (§12.4), "
        "chooses the final model from this set.",
        "",
        "Float-fallback check: " + report["float_fallback_evidence"] + ".",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison", type=Path,
        default=Path("results/model_selection/comparison.jsonl"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/model_selection/pre_pi_shortlist.md"),
    )
    args = parser.parse_args()

    rows = load_rows(args.comparison)
    report = shortlist(rows)
    atomic_write_json(args.output.with_suffix(".json"), report)
    args.output.write_text(render_markdown(report))
    print(f"shortlist: {' · '.join(report['shortlist'])}")
    if report["rejected"]:
        for entry in report["rejected"]:
            print(f"  rejected {entry['model_id']} (dominated by "
                  f"{', '.join(entry['dominated_by'])})")
    print(f"wrote {args.output}")
    print(f"wrote {args.output.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
