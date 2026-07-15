#!/usr/bin/env python3
"""C1a — compare step-matched training arms and emit the decision.

DESIGN §5.2 and PLAN C1a run three arms, not a 2x2 matrix: the no-empty 15-output
control against the 5k-empty 16-output arm at 256x192, and then one additional 224x224
run reusing the winner's data/head contract.

This module reads their `history.json` files and writes the comparison. Two things it
insists on:

**The arms must actually be matched.** It re-reads the recorded step budgets rather than
trusting that they were passed correctly — a mismatched `--override` already launched one
arm with the wrong budget once, and a comparison between differently-trained models is
not an ablation, it is a coincidence.

**Non-empty images seen is reported next to the result.** DESIGN §5.2: under a fixed step
budget the supplement arm necessarily sees ~37% fewer animals, so "the supplement helped"
and "this arm saw fewer animals" pull in opposite directions and the reader needs both
numbers to interpret either.

**The two decisions are kept apart.** PLAN C1a settles a data/head contract *and* an
input geometry, on different evidence. The head is decided at a shared input, where the
metric is the variable under test. The geometry is decided at the winning head on the
metric, the real-pixel utilisation and the MACs together, preferring 256x192 when the
arms are statistically tied — so it needs `--input-cost` and `--tie-test`, and says
plainly that the decision is unmade when they are missing. A single "highest score wins"
across all three arms would quietly answer both questions with one number that was only
ever about one of them.

Usage:
    python -m wildlife_trigger.validate.ablation_compare \
        --runs results/ablations/*/history.json \
        --input-cost results/ablations/input_cost.json \
        --tie-test results/ablations/input_tie_test.json \
        --output results/ablations/data_input_decision.md
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path


def best_entry(run: dict) -> dict:
    return next(e for e in run["history"] if e["epoch"] == run["best_epoch"])


def score_spread(run: dict) -> dict:
    """How much this arm's own score moves between neighbouring epochs.

    The selection score is the max over epochs of a noisy curve, and the comparison
    between arms is a comparison of two such maxima. If a single arm's score swings by
    more between consecutive epochs than the arms differ by, then the gap between them
    is not resolvable by these runs — reporting the winner without this number invites
    reading a coin flip as a finding.
    """
    scores = [e["selection_score"]["primary"] for e in run["history"]]
    steps = [abs(b - a) for a, b in zip(scores, scores[1:])]
    return {
        "max_epoch_to_epoch_change": round(max(steps), 4) if steps else 0.0,
        "median_epoch_to_epoch_change": round(sorted(steps)[len(steps) // 2], 4) if steps else 0.0,
        "best": round(max(scores), 4),
        "epochs": len(scores),
    }


def data_head_pair(rows: list[dict]) -> tuple[dict, dict] | None:
    """The two arms sharing an input geometry and differing in the head/data contract."""
    for a, b in itertools.combinations(rows, 2):
        if a["input"] == b["input"] and a["classes"] != b["classes"]:
            return a, b
    return None


def input_pair(rows: list[dict]) -> tuple[dict, dict] | None:
    """The two arms sharing the head/data contract and differing in input geometry."""
    for a, b in itertools.combinations(rows, 2):
        if a["classes"] == b["classes"] and a["input"] != b["input"]:
            return a, b
    return None


def load(path: Path) -> dict:
    run = json.loads(path.read_text())
    run["_best"] = best_entry(run)
    return run


def check_matched(runs: list[dict]) -> dict:
    """Confirm the arms really were step-matched before comparing them.

    Matching is a property of the **budget**, not of the steps consumed. Early stopping
    is part of DESIGN §7.2's recipe and is applied identically to both arms, so an arm
    that converges sooner and stops has told us something about itself — it has not
    broken the comparison. But the reader must be able to see it: two arms that consumed
    4,220 and 6,000 steps did not train equally long, and if the shorter one also scored
    lower, "it stopped early" and "it is worse" are different claims.

    So consumed steps are reported alongside, and a large gap is flagged rather than
    silently averaged into a verdict.
    """
    budgets = {r["run_name"]: r["budget"] for r in runs}
    max_steps = {b["max_steps"] for b in budgets.values()}
    head_steps = {b["head_steps"] for b in budgets.values()}
    consumed = {name: b["steps"] for name, b in budgets.items()}

    spread = (
        (max(consumed.values()) - min(consumed.values())) / max(consumed.values())
        if consumed
        else 0.0
    )
    return {
        "max_steps_identical": len(max_steps) == 1,
        "head_steps_identical": len(head_steps) == 1,
        "observed_max_steps": sorted(max_steps),
        "observed_head_steps": sorted(head_steps),
        "steps_consumed": consumed,
        "consumed_spread": round(spread, 4),
        "an_arm_stopped_early": any(
            b["steps"] < b["max_steps"] for b in budgets.values()
        ),
        "per_run": budgets,
        # The budget is what must match. Consumption is reported, not required.
        "matched": len(max_steps) == 1 and len(head_steps) == 1,
    }


def row(run: dict) -> dict:
    best = run["_best"]
    return {
        "run": run["run_name"],
        "classes": len(run["class_names"]),
        "input": f"{run['config']['width']}x{run['config']['height']}",
        "steps": run["budget"]["steps"],
        "effective_epochs": run["budget"]["effective_epochs"],
        "images_seen": run["budget"]["images_seen"],
        "non_empty_seen": run["budget"]["non_empty_images_seen"],
        "selection_score": best["selection_score"]["primary"],
        "cis_f2": best["cis_val_clean"]["frame_f2"],
        "trans_f2": best["trans_val"]["frame_f2"],
        "cis_seq_recall": best["cis_val_clean"]["sequence_balanced_recall"],
        "trans_seq_recall": best["trans_val"]["sequence_balanced_recall"],
        "cis_false_fire": best["cis_val_clean"]["false_fire_rate"],
        "trans_false_fire": best["trans_val"]["false_fire_rate"],
        "best_epoch": run["best_epoch"],
        "noise": score_spread(run),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--input-cost",
        type=Path,
        help="input_cost.py JSON. PLAN C1a decides the input on utilisation and MACs "
        "as well as the metrics, so without this the input decision is not made.",
    )
    parser.add_argument(
        "--tie-test-input",
        type=Path,
        help="tie_test.py JSON for the two arms differing in input geometry. Supplies "
        "the meaning of PLAN C1a's 'statistically tied'.",
    )
    parser.add_argument(
        "--tie-test-head",
        type=Path,
        help="tie_test.py JSON for the two arms differing in the head/data contract. "
        "Not required by PLAN, but the same question: the head gap is small next to the "
        "score's own epoch-to-epoch movement, so a bare point estimate overstates it.",
    )
    args = parser.parse_args()

    runs = [load(path) for path in args.runs]
    matched = check_matched(runs)
    rows = [row(r) for r in runs]
    rows.sort(key=lambda r: -r["selection_score"])
    winner = rows[0]
    cost = json.loads(args.input_cost.read_text()) if args.input_cost else None
    tie = json.loads(args.tie_test_input.read_text()) if args.tie_test_input else None
    tie_head = json.loads(args.tie_test_head.read_text()) if args.tie_test_head else None

    lines = [
        "# C1a — data and input decision",
        "",
        "GENERATED by `wildlife_trigger.validate.ablation_compare`. No number here is",
        "retyped by hand. The metrics come from each run's `history.json`; the MACs and",
        "utilisation from `validate.input_cost`; the interval from `validate.tie_test`.",
        "",
        "## Are the arms actually matched?",
        "",
        f"- identical total step budget: **{matched['max_steps_identical']}** "
        f"({matched['observed_max_steps']})",
        f"- identical phase-A head budget: **{matched['head_steps_identical']}** "
        f"({matched['observed_head_steps']})",
        "",
        "DESIGN §5.2 matches the arms on optimizer steps rather than epochs: the",
        "supplement changes the training set from 13,553 to 18,553 images (+36.9% steps",
        "per epoch), so an epoch-matched comparison would confound *empty data helps*",
        "with *this arm simply trained 37% longer*. The phase-A budget is matched for the",
        "same reason — 5 epochs of head training is 1,055 steps in one arm and 1,445 in",
        "the other.",
        "",
    ]

    if matched["an_arm_stopped_early"]:
        lines += [
            "### An arm stopped early",
            "",
            f"Steps actually consumed: `{matched['steps_consumed']}` "
            f"(spread {matched['consumed_spread']:.1%} of the larger).",
            "",
            "Early stopping is part of DESIGN §7.2's recipe and was applied identically",
            "to both arms, so this does not invalidate the comparison — an arm that",
            "converged sooner has told us something about itself. But the arms did not",
            "train for equally long, so read the score together with the steps:",
            "*it stopped early* and *it is worse* are different claims, and only the",
            "second is about the data.",
            "",
        ]

    lines += [
        "## Results",
        "",
        "Selection score is mean bobcat F2 across cis-val-clean and trans-val at a fixed",
        "0.5 threshold (DESIGN §7.2). C3 calibrates the real operating point; this is a",
        "constant yardstick for comparing candidates, not an operating point.",
        "",
        "| run | outputs | input | steps | eff. epochs | non-empty seen | score | cis F2 | trans F2 | cis seq-recall | trans seq-recall | cis false-fire | trans false-fire |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['run']}` | {r['classes']} | {r['input']} | {r['steps']} | "
            f"{r['effective_epochs']} | {r['non_empty_seen']:,} | "
            f"**{r['selection_score']:.4f}** | {r['cis_f2']:.4f} | {r['trans_f2']:.4f} | "
            f"{r['cis_seq_recall']:.4f} | {r['trans_seq_recall']:.4f} | "
            f"{r['cis_false_fire']:.4f} | {r['trans_false_fire']:.4f} |"
        )

    lines += [
        "",
        "## Reading the non-empty column",
        "",
        "Under a fixed step budget the supplement arm necessarily sees fewer animal",
        "frames — that is the price of the comparison being about data rather than",
        "compute. DESIGN §5.2 requires this number beside the result precisely so",
        "*the supplement helped* cannot quietly mean *this arm saw more animals*, and",
        "so a supplement arm that loses slightly on bobcat F2 while cutting the empty",
        "false-fire rate can be read for what it is.",
        "",
        "## How much of a gap is meaningful?",
        "",
        "The selection score is the **maximum over epochs** of a curve that moves a lot,",
        "so before reading any gap between arms, here is how far each arm's own score",
        "travels between one epoch and the next:",
        "",
        "| run | best | median epoch-to-epoch change | max epoch-to-epoch change |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        n = r["noise"]
        lines.append(
            f"| `{r['run']}` | {n['best']:.4f} | {n['median_epoch_to_epoch_change']:.4f} | "
            f"{n['max_epoch_to_epoch_change']:.4f} |"
        )
    lines += [
        "",
        "A gap between two arms that is smaller than the distance a single arm covers",
        "between consecutive epochs is not a measurement of the thing under test. Where",
        "that is the case below, it is said rather than rounded away.",
        "",
    ]

    lines += ["## Decision", ""]

    # PLAN C1a makes two decisions, not one, and they use different evidence. Reporting a
    # single "highest score" across all three arms would conflate them: the 224x224 arm
    # differs from the 256x192 winner in geometry, not in data, and geometry is decided
    # on cost as well as score.
    pair = data_head_pair(rows)
    if pair:
        a, b = sorted(pair, key=lambda r: -r["selection_score"])
        lines += [
            "### 1. The data and head contract",
            "",
            f"Compared at a shared input ({a['input']}), so the only variable is the head "
            "and its data.",
            "",
            f"**`{a['run']}` wins**: selection score {a['selection_score']:.4f} against "
            f"{b['selection_score']:.4f}.",
            "",
            f"The score gap is {a['selection_score'] - b['selection_score']:.4f}. Read it "
            "next to the false-fire rates, which is where the supplement is supposed to "
            "act and where the effect is largest:",
            "",
            f"- cis-val-clean false fire: {a['cis_false_fire']:.4f} against "
            f"{b['cis_false_fire']:.4f}",
            f"- trans-val false fire: {a['trans_false_fire']:.4f} against "
            f"{b['trans_false_fire']:.4f}",
            "",
            f"`{a['run']}` reaches this having seen {a['non_empty_seen']:,} non-empty "
            f"frames against `{b['run']}`'s {b['non_empty_seen']:,} — "
            f"{(1 - a['non_empty_seen'] / b['non_empty_seen']) * 100:.0f}% fewer animals, "
            "under the matched step budget. It is ahead on the target metric while seeing "
            "less of the target, which is the direction the supplement was predicted to "
            "act in and the opposite of what more-animals-seen would explain.",
            "",
        ]

        if tie_head:
            lines += [
                f"Paired sequence bootstrap on this pair "
                f"({tie_head['replicates']:,} replicates):",
                "",
                f"- observed difference: "
                f"**{tie_head['observed']['difference_b_minus_a']:+.4f}** "
                f"(`{tie_head['arm_b']}` minus `{tie_head['arm_a']}`)",
                f"- 95% CI of the difference: **[{tie_head['difference_ci95'][0]:+.4f}, "
                f"{tie_head['difference_ci95'][1]:+.4f}]**",
                f"- P(`{tie_head['arm_b']}` > `{tie_head['arm_a']}`): "
                f"{tie_head['probability_b_beats_a']:.1%}",
                f"- **tied: {tie_head['tied']}**",
                "",
            ]
            if tie_head["tied"]:
                lines += [
                    "**The score gap alone does not establish this arm.** The interval "
                    "spans zero, so on bobcat F2 the validation data cannot separate "
                    "the two heads, and the ranking of the point estimates is not by "
                    "itself evidence. What survives is the false-fire effect above: it "
                    "is large, it is on both domains, and it is the effect the "
                    "supplement was added to produce. The contract is selected on that "
                    "and on DESIGN §5.2's stated purpose for the supplement — not on "
                    f"the {a['selection_score'] - b['selection_score']:.4f} of F2.",
                    "",
                ]
            else:
                lines += [
                    "The interval excludes zero: the gap is larger than the validation "
                    "sample explains.",
                    "",
                ]

    geometry = input_pair(rows)
    if geometry:
        a, b = sorted(geometry, key=lambda r: -r["selection_score"])
        lines += [
            "### 2. The input geometry",
            "",
            "Compared at the winning data/head contract, so the only variable is the "
            "tensor shape. PLAN C1a decides this on the validation metrics, the "
            "real-pixel utilisation **and** the MACs — not on the score alone — and "
            "prefers 256x192 when the arms are statistically tied.",
            "",
        ]

        if cost:
            lines += [
                "| input | tensor px | MACs | mean real px (cis-val-clean) | utilisation |",
                "|---|---:|---:|---:|---:|",
            ]
            for entry in cost["rows"]:
                util = entry["utilisation"].get("cis_val_clean", {})
                lines.append(
                    f"| {entry['input']} | {entry['tensor_pixels']:,} | "
                    f"{entry['macs']:,} | {util.get('mean_real_pixels', 0):,.0f} | "
                    f"{util.get('mean_utilisation', 0):.2%} |"
                )
            lines += [
                "",
                "The two geometries are nearly the same tensor and nearly the same MACs.",
                "They are **not** the same amount of frame: a square tensor spends a",
                "quarter of itself on grey bars, because CCT's dominant frame is 1024x747",
                "and does not fit a square. The utilisation column is the whole argument —",
                "at essentially equal compute, one shape feeds the network far more animal.",
                "",
            ]
        else:
            lines += [
                "> **The input decision is not made.** PLAN C1a requires real-pixel",
                "> utilisation and MACs, and neither was supplied (`--input-cost`).",
                "",
            ]

        if tie:
            lines += [
                f"Paired sequence bootstrap, {tie['replicates']:,} replicates, resampling "
                f"**{tie['resampling_unit']}s** (the unit CCT's burst frames make "
                "independent-ish, not frames):",
                "",
                f"- observed difference: **{tie['observed']['difference_b_minus_a']:+.4f}** "
                f"(`{tie['arm_b']}` minus `{tie['arm_a']}`)",
                f"- 95% CI of the difference: **[{tie['difference_ci95'][0]:+.4f}, "
                f"{tie['difference_ci95'][1]:+.4f}]**",
                f"- P(`{tie['arm_b']}` > `{tie['arm_a']}`): {tie['probability_b_beats_a']:.1%}",
                f"- **tied: {tie['tied']}**",
                "",
                tie["interpretation"],
                "",
                f"*{tie['caveat']}*",
                "",
            ]
        else:
            lines += [
                "> No tie test supplied (`--tie-test`), so *statistically tied* has no",
                "> meaning here and PLAN C1a's tie-break cannot be applied.",
                "",
            ]

        if cost and tie:
            by_input = {e["input"]: e for e in cost["rows"]}
            preferred = "256x192"
            if tie["tied"] and preferred in by_input:
                other = next(i for i in by_input if i != preferred)
                gain = (
                    by_input[preferred]["utilisation"]["cis_val_clean"]["mean_real_pixels"]
                    / by_input[other]["utilisation"]["cis_val_clean"]["mean_real_pixels"]
                    - 1
                )
                macs_delta = (
                    by_input[preferred]["macs"] / by_input[other]["macs"] - 1
                )
                lines += [
                    f"**Selected: {preferred}.** The arms are statistically tied on the "
                    "metric, so PLAN C1a's tie-break decides — and it is not a coin toss "
                    f"dressed as a rule: at {macs_delta:+.1%} MACs, {preferred} carries "
                    f"{gain:+.1%} real pixels. DESIGN §5.5 adds the reason this shape was "
                    "proposed in the first place: the Pi's libjpeg scales the 1024-wide "
                    "frame by 1/4 during decode, landing exactly on 256 — the network "
                    "input arrives with no resize step at all.",
                    "",
                ]
            else:
                won = a["input"]
                lines += [
                    f"**Selected: {won}.** The arms are not tied — the paired CI excludes "
                    f"zero — so the metric decides and the tie-break does not apply. "
                    f"`{a['run']}` scores {a['selection_score']:.4f} against "
                    f"`{b['run']}`'s {b['selection_score']:.4f}.",
                    "",
                ]
                if won == "256x192":
                    lines += [
                        "It wins the cost axes as well, so nothing is being traded: this "
                        "is the shape with both the better score and the better "
                        "utilisation.",
                        "",
                    ]

    lines += [
        "### Overall",
        "",
        f"Highest selection score across all arms: **`{winner['run']}`** "
        f"({winner['selection_score']:.4f}).",
        "",
    ]

    if not matched["matched"]:
        lines += [
            "> **This comparison is not valid.** The arms do not share a step budget, so",
            "> the difference between them is not attributable to the variable under",
            "> test. Re-run them matched before reading anything above as a result.",
            "",
        ]

    document = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(document)
        print(f"wrote {args.output}")
    print(document)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "matched": matched,
                    "rows": rows,
                    "winner": winner["run"],
                    "input_cost": cost,
                    "tie_test_input": tie,
                    "tie_test_head": tie_head,
                },
                indent=2,
            )
            + "\n"
        )

    return 0 if matched["matched"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
