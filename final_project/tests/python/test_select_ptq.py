"""The pre-registered PTQ selection rule, as a computation over evidence files.

The fabricated world gives three methods distinguishable on every axis the rule
reads: coverage eligibility, the §7.2 primary, and the material-drop distance
from the reference. What must never happen silently — a float-fallback
candidate winning on F2, a debugging obligation vanishing, QOperator appearing
unwarranted — is what these tests pin.
"""

from __future__ import annotations

import json

import pytest

from wildlife_trigger.optimize import select_ptq as sp


def write_candidate(
    root,
    method: str,
    primary: float,
    cis_f2: float,
    trans_f2: float,
    integer_execution: bool = True,
    size: int = 2_300_000,
    threshold: float = 0.5381,
    status: str = "recall_floor_infeasible",
):
    directory = root / method
    directory.mkdir(parents=True, exist_ok=True)
    sha = f"{method:0<8}" + "0" * 56
    (directory / "candidate.json").write_text(json.dumps({
        "candidate_id": f"d1_m1_ptq_{method}",
        "method": method,
        "model": {"sha256": sha},
    }))
    (directory / "coverage.json").write_text(json.dumps({
        "verdict": {
            "integer_execution": integer_execution,
            "float_compute_executed": {} if integer_execution else {"Conv": {"float": 52}},
        },
    }))
    (directory / "evaluation.json").write_text(json.dumps({
        "label": f"d1_m1_ptq_{method}",
        "model": {"sha256": sha, "bytes": size},
        "selection_score": {
            "primary": primary,
            "primary_metric": "mean_bobcat_frame_f2_at_0.5",
            "tiebreak_1_sequence_balanced_recall": 0.5,
            "tiebreak_2_macro_f1": 0.4,
        },
        "domains": {
            "cis_val_clean": {"target": {"frame_f2": cis_f2, "average_precision": 0.6}},
            "trans_val": {"target": {"frame_f2": trans_f2, "average_precision": 0.1}},
        },
    }))
    (directory / "calibration.json").write_text(json.dumps({
        "selection": {
            "threshold": threshold,
            "status": status,
            "primary_rule_met": status == "primary_rule_met",
        },
    }))


def write_reference(root, primary=0.3663, cis_f2=0.6272, trans_f2=0.1054):
    directory = root / "m0_fp32_reference"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "evaluation.json").write_text(json.dumps({
        "label": "d1_m0_fp32_ort_reference",
        "selection_score": {"primary": primary},
        "domains": {
            "cis_val_clean": {"target": {"frame_f2": cis_f2}},
            "trans_val": {"target": {"frame_f2": trans_f2}},
        },
    }))


@pytest.fixture()
def root(tmp_path):
    write_reference(tmp_path)
    return tmp_path


def test_highest_primary_wins_among_eligible(root):
    write_candidate(root, "minmax", primary=0.350, cis_f2=0.61, trans_f2=0.09)
    write_candidate(root, "entropy", primary=0.362, cis_f2=0.62, trans_f2=0.104)
    write_candidate(root, "percentile", primary=0.358, cis_f2=0.615, trans_f2=0.10)
    report = sp.select(root, ["minmax", "entropy", "percentile"], "m0_fp32_reference")
    assert report["selected"]["method"] == "entropy"
    assert [r["method"] for r in report["ranking"]] == ["entropy", "percentile", "minmax"]


def test_float_fallback_cannot_win_on_f2(root):
    # The best F2 belongs to a candidate that runs float convs: excluded.
    write_candidate(root, "minmax", primary=0.999, cis_f2=0.99, trans_f2=0.99,
                    integer_execution=False)
    write_candidate(root, "entropy", primary=0.300, cis_f2=0.55, trans_f2=0.05)
    report = sp.select(root, ["minmax", "entropy"], "m0_fp32_reference")
    assert report["selected"]["method"] == "entropy"
    assert report["excluded"] == [
        {"method": "minmax", "reason": "float compute survived optimization"}
    ]
    # And a QDQ float fallback is exactly what makes QOperator worth generating.
    assert report["qoperator"]["warranted"] is True
    assert report["qoperator"]["qdq_float_fallback_methods"] == ["minmax"]


def test_no_eligible_candidate_is_its_own_verdict(root):
    write_candidate(root, "minmax", primary=0.3, cis_f2=0.5, trans_f2=0.05,
                    integer_execution=False)
    report = sp.select(root, ["minmax"], "m0_fp32_reference")
    assert report["verdict"] == "no_eligible_candidate"
    assert "selected" not in report


def test_bytes_break_exact_ties(root):
    write_candidate(root, "minmax", primary=0.36, cis_f2=0.6, trans_f2=0.1, size=2_400_000)
    write_candidate(root, "entropy", primary=0.36, cis_f2=0.6, trans_f2=0.1, size=2_300_000)
    report = sp.select(root, ["minmax", "entropy"], "m0_fp32_reference")
    assert report["selected"]["method"] == "entropy"


def test_material_drop_triggers_are_computed_not_asserted(root):
    # primary ratio 0.354/0.3663 = 0.966 > 0.95: primary not triggered; but
    # trans F2 drops 0.1054 -> 0.09 = -14.6% relative: domain trigger fires.
    write_candidate(root, "minmax", primary=0.354, cis_f2=0.618, trans_f2=0.09)
    report = sp.select(root, ["minmax"], "m0_fp32_reference")
    drop = report["material_drop"]
    assert drop["primary_triggered"] is False
    assert drop["per_domain"]["trans_val"]["triggered"] is True
    assert drop["quantization_debugging_required"] is True


def test_small_drops_do_not_trigger_debugging(root):
    write_candidate(root, "minmax", primary=0.360, cis_f2=0.62, trans_f2=0.101)
    report = sp.select(root, ["minmax"], "m0_fp32_reference")
    assert report["material_drop"]["quantization_debugging_required"] is False
    assert report["qoperator"]["warranted"] is False


def test_inconsistent_candidate_directory_is_refused(root):
    write_candidate(root, "minmax", primary=0.36, cis_f2=0.6, trans_f2=0.1)
    evaluation = json.loads((root / "minmax" / "evaluation.json").read_text())
    evaluation["model"]["sha256"] = "e" * 64
    (root / "minmax" / "evaluation.json").write_text(json.dumps(evaluation))
    with pytest.raises(RuntimeError, match="inconsistent"):
        sp.select(root, ["minmax"], "m0_fp32_reference")


def test_markdown_names_the_selection_and_the_obligations(root):
    write_candidate(root, "minmax", primary=0.30, cis_f2=0.52, trans_f2=0.08)
    report = sp.select(root, ["minmax"], "m0_fp32_reference")
    text = sp.render_markdown(report)
    assert "Selected: minmax" in text
    assert "REQUIRED" in text  # 0.30/0.3663 = 0.819 < 0.95
