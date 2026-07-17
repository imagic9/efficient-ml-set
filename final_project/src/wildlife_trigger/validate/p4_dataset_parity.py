#!/usr/bin/env python3
"""Gate P4 — C++ dataset parity on the validation manifests (DESIGN §10).

The corpus-scale version of the fixture checks: Python evaluation (the
candidate's committed `predictions.npz`, scored by `optimize.evaluate_onnx`)
against the C++ `run-dataset` output, frame by frame, in order. What P2/P3
proved on 200 fixtures this proves on every validation frame — including the
near-threshold band where a numeric gap actually flips decisions.

Per split, the registered comparison:

- the **header** must bind the JSONL to this model hash, this policy id, this
  class map, and this manifest's bytes — a runner that read anything else
  produces a file this comparator refuses;
- the **footer** must account for every frame: processed == manifest frames,
  skipped == 0 (the validation corpus is known-good; a skip is a finding);
- **ordered image ids** identical;
- **labels** identical (the runner echoes the manifest, proving it read the
  same rows), and the npz `present` column must equal the labels' verdict;
- **target scores** within 1e-4 (the same registered gate as P2/P3 — the INT8
  graph is expected to land near-bitwise, and the gate leaves room for float
  epilogue reassociation, not for bugs);
- **trigger decisions** identical, except frames whose Python score lies
  within 1e-4 of the threshold — those are listed by id, per the P2 carve-out;
- the **confusion matrix** (fire x target-present), computed on both sides
  over the non-carved frames, must be equal cell for cell.

Usage (gx10, after scripts/run_d1_p3p4.sh p4):
    python -m wildlife_trigger.validate.p4_dataset_parity \
        --candidate results/optimize/m1_ptq/<method> \
        --policy artifacts/policies/bobcat_m1_ptq_<method>_v1.json \
        --cpp-dir results/optimize/m1_ptq/<method>/p4 \
        --output results/optimize/m1_ptq/<method>/p4_dataset_parity.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..runs import atomic_write_json, sha256_file

SPLITS = ("cis_val_clean", "trans_val")

# The registered gates, shared with P2/P3 (DESIGN §10).
SCORE_MAX_ABS = 1e-4
DECISION_CARVE_OUT = 1e-4


def load_cpp_jsonl(path: Path) -> tuple[dict, list[dict], dict]:
    lines = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not lines or lines[0].get("kind") != "run_dataset_header":
        raise RuntimeError(f"{path} does not start with a run_dataset_header")
    if lines[-1].get("kind") != "run_dataset_footer":
        raise RuntimeError(
            f"{path} has no footer; the run did not finish and partial parity "
            "proves nothing"
        )
    return lines[0], lines[1:-1], lines[-1]


def compare_split(
    split: str,
    npz: np.lib.npyio.NpzFile,
    cpp_path: Path,
    manifest_path: Path,
    model_sha256: str,
    policy: dict,
    target: str,
    threshold: float,
    class_names: list[str],
    score_diagnostic: bool = False,
) -> dict:
    header, rows, footer = load_cpp_jsonl(cpp_path)
    failures: list[str] = []

    # Header bindings: the file must be about the artifact under test.
    if header["model_sha256"] != model_sha256:
        failures.append("header.model_sha256 differs from the candidate artifact")
    if header["policy_id"] != policy["policy_id"]:
        failures.append("header.policy_id differs from the policy under test")
    if header["class_map_sha256"] != policy["class_map_sha256"]:
        failures.append("header.class_map_sha256 differs from the policy's binding")
    manifest_sha = sha256_file(manifest_path)
    if header["manifest_sha256"] != manifest_sha:
        failures.append("header.manifest_sha256 differs from the manifest on disk")

    # Footer accounting: every frame, no silent drops.
    skipped = [r for r in rows if r.get("skipped")]
    scored = [r for r in rows if not r.get("skipped")]
    if footer["skipped"] != len(skipped) or footer["processed"] != len(scored):
        failures.append("footer counts disagree with the rows in the file")
    if skipped:
        failures.append(f"{len(skipped)} frames skipped; the validation corpus is known-good")

    python_ids = [str(i) for i in npz[f"{split}/image_ids"]]
    cpp_ids = [r["image_id"] for r in scored]
    if python_ids != cpp_ids:
        failures.append(
            f"ordered image ids differ (python {len(python_ids)}, c++ {len(cpp_ids)})"
        )
        return {"split": split, "passed": False, "failures": failures}

    # Labels: the runner echoes the manifest; python re-derives presence.
    manifest_labels = {}
    with manifest_path.open() as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                manifest_labels[record["image_id"]] = record["labels"]
    label_mismatches = [
        r["image_id"] for r in scored if r["labels"] != manifest_labels[r["image_id"]]
    ]
    if label_mismatches:
        failures.append(f"{len(label_mismatches)} label rows differ from the manifest")

    column = class_names.index(target)
    present = npz[f"{split}/present"][:, column] > 0
    labels_say = np.array([target in manifest_labels[i] for i in python_ids])
    if not np.array_equal(present, labels_say):
        failures.append(
            "npz present column disagrees with manifest labels; the npz does not "
            "belong to this manifest"
        )

    python_scores = npz[f"{split}/probabilities"][:, column].astype(float)
    cpp_scores = np.array([r["target_scores"][target] for r in scored], dtype=float)
    gaps = np.abs(python_scores - cpp_scores)
    worst = float(gaps.max())
    over_gate = int((gaps > SCORE_MAX_ABS).sum())
    # The 1e-4 gate is the INT8 near-bitwise gate (DESIGN §10): the quantized graph
    # runs the SAME integer kernels in Python and C++, and quantization clamps any
    # sub-quantum preprocessing difference. It does NOT apply to the FP32 baseline
    # compared across OpenCV versions — there the P1 INTER_LINEAR gap (4.6 apt vs 4.13
    # wheel) propagates linearly to the score, so score_diagnostic reports the gap
    # instead of gating on it. Correctness for M0 rests on the decision and confusion
    # gates below (a real runner bug flips decisions and breaks the matrix), plus
    # p_ort_cpp (C++ ORT == Python ORT on identical tensors) and P1 (the two
    # preprocessings agree to the registered budget).
    if worst > SCORE_MAX_ABS and not score_diagnostic:
        failures.append(
            f"worst score gap {worst:.2e} exceeds the {SCORE_MAX_ABS:.0e} gate "
            f"({over_gate} frames over)"
        )

    python_fire = python_scores >= threshold
    cpp_fire = np.array([bool(r["shutter_trigger"]) for r in scored])
    carved = np.abs(python_scores - threshold) <= DECISION_CARVE_OUT
    disagree = python_fire != cpp_fire
    hard_disagree = disagree & ~carved
    if hard_disagree.any():
        failures.append(
            f"{int(hard_disagree.sum())} decisions differ outside the carve-out"
        )

    def confusion(fire: np.ndarray) -> dict:
        keep = ~carved
        return {
            "true_fire": int((fire & present & keep).sum()),
            "false_fire": int((fire & ~present & keep).sum()),
            "missed": int((~fire & present & keep).sum()),
            "true_quiet": int((~fire & ~present & keep).sum()),
        }

    python_matrix = confusion(python_fire)
    cpp_matrix = confusion(cpp_fire)
    if python_matrix != cpp_matrix:
        failures.append("confusion matrices differ on the non-carved frames")

    return {
        "split": split,
        "frames": len(scored),
        "worst_score_gap": worst,
        "mean_score_gap": float(gaps.mean()),
        "scores_over_1e-4_gate": over_gate,
        "score_treatment": "diagnostic (FP32 baseline; INT8 1e-4 gate N/A)"
        if score_diagnostic
        else "gated at 1e-4 (INT8 near-bitwise)",
        "decisions_differing": int(disagree.sum()),
        "carved_out_frames": [
            {"image_id": python_ids[i], "python_score": float(python_scores[i])}
            for i in np.flatnonzero(carved & disagree)
        ],
        "near_threshold_frames": int(carved.sum()),
        "confusion_python": python_matrix,
        "confusion_cpp": cpp_matrix,
        "cpp_header": {
            key: header[key]
            for key in ("model_sha256", "policy_id", "manifest_sha256", "threads",
                        "onnxruntime_version")
        },
        "passed": not failures,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--cpp-dir", required=True, type=Path,
                        help="holds cpp_<split>.jsonl from run-dataset")
    parser.add_argument("--manifests-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--target", default="bobcat")
    # For a candidate that predates the optimize-candidate layout (M0's C2 evaluate
    # writes predictions.npz but no evaluation.json, and its npz carries no
    # model_sha256), the identity is supplied explicitly. The optimize candidates
    # (M1-M4) still resolve it from evaluation.json + npz, unchanged.
    parser.add_argument("--model-sha256", default="",
                        help="required when the candidate has no evaluation.json")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--label", default="")
    # For the FP32 baseline (M0), the 1e-4 score gate is the INT8 near-bitwise gate and
    # does not apply: C++ (OpenCV 4.6) vs Python (4.13) preprocessing differs by the P1
    # INTER_LINEAR budget, and FP32 propagates it to the score where INT8 would clamp
    # it. With this flag the score gap is reported, not gated; the decision and
    # confusion-matrix gates (the correctness verdict) stay in force.
    parser.add_argument("--score-diagnostic", action="store_true",
                        help="report the score gap instead of gating it at 1e-4 (FP32 baseline)")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    npz = np.load(args.candidate / "predictions.npz", allow_pickle=False)
    npz_has_sha = "model_sha256" in set(npz.keys())
    evaluation_path = args.candidate / "evaluation.json"
    if evaluation_path.exists():
        evaluation = json.loads(evaluation_path.read_text())
        model_sha256 = str(npz["model_sha256"]) if npz_has_sha else str(evaluation["model"]["sha256"])
        if npz_has_sha and model_sha256 != evaluation["model"]["sha256"]:
            raise RuntimeError("candidate directory is inconsistent (npz vs evaluation.json)")
        label = evaluation["label"]
        model_path = evaluation["model"]["path"]
    else:
        if not args.model_sha256:
            raise RuntimeError(
                "the candidate has no evaluation.json; pass --model-sha256 "
                "(and optionally --model-path/--label) to name the model under test"
            )
        model_sha256 = args.model_sha256
        if npz_has_sha and str(npz["model_sha256"]) != model_sha256:
            raise RuntimeError("npz model_sha256 disagrees with --model-sha256")
        label = args.label or args.candidate.name
        model_path = args.model_path

    policy = json.loads(args.policy.read_text())
    if policy["model_sha256"] != model_sha256:
        raise RuntimeError(
            "the policy is not bound to this candidate's ONNX; P4 compares the "
            "deployable pair, not arbitrary combinations"
        )
    (target_entry,) = [t for t in policy["targets"] if t["class"] == args.target]
    class_names = [str(name) for name in npz["class_names"]]

    splits = [
        compare_split(
            split,
            npz,
            args.cpp_dir / f"cpp_{split}.jsonl",
            args.manifests_dir / f"{split}.jsonl",
            model_sha256,
            policy,
            args.target,
            float(target_entry["threshold"]),
            class_names,
            args.score_diagnostic,
        )
        for split in SPLITS
    ]

    passed = all(s["passed"] for s in splits)
    report = {
        "gate": "P4 — C++ dataset parity (DESIGN §10)",
        "candidate_id": label,
        "onnx": {"path": model_path, "sha256": model_sha256},
        "policy_id": policy["policy_id"],
        "threshold": float(target_entry["threshold"]),
        "tolerances": {
            "score_max_abs": SCORE_MAX_ABS,
            "decision_carve_out": DECISION_CARVE_OUT,
            "score_treatment": "diagnostic (FP32 baseline; INT8 1e-4 gate N/A)"
            if args.score_diagnostic
            else "gated at 1e-4 (INT8 near-bitwise)",
        },
        "splits": splits,
        "verdict": {
            "passed": passed,
            "failed": [s["split"] for s in splits if not s["passed"]],
        },
    }
    atomic_write_json(args.output, report)
    for s in splits:
        print(
            f"{s['split']}: {'PASS' if s['passed'] else 'FAIL'} "
            f"({s.get('frames', '?')} frames, worst gap "
            f"{s.get('worst_score_gap', float('nan')):.2e}, "
            f"{s.get('decisions_differing', '?')} decisions differ, "
            f"{len(s.get('carved_out_frames', []))} carved out)"
        )
        for failure in s["failures"]:
            print(f"  FAIL: {failure}")
    print(f"P4 {'PASSED' if passed else 'FAILED'}; wrote {args.output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
