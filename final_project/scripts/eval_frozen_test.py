#!/usr/bin/env python3
"""F4 frozen full-test evaluation — the one-time test-set opening (PLAN F4, DESIGN §5.4).

After F3 froze everything, this computes the operating-point metrics on the SEALED test
splits (cis_test, trans_test) from the frozen C++/ORT run-dataset predictions, at the FROZEN
policy threshold — never selecting a new one. Reuses the canonical loaders/metrics verbatim:
validate.p4_dataset_parity.load_cpp_jsonl for the C++ output and metrics.target_presence_metrics
for the numbers. Nothing here reads test labels to *choose* anything (the threshold came from
validation, F3); it only scores the frozen decision on held-out data. Real numbers, generated.

Usage:
  python3 scripts/eval_frozen_test.py --model-id M2 \
      --policy results/e7/bundle/policies/M2.json \
      --manifests-dir data/manifests \
      --pred-cis results/f4/gx10_test_M2_cis_test.jsonl \
      --pred-trans results/f4/gx10_test_M2_trans_test.jsonl \
      --output results/f4/frozen_test_M2.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from wildlife_trigger import metrics
from wildlife_trigger.validate.p4_dataset_parity import load_cpp_jsonl


def manifest_index(path: Path) -> dict:
    idx = {}
    with path.open() as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                idx[r["image_id"]] = (r["labels"], r["seq_id"])
    return idx


def evaluate_split(pred_path: Path, manifest_path: Path, target: str, threshold: float) -> dict:
    header, rows, footer = load_cpp_jsonl(pred_path)
    scored = [r for r in rows if not r.get("skipped")]
    skipped = [r for r in rows if r.get("skipped")]
    idx = manifest_index(manifest_path)

    scores, present, seq_ids = [], [], []
    for r in scored:
        iid = r["image_id"]
        labels, seq_id = idx[iid]
        scores.append(float(r["target_scores"][target]))
        present.append(1 if target in labels else 0)
        seq_ids.append(seq_id)

    m = metrics.target_presence_metrics(
        np.asarray(scores), np.asarray(present), seq_ids, threshold
    )
    m["frames_scored"] = len(scored)
    m["frames_skipped"] = len(skipped)
    m["header_model_sha256"] = header.get("model_sha256")
    m["header_policy_id"] = header.get("policy_id")
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--policy", required=True, type=Path)
    ap.add_argument("--manifests-dir", required=True, type=Path)
    ap.add_argument("--pred-cis", required=True, type=Path)
    ap.add_argument("--pred-trans", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    policy = json.loads(args.policy.read_text())
    tgt = policy["targets"][0]
    target, threshold = tgt["class"], float(tgt["threshold"])

    result = {
        "kind": "f4_frozen_test_metrics",
        "schema_version": 1,
        "design": "PLAN F4 / DESIGN §5.4, §6.3",
        "model_id": args.model_id,
        "target": target,
        "frozen_threshold": threshold,
        "policy_id": policy["policy_id"],
        "note": "One-time test-set opening AFTER the F3 freeze. Threshold is the frozen "
                "validation-calibrated value; test labels are only scored, never used to select.",
        "per_domain": {
            "cis_test": evaluate_split(args.pred_cis, args.manifests_dir / "cis_test.jsonl", target, threshold),
            "trans_test": evaluate_split(args.pred_trans, args.manifests_dir / "trans_test.jsonl", target, threshold),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    for dom, m in result["per_domain"].items():
        print(f"  {args.model_id} {dom:11s} F2={m['frame_f2']:.4f} "
              f"seqRecall={m['sequence_balanced_recall']:.4f} "
              f"eventCapture={m['event_capture_rate']:.4f} "
              f"falseFire={m['false_fire_rate']:.4f} "
              f"(pos {m['positive_frames']}/{m['frames_scored']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
