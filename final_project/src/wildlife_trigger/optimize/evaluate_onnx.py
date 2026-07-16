#!/usr/bin/env python3
"""Score an ONNX artifact over the validation splits, through deployment ORT.

This is `validate.dump_predictions` for the D-phase: the same npz schema, the
same two reachable splits, but the scores come from the **shipped artifact's own
outputs** — ONNX Runtime, CPU EP, batch 1 — rather than from the PyTorch
checkpoint. For a quantized candidate that distinction is the whole point: its
INT8 arithmetic *is* the model, and a policy calibrated on FP32 torch scores
would describe a device that does not ship. This is the §6.3 amendment
(issue #30) taken to its conclusion: not merely TF32-off, but the deployment
runtime itself.

Batch 1 deliberately: the exported graphs carry a static batch-1 shape
(`models.export`), and the Pi infers one frame at a time. There is no batched
fast path to accidentally measure.

Two files per candidate directory:

- `predictions.npz` — per-frame probabilities/present/seq_ids/image_ids per
  split, `class_names`, and the scoring regime (ORT version, provider, threads,
  model sha) so a reader can tell *which* arithmetic produced these numbers;
- `evaluation.json` — bobcat metrics at the fixed 0.5 yardstick per domain,
  average precision, support-aware macro F1, and the frozen §7.2 selection
  score, each traceable to the npz by hash.

Only the validation splits are reachable, by the same construction as
`dump_predictions`: DESIGN §5.4 seals the test sets, and a flag that could
reach them is a leak waiting for a tired evening.

Usage (gx10):
    python -m wildlife_trigger.optimize.evaluate_onnx \
        --model results/optimize/m1_ptq/minmax/model.onnx \
        --label d1_m1_ptq_minmax \
        --output-dir results/optimize/m1_ptq/minmax
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from .. import metrics
from ..data.dataset import WildlifeDataset, load_class_names
from ..data.preprocess import PreprocessConfig
from ..runs import atomic_write_json, sha256_file

VALIDATION_SPLITS = ("cis_val_clean", "trans_val")

# The §7.2 yardstick every recorded selection in this project reads.
YARDSTICK = 0.5


def model_geometry(path: Path) -> tuple[int, int]:
    """(width, height) from the graph itself — the artifact states its contract.

    Passing geometry by hand is how a 224 cache ends up scored through a 256
    model with silent letterbox mismatch; the input tensor's static shape is
    already the truth, so read it.
    """
    model = onnx.load(str(path), load_external_data=False)
    initializers = {init.name for init in model.graph.initializer}
    (spec,) = [i for i in model.graph.input if i.name not in initializers]
    dims = [d.dim_value for d in spec.type.tensor_type.shape.dim]
    if len(dims) != 4 or dims[0] != 1 or dims[1] != 3:
        raise ValueError(
            f"{path} has input shape {dims}, expected static (1, 3, H, W); "
            "this evaluator only speaks the project's export contract"
        )
    return dims[3], dims[2]  # (width, height)


def build_datasets(
    class_names: list[str],
    config: PreprocessConfig,
    manifests_dir: Path,
    images_dir: Path,
    cache_dir: Path,
) -> dict[str, WildlifeDataset]:
    return {
        split: WildlifeDataset(
            manifests_dir / f"{split}.jsonl",
            class_names,
            config,
            images_dir,
            cache_root=cache_dir,
            train=False,  # deterministic; augmentation never touches evaluation
        )
        for split in VALIDATION_SPLITS
    }


def score_split(
    session: ort.InferenceSession, dataset: WildlifeDataset, input_name: str
) -> dict[str, np.ndarray]:
    """Batch-1 inference in manifest order — the deployment loop, verbatim."""
    probabilities, present, seq_ids, image_ids = [], [], [], []
    for index in range(len(dataset)):
        item = dataset[index]
        (logits,) = session.run(None, {input_name: item["image"].numpy()[None, ...]})
        logits = logits[0].astype(np.float64)
        shifted = np.exp(logits - logits.max())
        probabilities.append((shifted / shifted.sum()).astype(np.float32))
        present.append(item["present"].numpy())
        record = dataset.records[index]
        seq_ids.append(record["seq_id"])
        image_ids.append(record["image_id"])
    return {
        "probabilities": np.stack(probabilities),
        "present": np.stack(present),
        "seq_ids": np.array(seq_ids),
        "image_ids": np.array(image_ids),
    }


def domain_summary(
    probabilities: np.ndarray,
    present: np.ndarray,
    seq_ids: list[str],
    class_names: list[str],
    target: str = "bobcat",
) -> dict:
    column = class_names.index(target)
    scores, positives = probabilities[:, column], present[:, column]
    summary = metrics.target_presence_metrics(scores, positives, seq_ids, YARDSTICK)
    summary["average_precision"] = metrics.average_precision(scores, positives)
    classes = metrics.per_class_metrics(probabilities, present, class_names, seq_ids)
    return {
        "target": summary,
        "support_aware_macro_f1": classes["support_aware_macro_f1"],
        "top1_accuracy": classes["top1_accuracy"],
        "frames": int(len(scores)),
        "sequences": len(set(seq_ids)),
    }


def evaluate(
    model_path: Path,
    label: str,
    output_dir: Path,
    manifests_dir: Path = Path("data/manifests"),
    images_dir: Path = Path("data/raw/extracted/eccv_18_all_images_sm"),
    cache_dir: Path = Path("data/cache"),
    classes_config: Path = Path("configs/data/classes.yaml"),
    intra_op_threads: int = 0,
    target: str = "bobcat",
) -> dict:
    class_names = load_class_names(classes_config)
    width, height = model_geometry(model_path)
    datasets = build_datasets(
        class_names,
        PreprocessConfig(width=width, height=height),
        manifests_dir,
        images_dir,
        cache_dir,
    )

    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_op_threads
    session = ort.InferenceSession(
        str(model_path), options, providers=["CPUExecutionProvider"]
    )
    (input_spec,) = session.get_inputs()

    model_sha256 = sha256_file(model_path)
    payload: dict[str, np.ndarray] = {
        "class_names": np.array(class_names),
        "model_sha256": np.array(model_sha256),
        "ort_version": np.array(ort.__version__),
        "provider": np.array("CPUExecutionProvider"),
        "intra_op_threads": np.array(intra_op_threads),
        "batch_size": np.array(1),
    }
    summaries: dict[str, dict] = {}
    manifest_hashes: dict[str, str] = {}
    for split, dataset in datasets.items():
        result = score_split(session, dataset, input_spec.name)
        for key, value in result.items():
            payload[f"{split}/{key}"] = value
        summaries[split] = domain_summary(
            result["probabilities"],
            result["present"],
            result["seq_ids"].tolist(),
            class_names,
            target,
        )
        manifest_hashes[split] = sha256_file(dataset.manifest)
        print(
            f"{split}: {summaries[split]['frames']} frames, "
            f"{summaries[split]['sequences']} sequences, "
            f"{target} F2@{YARDSTICK} {summaries[split]['target']['frame_f2']:.4f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "predictions.npz"
    np.savez_compressed(npz_path, **payload)

    selection = metrics.selection_score(
        summaries["cis_val_clean"]["target"],
        summaries["trans_val"]["target"],
        (
            summaries["cis_val_clean"]["support_aware_macro_f1"]
            + summaries["trans_val"]["support_aware_macro_f1"]
        )
        / 2,
    )

    record = {
        "tool": "wildlife_trigger.optimize.evaluate_onnx",
        "label": label,
        "target": target,
        "yardstick_threshold": YARDSTICK,
        "model": {
            "path": str(model_path),
            "sha256": model_sha256,
            "bytes": model_path.stat().st_size,
        },
        "regime": {
            "runtime": f"onnxruntime {ort.__version__}",
            "provider": "CPUExecutionProvider",
            "intra_op_threads": intra_op_threads,
            "batch_size": 1,
            "input": f"{width}x{height}",
        },
        "manifests": manifest_hashes,
        "class_names": class_names,
        "domains": summaries,
        "selection_score": selection,
        "predictions_npz": {"path": str(npz_path), "sha256": sha256_file(npz_path)},
    }
    atomic_write_json(output_dir / "evaluation.json", record)
    print(
        f"selection primary ({selection['primary_metric']}): "
        f"{selection['primary']:.4f}"
    )
    print(f"wrote {npz_path}")
    print(f"wrote {output_dir / 'evaluation.json'}")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--label", required=True, help="e.g. d1_m1_ptq_minmax")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifests-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument(
        "--images-dir", type=Path, default=Path("data/raw/extracted/eccv_18_all_images_sm")
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument(
        "--classes-config", type=Path, default=Path("configs/data/classes.yaml")
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="intra-op threads; 0 lets ORT choose. Correctness work only — "
        "DESIGN §12.4 forbids reading gx10 latency as a ranking.",
    )
    args = parser.parse_args()

    evaluate(
        args.model,
        args.label,
        args.output_dir,
        manifests_dir=args.manifests_dir,
        images_dir=args.images_dir,
        cache_dir=args.cache_dir,
        classes_config=args.classes_config,
        intra_op_threads=args.threads,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
