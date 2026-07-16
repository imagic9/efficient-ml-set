#!/usr/bin/env python3
"""C4: export a trained run's selected checkpoint to the deployable FP32 ONNX.

Export is part of correctness, not packaging (DESIGN §10). This tool therefore
refuses to produce an artifact whose lineage it cannot prove:

- the checkpoint's bytes must hash to what the run's own `hashes.json` recorded
  at training time — a `best.pt` overwritten by a later run into the same
  directory exports someone else's weights;
- the checkpoint's epoch must be the history's `best_epoch` (same guard as
  `validate.dump_predictions`);
- when a calibrated policy is given, its `model_sha256` must equal the checkpoint
  hash — this is the C3→C4 chain: the thresholds in that policy were measured on
  these exact weights, and C4's re-bind (after P2 parity) is only meaningful if
  the graph being re-bound to came from them.

The written graph carries its provenance *inside itself* as `metadata_props`
(run id, epoch, checkpoint hash, class order hash, input contract), so an ONNX
file found loose on a Pi can still say what it is. Deliberately absent: wall
clock and git commit. The export must be byte-reproducible from the checkpoint —
same weights, same versions, same bytes — and a timestamp would break that for
nothing; the evidence JSON records the mutable context instead.

The ONNX itself stays in the run directory beside `best.pt` (both gitignored;
binaries are published through GitHub Releases per DESIGN §14). What is
committed is the evidence: `results/parity/<run_id>/export.json` with the
verified description — opset, IO contract, hashes, versions.

Usage:
    python -m wildlife_trigger.export \
        --run results/training/c2/c2_m0_fp32_seed42_20260716T061203Z \
        --policy artifacts/policies/bobcat_v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import onnx
import torch

from .models.export import P0_OPSET, describe, export_onnx, verify_exported_opset
from .models.mobilenet import build_mobilenet_v2, example_input
from .policy import canonical_json
from .runs import BEST_CHECKPOINT, atomic_write_json, resolve_run_id, sha256_file

# Namespaced keys, so a metadata reader can tell our provenance from anything a
# converter or optimizer later adds to the same map.
METADATA_PREFIX = "wildlife_trigger."


def load_run(run_dir: Path) -> dict:
    history = json.loads((run_dir / "history.json").read_text())
    hashes = json.loads((run_dir / "hashes.json").read_text())
    if "checkpoint:best" not in hashes:
        raise RuntimeError(
            f"{run_dir / 'hashes.json'} records no checkpoint:best hash; an export "
            "whose source cannot be verified is not evidence of anything"
        )
    return {
        "history": history,
        "hashes": hashes,
        "run_id": resolve_run_id(run_dir, history["run_name"]),
    }


def verify_checkpoint(run_dir: Path, run: dict) -> tuple[Path, str]:
    """The checkpoint file, proven to be the one the run selected and recorded."""
    checkpoint_path = run_dir / BEST_CHECKPOINT
    measured = sha256_file(checkpoint_path)
    recorded = run["hashes"]["checkpoint:best"]["sha256"]
    if measured != recorded:
        raise RuntimeError(
            f"{checkpoint_path} hashes to {measured[:16]}... but the run recorded "
            f"{recorded[:16]}.... The file was overwritten after training; whatever "
            "it holds now is not the model this run selected."
        )
    return checkpoint_path, measured


def verify_policy_binding(policy_path: Path, checkpoint_sha256: str) -> dict:
    """The C3→C4 chain: the policy's thresholds were measured on these weights."""
    policy = json.loads(policy_path.read_text())
    bound = policy.get("model_sha256", "")
    if bound != checkpoint_sha256:
        raise RuntimeError(
            f"{policy_path} is bound to model {bound[:16]}... but this run's "
            f"checkpoint is {checkpoint_sha256[:16]}.... The policy was calibrated "
            "on different weights; exporting this graph would not entitle anyone "
            "to re-bind that policy to it."
        )
    return policy


def embed_metadata(onnx_path: Path, entries: dict[str, str]) -> None:
    """Write provenance into the graph's metadata_props, deterministically.

    Sorted insertion, and the file is rewritten in place: the hash that gets
    recorded and re-bound must be the hash of the file with its provenance in it.
    """
    model = onnx.load(str(onnx_path), load_external_data=False)
    del model.metadata_props[:]
    for key in sorted(entries):
        model.metadata_props.add(key=METADATA_PREFIX + key, value=str(entries[key]))
    onnx.save(model, str(onnx_path))


def export_run(
    run_dir: Path,
    output: Path | None = None,
    policy_path: Path | None = None,
    evidence_root: Path = Path("results/parity"),
) -> dict:
    run = load_run(run_dir)
    history = run["history"]
    config = history["config"]
    class_names = history["class_names"]

    checkpoint_path, checkpoint_sha256 = verify_checkpoint(run_dir, run)
    policy_check = None
    if policy_path is not None:
        verify_policy_binding(policy_path, checkpoint_sha256)
        policy_check = {
            "policy": str(policy_path),
            "model_sha256_matches_checkpoint": True,
        }

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "epoch" in checkpoint and checkpoint["epoch"] != history["best_epoch"]:
        raise RuntimeError(
            f"{checkpoint_path} holds epoch {checkpoint['epoch']} but the history's "
            f"best_epoch is {history['best_epoch']}. This checkpoint is not this "
            "run's selected model."
        )
    model = build_mobilenet_v2(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model"])

    # The run's own geometry, never a constant: pairing one run's weights with
    # another run's input shape is the exact failure dump_predictions guards
    # against, one artifact later.
    shape = (1, 3, config["height"], config["width"])
    onnx_path = output or run_dir / f"{history['run_name']}.onnx"
    export_onnx(model, onnx_path, example_input(shape), opset=P0_OPSET, dynamo=False)

    class_names_sha256 = hashlib.sha256(
        canonical_json({"class_names": class_names}).encode()
    ).hexdigest()
    embed_metadata(
        onnx_path,
        {
            "run_id": run["run_id"],
            "best_epoch": history["best_epoch"],
            "checkpoint_sha256": checkpoint_sha256,
            "class_names_sha256": class_names_sha256,
            "input_contract": (
                f"1x3x{config['height']}x{config['width']} NCHW RGB float32 "
                "imagenet-normalized (DESIGN 5.5)"
            ),
        },
    )
    # Metadata rewrote the file; verify and describe what is actually on disk.
    verify_exported_opset(onnx_path, expected=P0_OPSET)
    description = describe(onnx_path)

    evidence = {
        "tool": "wildlife_trigger.export",
        "run_id": run["run_id"],
        "run_name": history["run_name"],
        "best_epoch": history["best_epoch"],
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha256},
        "policy_check": policy_check,
        "onnx": description,
        "exporter": "torchscript",
        "opset_contract": P0_OPSET,
        "input_shape_nchw": list(shape),
        "class_names_sha256": class_names_sha256,
        "versions": {
            "torch": torch.__version__,
            "onnx": onnx.__version__,
        },
    }
    evidence_path = evidence_root / run["run_id"] / "export.json"
    atomic_write_json(evidence_path, evidence)
    return {**evidence, "evidence_path": str(evidence_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="a training run directory")
    parser.add_argument(
        "--output", type=Path, help="ONNX destination (default: <run>/<run_name>.onnx)"
    )
    parser.add_argument(
        "--policy",
        type=Path,
        help="calibrated policy whose model_sha256 must match the checkpoint "
        "(the C3->C4 chain); omit for runs that have no policy yet",
    )
    parser.add_argument("--evidence-root", type=Path, default=Path("results/parity"))
    args = parser.parse_args()

    result = export_run(
        args.run,
        output=args.output,
        policy_path=args.policy,
        evidence_root=args.evidence_root,
    )
    print(f"run: {result['run_id']} (best epoch {result['best_epoch']})")
    print(f"onnx: {result['onnx']['path']}")
    print(f"  sha256: {result['onnx']['sha256']}")
    print(f"  opset: {result['onnx']['opset_import']}  nodes: {result['onnx']['node_count']}")
    print(f"  inputs: {result['onnx']['inputs']}")
    print(f"  outputs: {result['onnx']['outputs']}")
    if result["policy_check"]:
        print("policy binding: model_sha256 matches this checkpoint (C3->C4 chain holds)")
    print(f"wrote evidence: {result['evidence_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
