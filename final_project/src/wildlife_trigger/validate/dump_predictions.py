#!/usr/bin/env python3
"""Re-evaluate a saved checkpoint and persist its per-frame validation predictions.

`history.json` records aggregated metrics per epoch. Two things need the per-frame
numbers underneath them, and neither can be answered from an aggregate:

- **C1a's input decision.** PLAN C1a says to prefer 256x192 when the arms are
  *statistically tied*, and a point estimate cannot say whether two scores are tied.
  Deciding that requires resampling the validation set, which requires per-frame scores.
- **C3's calibration.** DESIGN §6.3 picks the operating threshold from validation
  scores, and PLAN C2 requires the selected run to persist its validation predictions
  rather than recompute them from a checkpoint whose data pipeline has since moved on.

The run's own `history.json` supplies the config, so the datasets are rebuilt exactly as
that run saw them — input geometry, class list and the 15-output arm's dropped `empty`
included. Passing the shape by hand would eventually pair one arm's checkpoint with
another arm's preprocessing and produce a number that looks fine.

**`seq_id` travels with every frame.** The resampling unit is the sequence, not the
frame: CCT frames arrive in bursts from one camera seconds apart, so frames within a
sequence are near-duplicates. Bootstrapping frames would treat 315 bobcat sequences as
937 independent observations and report an interval far narrower than the data supports.

**Scores are computed in the deployment regime** (DESIGN §6.3 amendment 2026-07-16,
issue #30). torch 2.11 defaults cuDNN convolutions to TF32 on this GPU, which is how
the original seed-42 npz came to sit up to 7.25e-3 away from the exported ONNX exactly
in the near-threshold band a calibration searches. TF32 is disabled before any batch is
scored, and the regime is recorded inside the npz so P2's reproduction check can verify
the file under the arithmetic it was actually written with.

Usage:
    python -m wildlife_trigger.validate.dump_predictions \
        --run results/ablations/c1a_empty5k_16out_256x192 \
        --output results/ablations/c1a_empty5k_16out_256x192/predictions.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import WildlifeDataset
from ..data.preprocess import PreprocessConfig
from ..models.mobilenet import build_mobilenet_v2

VALIDATION_SPLITS = ("cis_val_clean", "trans_val")


def enforce_deployment_regime() -> None:
    """Score with the arithmetic that ships: true FP32, no TF32 anywhere.

    The deployed device (ONNX Runtime CPU on the Pi) computes FP32. A calibration
    that reads TF32-batched scores searches thresholds the device will never see —
    issue #30 measured that gap at up to 7.25e-3 near the threshold. Registered as
    the DESIGN §6.3 amendment of 2026-07-16.
    """
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def load_run(run_dir: Path) -> dict:
    return json.loads((run_dir / "history.json").read_text())


def build_validation_loaders(config: dict, class_names: list[str]) -> dict[str, DataLoader]:
    """Rebuild this run's validation loaders from its recorded config.

    Only the validation splits are reachable from here. DESIGN §5.4 seals cis-test and
    trans-test, and a dump tool that could be pointed at them by a flag is a leak waiting
    for a tired evening.
    """
    preprocess = PreprocessConfig(width=config["width"], height=config["height"])
    manifests = Path(config["manifests_dir"])
    return {
        name: DataLoader(
            WildlifeDataset(
                manifests / f"{name}.jsonl",
                class_names,
                preprocess,
                Path(config["images_dir"]),
                cache_root=Path(config["cache_dir"]),
                train=False,  # no augmentation: this must be reproducible
            ),
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["workers"],
            pin_memory=True,
        )
        for name in VALIDATION_SPLITS
    }


@torch.inference_mode()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """inference_mode, not no_grad: no autograd graph is built at all."""
    model.eval()
    probabilities, present, seq_ids, image_ids = [], [], [], []
    for batch in loader:
        logits = model(batch["image"].to(device, non_blocking=True))
        probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        present.append(batch["present"].numpy())
        for i in batch["index"].tolist():
            record = loader.dataset.records[i]
            seq_ids.append(record["seq_id"])
            image_ids.append(record["image_id"])
    return {
        "probabilities": np.concatenate(probabilities),
        "present": np.concatenate(present),
        "seq_ids": np.array(seq_ids),
        "image_ids": np.array(image_ids),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="a run directory")
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run = load_run(args.run)
    config = run["config"]
    class_names = run["class_names"]

    enforce_deployment_regime()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_mobilenet_v2(num_classes=len(class_names), pretrained=False).to(device)

    # weights_only=False: torch 2.6 flipped this default to guard against unpickling a
    # checkpoint from a stranger. This one is our own train.py's output from our own
    # box, and it deliberately carries more than tensors — the run's score dict holds
    # numpy scalars, which the restricted unpickler refuses. The alternative, allowlisting
    # numpy's scalar constructor, buys nothing here and breaks when numpy moves it.
    checkpoint = torch.load(args.run / args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    # The checkpoint must be the one the run selected. A `best.pt` whose epoch disagrees
    # with `best_epoch` means the file was overwritten by a later run into the same
    # directory, and every number derived from it would belong to a different model.
    if "epoch" in checkpoint and checkpoint["epoch"] != run["best_epoch"]:
        raise RuntimeError(
            f"{args.checkpoint} holds epoch {checkpoint['epoch']} but the history's "
            f"best_epoch is {run['best_epoch']}. This checkpoint is not this run's "
            "selected model."
        )

    loaders = build_validation_loaders(config, class_names)
    payload = {
        "run_name": run["run_name"],
        "class_names": np.array(class_names),
        "best_epoch": run["best_epoch"],
        "width": config["width"],
        "height": config["height"],
        # The regime this file was scored under. P2's reproduction check reads it;
        # legacy npz files predate the key and ran under torch's default (TF32 on).
        "cudnn_tf32": np.array(False),
    }
    for name, loader in loaders.items():
        result = predict(model, loader, device)
        for key, value in result.items():
            payload[f"{name}/{key}"] = value
        print(f"{name}: {len(result['probabilities'])} frames, "
              f"{len(set(result['seq_ids'].tolist()))} sequences")

    output = args.output or args.run / "predictions.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
