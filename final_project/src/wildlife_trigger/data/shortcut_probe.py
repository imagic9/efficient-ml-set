#!/usr/bin/env python3
"""B2 — the shortcut probe: can anything tell the supplement from CCT-20?

DESIGN §5.2 requires this *before* training. A small binary classifier is trained to
separate supplement frames from CCT-20 `_sm` frames. Near-chance means the
resolution/encoding confound is closed. High accuracy means it is live, and training
must not proceed on the assumption that the supplement is clean.

## Why a probe rather than an argument

The downsize in `fetch_supplement` is *supposed* to have closed the confound. But
"supposed to" is the same standing that "BGR is RGB" had before someone measured it. The
supplement went through a different pipeline from CCT-20 — different original files,
different resampler, different JPEG encoder, different quality setting — and any residue
of that pipeline is a feature perfectly correlated with the `empty` label. So the
question is put to a classifier rather than to a reviewer.

## What the probe deliberately does not see

It is trained on the **decoded, resized tensor** the real model consumes, not on the
file. Feeding it file size or JPEG headers would let it win on metadata the network
never sees, and it would report a confound that does not exist for the actual task.

## Reading the result

The probe balances the two pools, so chance is 0.50. Interpreting the number:

  - **~0.50-0.60**: the confound is closed as far as this probe can tell.
  - **0.60-0.75**: something is separable. Report it; consider whether it is the
    unavoidable background difference (rule 3) rather than encoding.
  - **>0.75**: the confound is live. Fix the downsizing procedure. Do not train.

A probe that *can* separate the pools is not automatically a bug in the downsize: rule 3
makes the supplement location-disjoint, so backgrounds genuinely differ and that is
unavoidable (DESIGN §5.2). The probe cannot distinguish "different camera" from
"different encoder" — which is why a positive result is a signal to investigate, not a
verdict on which cause.

Usage:
    python -m wildlife_trigger.data.shortcut_probe --supplement S.jsonl \
        --cct20 data/manifests/train.jsonl --images-dir ... --supplement-dir ...
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Small enough to train in a minute, big enough to find an encoding signature. This is
# a detector, not a model we ship.
PROBE_SIZE = (96, 96)
PROBE_EPOCHS = 4
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# Where "near chance" stops being near chance. Above this the probe blocks training, per
# DESIGN §5.2.
BLOCKING_ACCURACY = 0.75
ATTENTION_ACCURACY = 0.60


def load_tensor(path: Path) -> np.ndarray | None:
    """Decode and resize exactly as far as the probe is allowed to see."""
    try:
        with Image.open(path) as image:
            array = np.asarray(
                image.convert("RGB").resize(PROBE_SIZE, Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
    except Exception:
        return None
    return (array / 255.0).transpose(2, 0, 1)


class Probe(nn.Module):
    """A deliberately small CNN. If *this* can separate the pools, the gap is gross."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def gather(paths: list[Path], label: int, limit: int) -> tuple[list, list]:
    tensors, labels = [], []
    for path in paths[:limit]:
        tensor = load_tensor(path)
        if tensor is not None:
            tensors.append(tensor)
            labels.append(label)
    return tensors, labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supplement", required=True, type=Path)
    parser.add_argument("--supplement-dir", required=True, type=Path)
    parser.add_argument("--cct20", required=True, type=Path)
    parser.add_argument("--cct20-dir", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=2000, help="Per pool.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    supplement = [json.loads(l) for l in args.supplement.read_text().splitlines()]
    cct20 = [json.loads(l) for l in args.cct20.read_text().splitlines()]
    rng.shuffle(supplement)
    rng.shuffle(cct20)

    print(f"loading up to {args.samples} frames per pool ...")
    supplement_x, supplement_y = gather(
        [args.supplement_dir / r["file_name"] for r in supplement], 1, args.samples
    )
    cct20_x, cct20_y = gather(
        [args.cct20_dir / r["file_name"] for r in cct20], 0, args.samples
    )

    # Balance the pools so chance is exactly 0.50 and accuracy needs no prior.
    size = min(len(supplement_x), len(cct20_x))
    supplement_x, supplement_y = supplement_x[:size], supplement_y[:size]
    cct20_x, cct20_y = cct20_x[:size], cct20_y[:size]
    print(f"balanced pools: {size} supplement, {size} CCT-20")

    x = torch.from_numpy(np.stack(supplement_x + cct20_x))
    y = torch.tensor(supplement_y + cct20_y)

    indices = torch.randperm(len(y), generator=torch.Generator().manual_seed(args.seed))
    x, y = x[indices], y[indices]
    split = int(0.8 * len(y))
    train_x, train_y = x[:split], y[:split]
    test_x, test_y = x[split:], y[split:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Probe().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(PROBE_EPOCHS):
        total = 0.0
        for start in range(0, len(train_y), BATCH_SIZE):
            batch_x = train_x[start : start + BATCH_SIZE].to(device)
            batch_y = train_y[start : start + BATCH_SIZE].to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimiser.step()
            total += float(loss) * len(batch_y)
        print(f"  epoch {epoch + 1}/{PROBE_EPOCHS}  loss {total / len(train_y):.4f}")

    model.eval()
    correct = 0
    with torch.inference_mode():
        for start in range(0, len(test_y), BATCH_SIZE):
            batch_x = test_x[start : start + BATCH_SIZE].to(device)
            batch_y = test_y[start : start + BATCH_SIZE].to(device)
            correct += int((model(batch_x).argmax(1) == batch_y).sum())
    accuracy = correct / len(test_y)

    if accuracy >= BLOCKING_ACCURACY:
        verdict = "LIVE — blocks training"
    elif accuracy >= ATTENTION_ACCURACY:
        verdict = "separable — investigate before trusting the ablation"
    else:
        verdict = "closed as far as this probe can tell"

    report = {
        "task": "B2-shortcut-probe",
        "held_out_accuracy": round(accuracy, 4),
        "chance": 0.5,
        "samples_per_pool": size,
        "held_out_samples": len(test_y),
        "verdict": verdict,
        "blocks_training": accuracy >= BLOCKING_ACCURACY,
        "thresholds": {
            "attention": ATTENTION_ACCURACY,
            "blocking": BLOCKING_ACCURACY,
        },
        "probe_sees": (
            "the decoded, resized RGB tensor only — never file size or JPEG headers, "
            "which the real model never sees either"
        ),
        "caveat": (
            "Rule 3 makes the supplement location-disjoint, so backgrounds genuinely "
            "differ and that confound is unavoidable (DESIGN §5.2). This probe cannot "
            "tell 'different camera' from 'different encoder'; a positive result says "
            "investigate, not which cause."
        ),
    }

    print(f"\nheld-out accuracy: {accuracy:.4f}  (chance 0.50)")
    print(f"verdict: {verdict}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.report}")

    return 1 if report["blocks_training"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
