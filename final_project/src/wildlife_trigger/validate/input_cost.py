#!/usr/bin/env python3
"""What each candidate input geometry costs, and what it actually delivers.

PLAN C1a decides the Core input on three things, not one: the validation metrics, the
**real-pixel utilization**, and the **MACs**. The metrics come from the runs. The other
two are properties of the geometry against this dataset, and they are what stops the
decision from resting on a single noisy score.

The distinction the utilization number exists to make: a letterbox tensor is part frame
and part grey bar, and only the frame carries signal. Two geometries with nearly
identical tensor areas can therefore differ enormously in how much animal reaches the
network. 256x192 and 224x224 are almost exactly the same tensor (49,152 against 50,176
pixels, a 2% difference) — but CCT's dominant frame is 1024x747, a 1.37 aspect ratio
that fits 256x192 almost exactly and leaves a quarter of a square tensor as grey bars.

MACs are counted, not estimated from the pixel ratio. Depthwise-separable stacks do not
scale perfectly with input area — stride and rounding move the feature-map sizes — and
the classifier head does not scale with it at all.

Usage:
    python -m wildlife_trigger.validate.input_cost \
        --shapes 256x192 224x224 --manifests data/manifests
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ..data.preprocess import PreprocessConfig, letterbox_geometry

MANIFEST_SPLITS = ("train", "cis_val_clean", "trans_val")


def utilisation_over_manifests(width: int, height: int, manifests_dir: Path) -> dict:
    """Mean real-pixel utilization over the actual data, not over the nominal frame.

    Weighted by how often each source geometry occurs: CCT-20 is dominated by 1024x747
    but is not only 1024x747, and quoting the utilization of the dominant frame as the
    dataset's would be a claim about one geometry dressed as a claim about the corpus.
    """
    config = PreprocessConfig(width=width, height=height)
    per_split = {}
    for split in MANIFEST_SPLITS:
        path = manifests_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        geometries = Counter()
        for line in path.read_text().splitlines():
            record = json.loads(line)
            geometries[(record["observed_width"], record["observed_height"])] += 1

        total = sum(geometries.values())
        real_px = 0
        for (source_width, source_height), count in geometries.items():
            resized_width, resized_height, _ = letterbox_geometry(
                source_width, source_height, config
            )
            real_px += resized_width * resized_height * count

        tensor_px = width * height
        per_split[split] = {
            "images": total,
            "distinct_geometries": len(geometries),
            "mean_real_pixels": round(real_px / total, 1),
            "tensor_pixels": tensor_px,
            "mean_utilisation": round(real_px / total / tensor_px, 4),
        }
    return per_split


def macs_at(width: int, height: int, num_classes: int) -> int:
    """Multiply-accumulates for one forward pass at this geometry.

    torch's own counter reports FLOPs by the convention that one MAC is two — halved
    here so the number means what DESIGN and the literature mean by MACs when they
    quote MobileNetV2 at 300M.
    """
    from ..models.mobilenet import build_mobilenet_v2

    model = build_mobilenet_v2(num_classes=num_classes, pretrained=False)
    return macs_of_model(model, width, height)


def macs_of_model(model, width: int, height: int) -> int:
    """The same ladder MAC convention, for an arbitrary module.

    Split out of `macs_at` for D4: a pruned candidate's row must count MACs in
    the convention every other row uses, and `macs_at` can only build the
    unpruned architecture. Same counter, same halving, same meaning.
    """
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    was_training = model.training
    model.eval()
    try:
        inputs = torch.zeros(1, 3, height, width)
        with torch.inference_mode(), FlopCounterMode(display=False) as counter:
            model(inputs)
        return counter.get_total_flops() // 2
    finally:
        model.train(was_training)


def parse_shape(text: str) -> tuple[int, int]:
    width, height = text.lower().split("x")
    return int(width), int(height)


def report(shapes: list[tuple[int, int]], manifests_dir: Path, num_classes: int) -> dict:
    rows = []
    for width, height in shapes:
        utilisation = utilisation_over_manifests(width, height, manifests_dir)
        rows.append(
            {
                "input": f"{width}x{height}",
                "width": width,
                "height": height,
                "tensor_pixels": width * height,
                "macs": macs_at(width, height, num_classes),
                "utilisation": utilisation,
            }
        )
    return {"num_classes": num_classes, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", default=["256x192", "224x224"])
    parser.add_argument("--manifests", type=Path, default=Path("data/manifests"))
    parser.add_argument("--num-classes", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = report([parse_shape(s) for s in args.shapes], args.manifests, args.num_classes)
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
