#!/usr/bin/env python3
"""QAT candidate 1 — QDQ fake-quant inserted directly, exported with torch.onnx.

DESIGN §8.2 deliberately does not pre-select a QAT library: the tool is an *output*
of parity gate P0, not an input to it. This module implements the first candidate
on that list — fake-quant modules we place ourselves, exported with
`torch.onnx.export` — chosen first because the export semantics are then ours
rather than a library's. If it yields a QDQ graph ORT executes as integer, P0 stops
here and `pytorch-quantization` / `torchao` are never introduced.

Three decisions are load-bearing; each is the kind of thing that silently produces
a float graph carrying rounded weights instead of a quantized one:

1. **BatchNorm is folded into Conv before QAT, not after.** Fake-quantizing a
   weight and *then* folding BN rescales that weight by gamma/sigma, moving it off
   the quantization grid the scales describe — the exported model's QDQ scales
   would be wrong in a way nothing downstream detects. Folding first makes the
   weight that is quantized the weight that is deployed. DESIGN §8.2 already plans
   to freeze BN statistics after stabilization; folding is that limit case, and it
   means M2 must fine-tune in FP32 briefly *before* this stage rather than during.
2. **Weights are per-channel symmetric, activations per-tensor affine.** That is
   what ORT's S8S8 QDQ representation consumes (§8.1). Per-channel symmetric pins
   weight zero-points at 0, which is what the ARM64 signed dot-product kernels
   want.
3. **Observers calibrate first, then freeze; only then does the STE train.**
   Training against a scale that is still moving optimises the weights against a
   grid that no longer exists by the time it is exported.

Usage (spike only):
    python -m wildlife_trigger.optimize.qat --output m2_qat.onnx --steps 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.ao.quantization import FakeQuantize
from torch.ao.quantization.observer import (
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)
from torch.nn.utils.fusion import fuse_conv_bn_eval

from wildlife_trigger.models.export import P0_OPSET, export_onnx
from wildlife_trigger.models.mobilenet import build_mobilenet_v2, example_input

# INT8 signed range. Not a free parameter: it is the S8 half of the S8S8 scheme
# DESIGN §8.1 fixed, and torch's ONNX symbolic only lowers (-128,127) or (0,255)
# to QuantizeLinear/DequantizeLinear.
QMIN, QMAX = -128, 127


def activation_fake_quant() -> FakeQuantize:
    """Per-tensor affine activation fake-quant, matching ORT S8S8 activations."""
    return FakeQuantize(
        observer=MovingAverageMinMaxObserver,
        quant_min=QMIN,
        quant_max=QMAX,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_affine,
    )


def weight_fake_quant(channels: int) -> FakeQuantize:
    """Per-channel symmetric weight fake-quant over output channels (axis 0)."""
    return FakeQuantize(
        observer=MovingAveragePerChannelMinMaxObserver,
        quant_min=QMIN,
        quant_max=QMAX,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,
    )


def fold_conv_bn(model: nn.Module) -> int:
    """Fold every adjacent Conv2d->BatchNorm2d pair, in place. Returns the count.

    Walks containers and rewrites the BatchNorm to Identity rather than deleting
    it, so positional indexing into `nn.Sequential` — which MobileNetV2 relies on
    throughout `features` — keeps working.

    The model must be in eval mode: `fuse_conv_bn_eval` folds the *running*
    statistics, which is only the correct arithmetic when BN is not updating them.
    """
    if model.training:
        raise RuntimeError(
            "fold_conv_bn requires eval mode: it folds BatchNorm running "
            "statistics into the convolution weights, which is only equivalent to "
            "the original network when BN has stopped updating them. Call "
            "model.eval() first."
        )

    folded = 0
    for module in model.modules():
        children = list(module.named_children())
        for (conv_name, conv), (bn_name, bn) in zip(children, children[1:]):
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                setattr(module, conv_name, fuse_conv_bn_eval(conv, bn))
                setattr(module, bn_name, nn.Identity())
                folded += 1
    return folded


class QuantizedConv2d(nn.Module):
    """A Conv2d with QDQ on its input activation and on its weight.

    Wrapping rather than subclassing keeps the original Conv2d — and therefore its
    folded weight — intact and inspectable.

    The bias is deliberately not fake-quantized. ORT derives the bias scale from
    input_scale * weight_scale and stores bias as INT32; a fake-quant here would
    impose an INT8 grid the runtime never uses, degrading accuracy to model
    something that does not happen.
    """

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.conv = conv
        self.input_quant = activation_fake_quant()
        self.weight_quant = weight_fake_quant(conv.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_quant(x)
        weight = self.weight_quant(self.conv.weight)
        return self.conv._conv_forward(x, weight, self.conv.bias)


class QuantizedLinear(nn.Module):
    """The Linear counterpart of QuantizedConv2d, for the classifier head."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear
        self.input_quant = activation_fake_quant()
        self.weight_quant = weight_fake_quant(linear.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_quant(x)
        weight = self.weight_quant(self.linear.weight)
        return nn.functional.linear(x, weight, self.linear.bias)


def insert_fake_quant(model: nn.Module) -> int:
    """Wrap every Conv2d and Linear in its quantized equivalent. Returns the count.

    Collecting the targets before mutating: replacing modules while iterating
    `named_modules()` would descend into the wrappers just created and quantize
    their inner Conv2d a second time.
    """
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]

    for name, module in targets:
        parent_path, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        wrapper = (
            QuantizedConv2d(module)
            if isinstance(module, nn.Conv2d)
            else QuantizedLinear(module)
        )
        setattr(parent, attribute, wrapper)
    return len(targets)


def set_observers(model: nn.Module, *, observe: bool, fake_quant: bool) -> None:
    """Toggle observation and fake-quant across every FakeQuantize module."""
    for module in model.modules():
        if isinstance(module, FakeQuantize):
            module.observer_enabled[0] = int(observe)
            module.fake_quant_enabled[0] = int(fake_quant)


def calibrate(model: nn.Module, batches: int, batch_size: int, seed: int = 0) -> None:
    """Populate activation ranges before any training step.

    Observers on, fake-quant off: the ranges should describe the FP32 activations
    the network actually produces, not activations already distorted by a
    fake-quant using the uninitialised scale it is trying to measure.
    """
    set_observers(model, observe=True, fake_quant=False)
    generator = torch.Generator().manual_seed(seed)
    model.eval()
    with torch.inference_mode():
        for _ in range(batches):
            model(torch.randn(batch_size, 3, 224, 224, generator=generator))
    set_observers(model, observe=False, fake_quant=True)


def train_steps(
    model: nn.Module,
    steps: int,
    batch_size: int,
    lr: float = 1e-5,
    seed: int = 1,
) -> list[float]:
    """Run the straight-through estimator for `steps` on synthetic data.

    This is a toolchain proof, not training: the data is noise and the labels are
    random, so the loss means nothing. What it establishes is that gradients
    survive the fake-quant modules — a QAT path whose STE silently zeroed every
    gradient would export a perfectly valid QDQ graph and only reveal itself as
    unexplained accuracy loss weeks later, during M2.

    The learning rate is DESIGN §8.2's lower bound (1e-5), so the exercise runs the
    same code path M2 will.
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)

    losses = []
    for _ in range(steps):
        images = torch.randn(batch_size, 3, 224, 224, generator=generator)
        labels = torch.randint(0, 1000, (batch_size,), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def gradient_reaches_weights(model: nn.Module) -> dict:
    """Confirm the STE actually passes gradient to the wrapped conv weights.

    Reported as evidence rather than asserted silently: `fake_quantize_*` uses a
    straight-through estimator, and if it were behaving as a hard round the
    gradient would be exactly zero everywhere and QAT would be a no-op wearing the
    right graph structure.
    """
    convs = [m for m in model.modules() if isinstance(m, QuantizedConv2d)]
    with_grad = [c for c in convs if c.conv.weight.grad is not None]
    nonzero = [
        c for c in with_grad if float(c.conv.weight.grad.abs().sum()) > 0.0
    ]
    return {
        "quantized_convs": len(convs),
        "convs_with_gradient": len(with_grad),
        "convs_with_nonzero_gradient": len(nonzero),
        "ste_passes_gradient": len(nonzero) == len(convs) and bool(convs),
    }


def build_qat_model(pretrained: bool = True) -> tuple[nn.Module, dict]:
    """FP32 MobileNetV2 -> BN folded -> fake-quant inserted."""
    model = build_mobilenet_v2(pretrained=pretrained)
    model.eval()
    folded = fold_conv_bn(model)
    remaining_bn = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    wrapped = insert_fake_quant(model)

    if remaining_bn:
        raise RuntimeError(
            f"{remaining_bn} BatchNorm2d modules survived folding. Every one of "
            "them would be constant-folded into a convolution *after* its weight "
            "was fake-quantized, silently invalidating that weight's scale."
        )
    return model, {"conv_bn_pairs_folded": folded, "modules_fake_quantized": wrapped}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--calibration-batches", type=int, default=8)
    parser.add_argument("--opset", type=int, default=P0_OPSET)
    parser.add_argument("--describe-json", type=Path)
    args = parser.parse_args()

    model, structure = build_qat_model(pretrained=True)
    print(f"folded {structure['conv_bn_pairs_folded']} Conv+BN pairs, "
          f"fake-quantized {structure['modules_fake_quantized']} modules")

    calibrate(model, args.calibration_batches, args.batch_size)
    losses = train_steps(model, args.steps, args.batch_size)
    ste = gradient_reaches_weights(model)
    print(f"STE gradient check: {ste}")
    if not ste["ste_passes_gradient"]:
        raise RuntimeError(
            "the straight-through estimator did not deliver a non-zero gradient "
            f"to every quantized convolution: {ste}. QAT would be a no-op."
        )

    description = export_onnx(
        model,
        args.output,
        example_input(),
        opset=args.opset,
        dynamo=False,
    )
    description["qat"] = {
        "candidate": "1 — direct QDQ fake-quant + torch.onnx.export (DESIGN §8.2)",
        "structure": structure,
        "ste_check": ste,
        "steps": args.steps,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "training_data": "SYNTHETIC — toolchain proof, not M2",
    }

    print(json.dumps(description, indent=2))
    if args.describe_json:
        args.describe_json.parent.mkdir(parents=True, exist_ok=True)
        args.describe_json.write_text(json.dumps(description, indent=2) + "\n")
        print(f"wrote {args.describe_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
