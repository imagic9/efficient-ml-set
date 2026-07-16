#!/usr/bin/env python3
"""QAT candidate 1 — QDQ fake-quant inserted directly, exported with torch.onnx.

DESIGN §8.2 deliberately does not pre-select a QAT library: the tool is an *output*
of parity gate P0, not an input to it. This module implements the first candidate
on that list — fake-quant modules we place ourselves, exported with
`torch.onnx.export` — chosen first because the export semantics are then ours
rather than a library's.

## What P0 measured, and why this file looks the way it does

The obvious construction — fake-quant on each convolution's *input* and weight,
which is what `pytorch-quantization` and most TensorRT-oriented flows emit — was
tried first and **failed**, on 2026-07-15, on gx10:

    optimized graph: 45 FusedConv, 5 Conv, only 2 QLinearConv
    float kernels executed: FusedConv x145, Conv x15, Gemm x3

That is the exact failure DESIGN §8.2 warns about: a float graph carrying rounded
weights. The cause is visible in the optimized graph rather than inferred. ORT
matches `DQ -> Conv -> Q` to build a QLinearConv, and MobileNetV2's ReLU6 sits
between the convolution and the next quantizer. ORT's float-level
ConvActivationFusion reaches `Conv + Clip -> FusedConv` first, and a FusedConv can
never match the QDQ rule afterwards.

The lesson is worth more than the fix: **the QAT library is not the axis that
matters — QDQ placement against ORT's fusion rules is.** Candidates 2 and 3
(`pytorch-quantization`, `torchao`) place QDQ the same input-side way, because they
target TensorRT, which fuses `DQ -> Conv -> ReLU` happily. Swapping library would
have reproduced this failure with more dependencies.

So this module quantizes **every tensor boundary on the output side**, which is the
canonical QDQ form ORT consumes and the same shape ORT's own PTQ quantizer
produces: input, each convolution output, each residual add, the pooled vector, and
the classifier output.

## The ReLU6 subtlety, and why it is exact rather than convenient

To give ORT `DQ -> Conv -> Q`, ReLU6 must not sit between the convolution and its
quantizer. It is *removed at export*, and this is exact, not an approximation:

  - the quantizer observes post-ReLU6 activations, so its range is [0, m], m <= 6;
  - a fake-quant over [0, m] already clamps its input to [0, m];
  - so for any input, relu6-then-quantize and quantize-alone produce identical
    output — both clamp to [0, m] on the same grid.

ORT's PTQ quantizer performs precisely this removal, which is why the PTQ graph has
no Clip nodes either. The argument is checked numerically by
`verify_relu6_removal_is_exact` rather than trusted: an exactness claim that is only
prose is how a silent accuracy regression enters.

Usage (spike only):
    python -m wildlife_trigger.optimize.qat --output m2_qat.onnx --steps 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import FakeQuantize
from torch.ao.quantization.observer import (
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)
from torch.nn.utils.fusion import fuse_conv_bn_eval
from torchvision.models.mobilenetv2 import InvertedResidual

from wildlife_trigger.models.export import P0_OPSET, describe, export_onnx
from wildlife_trigger.models.mobilenet import build_mobilenet_v2, example_input
from wildlife_trigger.optimize.qdq_scalar import scalarize_per_tensor_qdq

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


def weight_fake_quant() -> FakeQuantize:
    """Per-channel symmetric weight fake-quant over output channels (axis 0).

    Symmetric pins every weight zero-point at 0, which is what ORT's S8S8
    QLinearConv and the ARM64 signed dot-product kernels consume.
    """
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

    BatchNorm is folded *before* fake-quant is inserted, never after. Quantizing a
    weight and then folding BN rescales that weight by gamma/sigma, moving it off
    the grid its scale describes — the exported QDQ scales would be wrong in a way
    nothing downstream detects. Folding first makes the weight that is quantized
    the weight that is deployed.

    DESIGN §8.2 already plans to freeze BN statistics after stabilization; folding
    is that limit case, and it means M2 fine-tunes in FP32 briefly *before* this
    stage rather than during it.

    The BatchNorm becomes Identity rather than being deleted, so positional
    indexing into `nn.Sequential` — which MobileNetV2 relies on throughout
    `features` — keeps working.
    """
    if model.training:
        raise RuntimeError(
            "fold_conv_bn requires eval mode: it folds BatchNorm running "
            "statistics into the convolution weights, which is only equivalent to "
            "the original network when BN has stopped updating them. Call "
            "model.eval() first."
        )

    folded = 0
    for module in list(model.modules()):
        children = list(module.named_children())
        for (conv_name, conv), (bn_name, bn) in zip(children, children[1:]):
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                setattr(module, conv_name, fuse_conv_bn_eval(conv, bn))
                setattr(module, bn_name, nn.Identity())
                folded += 1
    return folded


class QuantConv2d(nn.Module):
    """Conv2d with a fake-quantized weight and a fake-quantized output.

    `has_relu6` records that a ReLU6 followed this convolution and has been
    absorbed: it is applied during calibration and training, and dropped in export
    mode so ORT sees `DQ -> Conv -> Q`. See the module docstring for why that is
    exact.

    The bias is deliberately not fake-quantized. ORT derives the bias scale from
    input_scale * weight_scale and stores bias as INT32; a fake-quant here would
    impose an INT8 grid the runtime never uses, degrading accuracy to model
    something that does not happen.
    """

    def __init__(self, conv: nn.Conv2d, has_relu6: bool):
        super().__init__()
        self.conv = conv
        self.has_relu6 = has_relu6
        self.weight_quant = weight_fake_quant()
        self.output_quant = activation_fake_quant()
        self.export_mode = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight_quant(self.conv.weight)
        y = self.conv._conv_forward(x, weight, self.conv.bias)
        if self.has_relu6 and not self.export_mode:
            y = F.relu6(y)
        return self.output_quant(y)


class QuantLinear(nn.Module):
    """The Linear counterpart of QuantConv2d, for the classifier head.

    Unlike QuantConv2d this also quantizes its *input*, which is not redundant.
    `torch.flatten` sits between the pooled vector's quantizer and this layer, so
    the Gemm's activation input is the Flatten's output — a tensor boundary no
    other module can reach. Without a quantizer here ORT sees a bare
    `Flatten -> Gemm` with nothing to match, and the classifier stays float while
    every convolution around it runs as integer. Measured: that is exactly what
    happened, and ORT's own PTQ output puts a DequantizeLinear in the same place
    (`/Flatten_output_0_DequantizeLinear`).
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear
        self.input_quant = activation_fake_quant()
        self.weight_quant = weight_fake_quant()
        self.output_quant = activation_fake_quant()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_quant(x)
        weight = self.weight_quant(self.linear.weight)
        return self.output_quant(F.linear(x, weight, self.linear.bias))


class QuantResidual(nn.Module):
    """An InvertedResidual whose skip-connection Add output is quantized.

    Without this the Add runs in float between two integer regions, which ORT's
    PTQ output shows is unnecessary: it produces QLinearAdd for all ten of them.
    """

    def __init__(self, block: InvertedResidual):
        super().__init__()
        self.block = block
        self.output_quant = activation_fake_quant()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_quant(x + self.block.conv(x))


class QuantMobileNetV2(nn.Module):
    """MobileNetV2 with quantizers at the boundaries its forward() creates.

    MobileNetV2.forward applies the pool and flatten *functionally*, so no module
    replacement can reach those tensors. Restating the forward pass is the only way
    to quantize the pooled vector — and ORT's PTQ result shows it pays: it yields
    QLinearGlobalAveragePool rather than a float pool between integer regions.
    """

    def __init__(self, base: nn.Module):
        super().__init__()
        self.input_quant = activation_fake_quant()
        self.features = base.features
        self.pool_quant = activation_fake_quant()
        self.classifier = base.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_quant(x)
        x = self.features(x)
        x = self.pool_quant(F.adaptive_avg_pool2d(x, (1, 1)))
        x = torch.flatten(x, 1)
        return self.classifier(x)


def rewrite_convolutions(model: nn.Module) -> dict:
    """Replace every Conv2d with QuantConv2d, absorbing any ReLU6 that follows.

    Works on container children rather than on torchvision's class names: after BN
    folding, `Conv2dNormActivation` is a Sequential of [Conv2d, Identity, ReLU6]
    and a linear-bottleneck is [.., Conv2d, Identity]. Scanning children handles
    both without depending on torchvision's internal class layout, which is not a
    stable API.

    The absorbed ReLU6 is replaced by Identity so `nn.Sequential`'s positional
    indexing survives.

    `list(model.modules())` is materialised before mutating. `modules()` is a lazy
    generator: replacing a Conv2d with a QuantConv2d that *holds* that Conv2d makes
    the generator descend into the new wrapper and quantize its inner convolution
    again, without end. That is not hypothetical — it recursed on the first run.
    """
    converted = 0
    absorbed = 0

    for module in list(model.modules()):
        children = list(module.named_children())
        for index, (name, child) in enumerate(children):
            if not isinstance(child, nn.Conv2d):
                continue

            # Look past the Identity left by BN folding for a ReLU6.
            relu_slot = None
            for offset in (1, 2):
                if index + offset >= len(children):
                    break
                next_name, next_child = children[index + offset]
                if isinstance(next_child, nn.ReLU6):
                    relu_slot = next_name
                    break
                if not isinstance(next_child, nn.Identity):
                    break

            setattr(module, name, QuantConv2d(child, has_relu6=relu_slot is not None))
            converted += 1
            if relu_slot is not None:
                setattr(module, relu_slot, nn.Identity())
                absorbed += 1

    return {"convolutions_quantized": converted, "relu6_absorbed": absorbed}


def rewrite_residuals(model: nn.Module) -> int:
    """Wrap every skip-connected InvertedResidual so its Add output is quantized."""
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, InvertedResidual) and module.use_res_connect
    ]
    for name, block in targets:
        parent_path, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, attribute, QuantResidual(block))
    return len(targets)


def rewrite_classifier(model: nn.Module) -> int:
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    for name, linear in targets:
        parent_path, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, attribute, QuantLinear(linear))
    return len(targets)


def set_export_mode(model: nn.Module, enabled: bool) -> None:
    """Drop or restore the absorbed ReLU6 ops across every QuantConv2d."""
    for module in model.modules():
        if isinstance(module, QuantConv2d):
            module.export_mode = enabled


def set_observers(model: nn.Module, *, observe: bool, fake_quant: bool) -> None:
    """Toggle observation and fake-quant across every FakeQuantize module."""
    for module in model.modules():
        if isinstance(module, FakeQuantize):
            module.observer_enabled[0] = int(observe)
            module.fake_quant_enabled[0] = int(fake_quant)


def synthetic_batch(batch_size: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(batch_size, 3, 224, 224, generator=generator)


def calibrate(model: nn.Module, batches: int, batch_size: int, seed: int = 0) -> None:
    """Populate activation ranges before any training step.

    Observers on, fake-quant off: the ranges must describe the FP32 activations the
    network actually produces, not activations already distorted by a fake-quant
    using the uninitialised scale it is trying to measure.
    """
    set_observers(model, observe=True, fake_quant=False)
    generator = torch.Generator().manual_seed(seed)
    model.eval()
    with torch.inference_mode():
        for _ in range(batches):
            model(synthetic_batch(batch_size, generator))
    set_observers(model, observe=False, fake_quant=True)


def verify_relu6_removal_is_exact(
    model: nn.Module, batches: int = 4, batch_size: int = 4, seed: int = 7
) -> dict:
    """Check that export mode changes no output bit, and report the margin.

    The equivalence argument (a quantizer over [0, m<=6] already clamps exactly as
    ReLU6 does) is sound but rests on every absorbed ReLU6's observed range being
    non-negative and at most 6. That is a property of the calibration data, so it
    is measured here rather than asserted. A non-zero difference means some
    quantizer's range does not bound its ReLU6, and the export is not equivalent.
    """
    generator = torch.Generator().manual_seed(seed)
    device = next(model.parameters()).device
    model.eval()
    worst = 0.0
    with torch.inference_mode():
        for _ in range(batches):
            x = synthetic_batch(batch_size, generator).to(device)
            set_export_mode(model, False)
            reference = model(x)
            set_export_mode(model, True)
            exported = model(x)
            worst = max(worst, float((reference - exported).abs().max()))
    set_export_mode(model, False)

    ranges = [
        (float(m.output_quant.activation_post_process.min_val),
         float(m.output_quant.activation_post_process.max_val))
        for m in model.modules()
        if isinstance(m, QuantConv2d) and m.has_relu6
    ]
    violations = [r for r in ranges if r[0] < 0.0 or r[1] > 6.0]

    return {
        "max_abs_difference": worst,
        "exact": worst == 0.0 and not violations,
        "absorbed_relu6_count": len(ranges),
        "range_violations": violations,
    }


def train_steps(
    model: nn.Module,
    steps: int,
    batch_size: int,
    lr: float = 1e-5,
    seed: int = 1,
) -> list[float]:
    """Run the straight-through estimator for `steps` on synthetic data.

    A toolchain proof, not training: the data is noise and the labels are random,
    so the loss means nothing. What it establishes is that gradients survive the
    fake-quant modules — a QAT path whose STE silently zeroed every gradient would
    export a perfectly valid QDQ graph and only reveal itself as unexplained
    accuracy loss weeks later, during M2.

    The learning rate is DESIGN §8.2's lower bound (1e-5), so the exercise runs the
    same code path M2 will.
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)

    losses = []
    for _ in range(steps):
        images = synthetic_batch(batch_size, generator)
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
    straight-through estimator, and if it behaved as a hard round the gradient
    would be exactly zero everywhere — QAT would be a no-op wearing the right graph
    structure.
    """
    convs = [m for m in model.modules() if isinstance(m, QuantConv2d)]
    with_grad = [c for c in convs if c.conv.weight.grad is not None]
    nonzero = [c for c in with_grad if float(c.conv.weight.grad.abs().sum()) > 0.0]
    return {
        "quantized_convs": len(convs),
        "convs_with_gradient": len(with_grad),
        "convs_with_nonzero_gradient": len(nonzero),
        "ste_passes_gradient": len(nonzero) == len(convs) and bool(convs),
    }


def build_qat_model(
    pretrained: bool = True, base: nn.Module | None = None
) -> tuple[nn.Module, dict]:
    """FP32 MobileNetV2 -> BN folded -> output-side QDQ inserted everywhere.

    `base` lets M2 hand over the M0 fine-tuned network (16 outputs, checkpoint
    weights already loaded) instead of the ImageNet factory model the P0 spike
    used — DESIGN §8.2's "initialize from M0, never from M1" is the caller's
    responsibility, and the caller proves it by hash before building this.
    """
    if base is None:
        base = build_mobilenet_v2(pretrained=pretrained)
    base.eval()

    folded = fold_conv_bn(base)
    remaining_bn = sum(1 for m in base.modules() if isinstance(m, nn.BatchNorm2d))
    if remaining_bn:
        raise RuntimeError(
            f"{remaining_bn} BatchNorm2d modules survived folding. Every one would "
            "be constant-folded into a convolution *after* its weight was "
            "fake-quantized, silently invalidating that weight's scale."
        )

    residuals = rewrite_residuals(base)
    convs = rewrite_convolutions(base)
    linears = rewrite_classifier(base)
    model = QuantMobileNetV2(base)

    return model, {
        "conv_bn_pairs_folded": folded,
        "residual_adds_quantized": residuals,
        "linears_quantized": linears,
        **convs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--calibration-batches", type=int, default=8)
    parser.add_argument("--opset", type=int, default=P0_OPSET)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--describe-json", type=Path)
    args = parser.parse_args()

    # The global RNG, not only the local generators. MobileNetV2's classifier holds
    # an nn.Dropout, which draws from torch's global stream during training and
    # ignores any generator passed to the data. Without this the exported QAT model
    # differs on every run: P0 measured the same pipeline produce argmax 21 and then
    # 908 from identical inputs. DESIGN §9.2 requires seeds to be recorded, and an
    # artifact that cannot be regenerated is not evidence.
    torch.manual_seed(args.seed)

    model, structure = build_qat_model(pretrained=True)
    print(f"structure: {structure}")

    calibrate(model, args.calibration_batches, args.batch_size)

    equivalence = verify_relu6_removal_is_exact(model)
    print(f"ReLU6 removal equivalence: {equivalence}")
    if not equivalence["exact"]:
        raise RuntimeError(
            "dropping the absorbed ReLU6 ops changed the model's output "
            f"({equivalence}). The export would not be equivalent to what was "
            "trained; do not ship this artifact."
        )

    losses = train_steps(model, args.steps, args.batch_size)
    ste = gradient_reaches_weights(model)
    print(f"STE gradient check: {ste}")
    if not ste["ste_passes_gradient"]:
        raise RuntimeError(
            "the straight-through estimator did not deliver a non-zero gradient to "
            f"every quantized convolution: {ste}. QAT would be a no-op."
        )

    set_export_mode(model, True)
    try:
        raw_export = args.output.with_suffix(".raw.onnx")
        export_onnx(
            model, raw_export, example_input(), opset=args.opset, dynamo=False
        )
    finally:
        set_export_mode(model, False)

    # torch emits per-tensor QDQ scales as shape-[1] tensors; ORT requires rank 0
    # and refuses to load the graph otherwise. The raw export is kept beside the
    # repaired one so the difference stays inspectable rather than becoming folklore.
    scalar_fix = scalarize_per_tensor_qdq(raw_export, args.output)
    print(f"QDQ scalar fix: {scalar_fix}")

    description = describe(args.output)
    description["exporter"] = "torchscript"
    description["qdq_scalar_fix"] = scalar_fix
    description["qat"] = {
        "candidate": "1 — direct QDQ fake-quant + torch.onnx.export (DESIGN §8.2)",
        "placement": "output-side QDQ at every tensor boundary; ReLU6 absorbed",
        "structure": structure,
        "relu6_removal_equivalence": equivalence,
        "ste_check": ste,
        "seed": args.seed,
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
