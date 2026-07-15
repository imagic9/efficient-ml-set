#!/usr/bin/env python3
"""MobileNetV2 factory — the single place the architecture is constructed.

DESIGN §8 starts every optimization candidate (M0-M4) from an ImageNet-pretrained
MobileNetV2. One factory keeps M0 training, the P0 spike, PTQ calibration and QAT
from drifting into three subtly different architectures whose numbers cannot be
compared.

The 224x224 default here is ImageNet's, not the project's: DESIGN §5.5 leaves the
224x224-versus-256x192 input contract as a pre-registered control to be resolved
before M0. Nothing in this module freezes it, and `INPUT_SHAPE_IMAGENET` is named
for what it is so no later reader mistakes it for a decision.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

# ImageNet's own input geometry. DESIGN §5.5 may replace the project's with
# 256x192; that control is not this module's to decide.
INPUT_SHAPE_IMAGENET = (1, 3, 224, 224)

# torchvision's ImageNet normalisation, repeated in DESIGN §5.6 line 603. The C++
# preprocessor must match these exactly or P1 parity fails.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_mobilenet_v2(
    num_classes: int = 1000,
    pretrained: bool = True,
    width_mult: float = 1.0,
) -> nn.Module:
    """Build MobileNetV2, optionally re-heading it for `num_classes`.

    `pretrained` loads IMAGENET1K_V2 weights, the stronger of torchvision's two
    ImageNet checkpoints. M2 QAT initialises from the M0 FP32 checkpoint rather
    than from here (DESIGN §8.2); this factory only builds the architecture.

    A non-default `width_mult` cannot use pretrained weights — the tensor shapes
    no longer match — so that combination raises instead of silently returning a
    randomly initialised network that would look trained.
    """
    if pretrained and width_mult != 1.0:
        raise ValueError(
            f"pretrained=True is incompatible with width_mult={width_mult}: "
            "torchvision publishes ImageNet weights for width_mult=1.0 only. "
            "Pass pretrained=False to build a scaled architecture from scratch."
        )

    weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
    model = mobilenet_v2(weights=weights, width_mult=width_mult)

    if num_classes != 1000:
        # Replace only the final Linear and keep the classifier's dropout, so the
        # head stays the architecture torchvision trained, minus its class count.
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model


def example_input(shape: tuple[int, ...] = INPUT_SHAPE_IMAGENET) -> torch.Tensor:
    """A deterministic tensor for export tracing and smoke runs.

    Seeded locally rather than through the global RNG: an export helper that
    silently reorders the caller's random stream would make a training run
    irreproducible depending on whether it exported first.
    """
    generator = torch.Generator().manual_seed(0)
    return torch.randn(*shape, generator=generator)
