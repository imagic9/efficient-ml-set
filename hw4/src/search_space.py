"""The discrete NAS search space for CIFAR-10 and a standalone network builder.

The macro-skeleton is fixed: 4 sequential stages, each a single searchable block
followed by a 2x2 max-pool (32 -> 16 -> 8 -> 4 -> 2 spatially), then a global
average pool and a linear classifier. What the search *decides* is three axes,
exactly the ones the assignment asks for:

  * block operation (per stage) -- standard 3x3 conv vs. two efficient variants,
  * width multiplier            -- one global channel scale,
  * activation function         -- one global non-linearity.

Base channel widths (at multiplier 1.0) are [64, 128, 256, 256]; all four
multipliers keep every stage divisible by 8, so no rounding surprises. The three
operations are:

  conv3x3  -- Conv3x3 -> BN -> act. The heavy, expressive baseline block.
  dwsep    -- depthwise 3x3 -> BN -> act -> pointwise 1x1 -> BN -> act
              (MobileNet-v1 separable conv); far fewer params for the same shape.
  mbconv   -- inverted residual (MobileNet-v2): 1x1 expand(x3) -> BN -> act ->
              depthwise 3x3 -> BN -> act -> 1x1 project -> BN, with a residual
              add when in==out channels.

Space size: 3 ops ^ 4 stages x 4 widths x 5 activations = 1620 architectures.

`StandaloneNet` builds an ordinary (non-weight-shared) module for a given
architecture -- used both to retrain the best-found net from scratch and as the
honest source of truth for a design's parameter count.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------- #
# Search-space definition
# ----------------------------------------------------------------------------- #
BASE_CHANNELS = [64, 128, 256, 256]     # per-stage output width at multiplier 1.0
NUM_STAGES = len(BASE_CHANNELS)
OPS = ["conv3x3", "dwsep", "mbconv"]    # block operations (per stage)
WIDTHS = [0.5, 0.75, 1.0, 1.25]         # global width multipliers
ACTS = ["relu", "relu6", "silu", "gelu", "leakyrelu"]  # global activation choices
MB_EXPAND = 3                           # MBConv inverted-residual expansion ratio
MAX_WIDTH = max(WIDTHS)                 # supernet stores weights at this width


def round_ch(channels: int, mult: float, divisor: int = 8) -> int:
    """Scale a channel count by `mult`, rounded to a multiple of `divisor` (min 8).

    With our base widths and multipliers this is already exact, but rounding keeps
    the space well-defined if either is ever changed.
    """
    c = channels * mult
    r = max(divisor, int(c + divisor / 2) // divisor * divisor)
    if r < 0.9 * c:                     # never round down by more than 10%
        r += divisor
    return int(r)


def stage_channels(mult: float):
    """Per-stage output channel widths at a given width multiplier."""
    return [round_ch(c, mult) for c in BASE_CHANNELS]


def max_channels():
    """Per-stage output widths at the maximum multiplier (the supernet width)."""
    return stage_channels(MAX_WIDTH)


# ----------------------------------------------------------------------------- #
# Activation registry (functional form for the supernet, module form for standalone)
# ----------------------------------------------------------------------------- #
def act_fn(name: str):
    return {
        "relu": F.relu,
        "relu6": F.relu6,
        "silu": F.silu,
        "gelu": F.gelu,
        "leakyrelu": lambda x: F.leaky_relu(x, 0.1),
    }[name]


def act_module(name: str) -> nn.Module:
    return {
        "relu": lambda: nn.ReLU(inplace=True),
        "relu6": lambda: nn.ReLU6(inplace=True),
        "silu": lambda: nn.SiLU(inplace=True),
        "gelu": nn.GELU,
        "leakyrelu": lambda: nn.LeakyReLU(0.1, inplace=True),
    }[name]()


# ----------------------------------------------------------------------------- #
# Architecture representation: {"ops": [...], "width": float, "act": str}
# ----------------------------------------------------------------------------- #
def arch_key(arch) -> tuple:
    """Hashable, canonical key for an architecture (for de-dup / caching)."""
    return (tuple(arch["ops"]), float(arch["width"]), arch["act"])


def sample_arch(rng) -> dict:
    """Uniformly sample one architecture using a random.Random `rng`."""
    return {
        "ops": [rng.choice(OPS) for _ in range(NUM_STAGES)],
        "width": rng.choice(WIDTHS),
        "act": rng.choice(ACTS),
    }


def space_size() -> int:
    return len(OPS) ** NUM_STAGES * len(WIDTHS) * len(ACTS)


# ----------------------------------------------------------------------------- #
# Standalone (non-shared) building blocks
# ----------------------------------------------------------------------------- #
class Conv3x3Op(nn.Module):
    def __init__(self, in_c, out_c, act):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = act_module(act)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWSepOp(nn.Module):
    """Depthwise 3x3 + pointwise 1x1 (MobileNet-v1 separable convolution)."""

    def __init__(self, in_c, out_c, act):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c, bias=False)
        self.bn1 = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act = act_module(act)
        self.act2 = act_module(act)

    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        return self.act2(self.bn2(self.pw(x)))


class MBConvOp(nn.Module):
    """Inverted residual (MobileNet-v2): expand -> depthwise -> project, +skip."""

    def __init__(self, in_c, out_c, act, expand=MB_EXPAND):
        super().__init__()
        hid = in_c * expand
        self.use_res = in_c == out_c
        self.expand = nn.Conv2d(in_c, hid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hid)
        self.dw = nn.Conv2d(hid, hid, 3, padding=1, groups=hid, bias=False)
        self.bn2 = nn.BatchNorm2d(hid)
        self.project = nn.Conv2d(hid, out_c, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_c)
        self.act = act_module(act)
        self.act2 = act_module(act)

    def forward(self, x):
        out = self.act(self.bn1(self.expand(x)))
        out = self.act2(self.bn2(self.dw(out)))
        out = self.bn3(self.project(out))          # no activation after projection
        return out + x if self.use_res else out


def build_op(name, in_c, out_c, act) -> nn.Module:
    return {"conv3x3": Conv3x3Op, "dwsep": DWSepOp, "mbconv": MBConvOp}[name](
        in_c, out_c, act)


class StandaloneNet(nn.Module):
    """An ordinary CNN realising one architecture (no weight sharing).

    Used to retrain the best-found design from scratch and to count parameters
    honestly. Its per-layer channel shapes match exactly the sub-network the
    supernet evaluates for the same architecture.
    """

    def __init__(self, arch, num_classes=10):
        super().__init__()
        chs = stage_channels(arch["width"])
        blocks = []
        in_c = 3
        for i in range(NUM_STAGES):
            blocks.append(build_op(arch["ops"][i], in_c, chs[i], arch["act"]))
            in_c = chs[i]
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(chs[-1], num_classes)
        self.arch = arch

    def forward(self, x):
        for blk in self.blocks:
            x = F.max_pool2d(blk(x), 2)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)


def count_arch_params(arch, num_classes=10) -> int:
    """Trainable parameter count of a design (via its standalone realisation)."""
    net = StandaloneNet(arch, num_classes)
    return sum(p.numel() for p in net.parameters())
