"""A single-path one-shot supernet (SPOS-style) with weight sharing.

This is the one-shot proxy that makes the architecture search cheap. Instead of
training every candidate from scratch, we train **one** over-parameterised network
whose weights are *shared* across the whole search space, then read off a proxy
score for any candidate by simply running its sub-path -- no per-candidate training.

How the sharing works, extending the slimmable-net trick we already used for NetAug
in HW3:

  * Weights live at the **maximum** width (multiplier 1.25). A narrower sub-network
    is the channel slice `W[:out, :in]` of the shared weight, so gradients from every
    sampled width accumulate into the same parameters.
  * Every stage holds **all** candidate operations (conv3x3 / dwsep / mbconv) as
    parallel branches; a sub-network activates exactly one branch per stage.
  * BatchNorm is applied functionally on the active channel slice. One-shot BN
    statistics are unreliable (they mix every width/op seen during training), so
    before evaluating any sub-network we **recalibrate** its BN running stats on a
    handful of training batches -- standard one-shot NAS practice.

Training (`spos_train`): each step samples ONE architecture uniformly at random and
updates only its path (Single-Path One-Shot, Guo et al., ECCV 2020). Over many steps
every operation/width/activation is trained, and no sub-path is favoured a priori.

Reference: Guo et al., "Single Path One-Shot Neural Architecture Search with Uniform
Sampling", ECCV 2020 (arXiv:1904.00420).
"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .search_space import (NUM_STAGES, OPS, MB_EXPAND, act_fn, max_channels,
                           stage_channels, sample_arch)
from .utils import AverageMeter


class SlicedBN(nn.Module):
    """BatchNorm stored at max width, applied to the first `c` channels.

    Kept functional (rather than nn.BatchNorm2d) so the same module serves every
    width by slicing its affine params and running stats to `[:c]`.
    """

    def __init__(self, max_c, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(max_c))
        self.bias = nn.Parameter(torch.zeros(max_c))
        self.register_buffer("running_mean", torch.zeros(max_c))
        self.register_buffer("running_var", torch.ones(max_c))

    def forward(self, x, c, training, momentum=0.1):
        return F.batch_norm(x, self.running_mean[:c], self.running_var[:c],
                            self.weight[:c], self.bias[:c], training, momentum, self.eps)

    @torch.no_grad()
    def reset(self, c):
        self.running_mean[:c].zero_()
        self.running_var[:c].fill_(1.0)


class SharedConv3x3(nn.Module):
    def __init__(self, in_max, out_max):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_max, in_max, 3, 3))
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        self.bn = SlicedBN(out_max)

    def forward(self, x, in_c, out_c, act, training):
        x = F.conv2d(x, self.weight[:out_c, :in_c], padding=1)
        return act(self.bn(x, out_c, training))

    def bns(self, out_c):
        return [(self.bn, out_c)]


class SharedDWSep(nn.Module):
    def __init__(self, in_max, out_max):
        super().__init__()
        self.dw = nn.Parameter(torch.empty(in_max, 1, 3, 3))
        self.pw = nn.Parameter(torch.empty(out_max, in_max, 1, 1))
        nn.init.kaiming_normal_(self.dw, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.pw, mode="fan_out", nonlinearity="relu")
        self.bn_dw = SlicedBN(in_max)
        self.bn_pw = SlicedBN(out_max)

    def forward(self, x, in_c, out_c, act, training):
        x = F.conv2d(x, self.dw[:in_c], padding=1, groups=in_c)
        x = act(self.bn_dw(x, in_c, training))
        x = F.conv2d(x, self.pw[:out_c, :in_c])
        return act(self.bn_pw(x, out_c, training))

    def bns(self, out_c):
        return [(self.bn_dw, "in"), (self.bn_pw, out_c)]


class SharedMBConv(nn.Module):
    """Inverted residual with a shared, sliceable expand/depthwise/project stack."""

    def __init__(self, in_max, out_max, expand=MB_EXPAND):
        super().__init__()
        self.expand = expand
        hid_max = in_max * expand
        self.expand_w = nn.Parameter(torch.empty(hid_max, in_max, 1, 1))
        self.dw = nn.Parameter(torch.empty(hid_max, 1, 3, 3))
        self.project_w = nn.Parameter(torch.empty(out_max, hid_max, 1, 1))
        for w in (self.expand_w, self.dw, self.project_w):
            nn.init.kaiming_normal_(w, mode="fan_out", nonlinearity="relu")
        self.bn1 = SlicedBN(hid_max)
        self.bn2 = SlicedBN(hid_max)
        self.bn3 = SlicedBN(out_max)

    def forward(self, x, in_c, out_c, act, training):
        hid = in_c * self.expand
        out = F.conv2d(x, self.expand_w[:hid, :in_c])
        out = act(self.bn1(out, hid, training))
        out = F.conv2d(out, self.dw[:hid], padding=1, groups=hid)
        out = act(self.bn2(out, hid, training))
        out = F.conv2d(out, self.project_w[:out_c, :hid])
        out = self.bn3(out, out_c, training)            # no activation after projection
        return out + x if in_c == out_c else out

    def bns(self, out_c):
        # hidden-width BNs recalibrate at whatever `in_c` the active path feeds them
        return [(self.bn1, "hid"), (self.bn2, "hid"), (self.bn3, out_c)]


_SHARED_OP = {"conv3x3": SharedConv3x3, "dwsep": SharedDWSep, "mbconv": SharedMBConv}


class SlicedLinear(nn.Module):
    """Classifier stored at the max last-stage width; slices its input dimension."""

    def __init__(self, in_max, num_classes):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_max))
        self.bias = nn.Parameter(torch.zeros(num_classes))
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, x, in_c):
        return F.linear(x, self.weight[:, :in_c], self.bias)


class SuperNet(nn.Module):
    """Weight-sharing supernet spanning the whole search space.

    forward(x, arch) runs the single sub-path selected by `arch`
    (`{"ops": [...], "width": float, "act": str}`) with BN in the given `training`
    mode. During SPOS training we forward in train mode; for proxy evaluation we
    recalibrate BN (see `recalibrate_bn`) then forward in eval mode.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        maxc = max_channels()
        self.stages = nn.ModuleList()
        in_max = 3
        for i in range(NUM_STAGES):
            self.stages.append(nn.ModuleDict(
                {op: _SHARED_OP[op](in_max, maxc[i]) for op in OPS}))
            in_max = maxc[i]
        self.head = SlicedLinear(maxc[-1], num_classes)

    def forward(self, x, arch, training=None):
        if training is None:
            training = self.training
        act = act_fn(arch["act"])
        chs = stage_channels(arch["width"])
        in_c = 3
        for i in range(NUM_STAGES):
            block = self.stages[i][arch["ops"][i]]
            x = block(x, in_c, chs[i], act, training)
            x = F.max_pool2d(x, 2)
            in_c = chs[i]
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x, in_c)


@torch.no_grad()
def recalibrate_bn(model, arch, loader, device, num_batches=64):
    """Refresh the BN running stats of the sub-path selected by `arch`.

    One-shot BN stats mix every width/op seen in training, so they are meaningless
    for a specific sub-network. We reset the active path's BN and re-accumulate its
    statistics (EMA) over a few training batches before evaluating.
    """
    chs = stage_channels(arch["width"])
    in_c = 3
    active = []                                     # (SlicedBN, channel_count) to reset
    for i in range(NUM_STAGES):
        block = model.stages[i][arch["ops"][i]]
        for bn, tag in block.bns(chs[i]):
            c = {"in": in_c, "hid": in_c * MB_EXPAND}.get(tag, tag)
            active.append((bn, c))
        in_c = chs[i]
    for bn, c in active:
        bn.reset(c)
    model.train()
    seen = 0
    for images, _ in loader:
        model(images.to(device), arch, training=True)
        seen += 1
        if seen >= num_batches:
            break
    model.eval()


@torch.inference_mode()
def evaluate_subnet(model, arch, loader, device, criterion=None):
    """Evaluate one sub-network (assumes BN already recalibrated). Returns loss, acc."""
    model.eval()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    loss_m, acc_m = AverageMeter(), AverageMeter()
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images, arch, training=False)
        loss_m.update(criterion(logits, targets).item(), images.size(0))
        acc_m.update((logits.argmax(1) == targets).float().mean().item(), images.size(0))
    return loss_m.avg, acc_m.avg


def spos_train(model, train_loader, device, epochs, lr=0.05, momentum=0.9,
               weight_decay=4e-5, seed=42, log_prefix=""):
    """Single-Path One-Shot training: each step trains one uniformly-sampled path.

    Returns the per-epoch mean training loss/acc of the sampled paths (a coarse
    health signal -- the supernet is not one model but a distribution over paths).
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                weight_decay=weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    rng = random.Random(seed)

    history = {"train_loss": [], "train_acc": []}
    for epoch in range(epochs):
        model.train()
        loss_m, acc_m = AverageMeter(), AverageMeter()
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            arch = sample_arch(rng)                 # one random path per step
            optimizer.zero_grad()
            logits = model(images, arch, training=True)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            loss_m.update(loss.item(), images.size(0))
            acc_m.update((logits.argmax(1) == targets).float().mean().item(),
                         images.size(0))
        scheduler.step()
        history["train_loss"].append(loss_m.avg)
        history["train_acc"].append(acc_m.avg)
        print(f"{log_prefix}epoch {epoch + 1:3d}/{epochs} "
              f"path_train_loss={loss_m.avg:.4f} path_train_acc={acc_m.avg:.4f}")
    return history
