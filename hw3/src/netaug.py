"""NetAug (Network Augmentation, Cai et al., ICLR 2022) for a tiny VGG11, + KD.

The insight behind NetAug is the opposite of the usual one: **tiny models underfit**,
not overfit. So instead of *regularising* a small net (dropout, heavy augmentation),
we *augment its capacity during training* -- we embed the target tiny model into a
wider "augmented" supernet that **shares its weights**, train both at each step, and
keep only the tiny base model at inference. The augmented forward pushes extra
gradient signal through the shared weights, which helps the underfitting base net.

Here the target is a width-compressed VGG11 (0.25x channels) -- a genuinely small
model, the regime NetAug is designed for. This is a fourth axis of compression
(width), complementing the pruning/quantization students in the core.

Implementation: one set of conv/linear weights stored at the **augmented** width.
The base network is the sub-network that reads the first `base` channels of every
layer, i.e. a slice `W[:out_base, :in_base]` of the shared augmented weight -- so
base-loss and aug-loss gradients accumulate into the *same* parameters. BatchNorm
is width-specific (the standard slimmable-net trick): a separate BN per width, since
running stats differ. Only the base BN + base head are used at inference.

KD integration: both the base and the augmented forward are trained against the
teacher's soft targets (DistillLoss), so NetAug and distillation compound.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .engine import evaluate
from .distill import _teacher_logits
from .utils import AverageMeter

# VGG11 (config 'A') convolutional channel widths at full (1.0x) scale; 'M' = maxpool.
VGG11_CHANNELS = [64, 128, 256, 256, 512, 512, 512, 512]
VGG11_POOL_AFTER = {0, 1, 3, 5, 7}   # indices of conv layers followed by a 2x2 maxpool


def _scaled(channels, mult):
    """Per-layer channel widths at a given width multiplier (>=1 channel)."""
    return [max(1, round(c * mult)) for c in channels]


class ElasticVGG11(nn.Module):
    """Width-elastic VGG11-BN whose conv/linear weights live at the augmented width.

    forward(x, mode="base"|"aug"): runs the sub-network at the base or augmented
    widths, slicing the shared conv/linear weights and using the width-specific BN.
    The base model (mode="base") is the deliverable; the augmented model only exists
    to help train the shared weights during NetAug.
    """

    def __init__(self, base_mult=0.25, aug_mult=1.0, num_classes=10):
        super().__init__()
        self.base = _scaled(VGG11_CHANNELS, base_mult)
        self.aug = _scaled(VGG11_CHANNELS, aug_mult)
        self.base_mult, self.aug_mult = base_mult, aug_mult

        # shared conv weights at the augmented width (bias-free; BN carries the shift)
        self.conv_w = nn.ParameterList()
        in_aug = 3
        for out_aug in self.aug:
            self.conv_w.append(nn.Parameter(torch.empty(out_aug, in_aug, 3, 3)))
            in_aug = out_aug
        for w in self.conv_w:
            nn.init.kaiming_normal_(w, mode="fan_out", nonlinearity="relu")

        # width-specific BatchNorm (one per width per conv position)
        self.bn_base = nn.ModuleList(nn.BatchNorm2d(c) for c in self.base)
        self.bn_aug = nn.ModuleList(nn.BatchNorm2d(c) for c in self.aug)

        # single shared classifier at the augmented last-conv width; base slices it
        self.head_w = nn.Parameter(torch.empty(num_classes, self.aug[-1]))
        self.head_b = nn.Parameter(torch.zeros(num_classes))
        nn.init.normal_(self.head_w, std=0.01)

    def forward(self, x, mode="base"):
        cfg = self.base if mode == "base" else self.aug
        bns = self.bn_base if mode == "base" else self.bn_aug
        in_c = 3
        for i, out_c in enumerate(cfg):
            w = self.conv_w[i][:out_c, :in_c]              # slice shared aug weight
            x = F.conv2d(x, w, padding=1)
            x = F.relu(bns[i](x), inplace=True)
            if i in VGG11_POOL_AFTER:
                x = F.max_pool2d(x, 2)
            in_c = out_c
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)         # [B, out_c]
        logits = F.linear(x, self.head_w[:, :cfg[-1]], self.head_b)
        return logits

    @torch.no_grad()
    def base_param_count(self):
        """Parameter count of the deliverable (base) sub-network only."""
        n, in_c = 0, 3
        for out_c in self.base:
            n += out_c * in_c * 9 + 2 * out_c              # conv slice + BN affine
            in_c = out_c
        n += self.base[-1] * self.head_w.size(0) + self.head_b.numel()
        return n


@torch.no_grad()
def recalibrate_base_bn(model, loader, device, num_batches=50):
    """Refresh the base BN running stats (the base weights are a slice that shifts as
    the shared aug weights train, so its BN stats need a short re-accumulation)."""
    saved = {}
    for bn in model.bn_base:
        saved[bn] = bn.momentum
        bn.reset_running_stats()
        bn.momentum = None
    model.train()
    seen = 0
    for images, _ in loader:
        model(images.to(device), mode="base")
        seen += 1
        if seen >= num_batches:
            break
    model.eval()
    for bn, mom in saved.items():
        bn.momentum = mom


def netaug_train(model, teacher, train_loader, val_loader, device, epochs, distill,
                 netaug=True, aug_weight=1.0, lr=0.05, momentum=0.9,
                 weight_decay=5e-4, log_prefix=""):
    """Train the base VGG11 sub-network, optionally with NetAug and/or KD.

    netaug=True  -> each step also forwards the augmented (full-width) net and adds
                    aug_weight * loss(aug); both losses use `distill` (KD if a teacher
                    is given, pure CE if distill has alpha=1 / teacher is None).
    netaug=False -> plain base-only training (the KD / CE baselines).

    Model selection + returned best state are on the BASE net's validation accuracy.
    Returns (history, best_val, best_state).
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                weight_decay=weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if teacher is not None:
        teacher.eval()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, best_state = 0.0, None
    for epoch in range(epochs):
        model.train()
        loss_m, acc_m = AverageMeter(), AverageMeter()
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            t_logits = _teacher_logits(teacher, images) if teacher is not None else None
            optimizer.zero_grad()
            base_logits = model(images, mode="base")
            loss = distill(base_logits, t_logits, targets)
            if netaug:                                     # add the augmented-net term
                aug_logits = model(images, mode="aug")
                loss = loss + aug_weight * distill(aug_logits, t_logits, targets)
            loss.backward()
            optimizer.step()
            acc_m.update((base_logits.argmax(1) == targets).float().mean().item(),
                         images.size(0))
            loss_m.update(loss.item(), images.size(0))
        scheduler.step()

        if netaug:                                         # base BN drifts as shared weights move
            recalibrate_base_bn(model, train_loader, device)
        va_loss, va_acc = evaluate_base(model, val_loader, device)
        history["train_loss"].append(loss_m.avg)
        history["train_acc"].append(acc_m.avg)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"{log_prefix}epoch {epoch + 1:3d}/{epochs} "
              f"train_acc={acc_m.avg:.4f} val_acc={va_acc:.4f} best={best_acc:.4f}")
    return history, best_acc, best_state


@torch.inference_mode()
def evaluate_base(model, loader, device, criterion=None):
    """Evaluate the base sub-network (the deliverable)."""
    model.eval()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    loss_m, acc_m = AverageMeter(), AverageMeter()
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images, mode="base")
        loss_m.update(criterion(logits, targets).item(), images.size(0))
        acc_m.update((logits.argmax(1) == targets).float().mean().item(), images.size(0))
    return loss_m.avg, acc_m.avg
