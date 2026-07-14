"""Bonus: structured (channel) pruning guided by sensitivity, via torch-pruning.

Unlike fine-grained pruning, this physically removes whole conv filters, so it
actually shrinks MACs / params -- the kind of pruning that speeds up inference on
a Raspberry Pi. Here the per-layer budget genuinely matters (there is no single
global magnitude criterion across differently shaped layers), so a
sensitivity-guided allocation can beat a uniform one at the same MAC budget.
"""
import copy

import torch
import torch_pruning as tp

from .engine import evaluate, train


def macs_params(model, device):
    ex = torch.randn(1, 3, 32, 32, device=device)
    macs, params = tp.utils.count_ops_and_params(model, ex)
    return macs, params


def conv_names(model):
    return [n for n, m in model.named_modules() if isinstance(m, torch.nn.Conv2d)]


def _prune_model(model, device, ratio=0.0, ratio_dict=None, global_pruning=False):
    """Remove channels in-place. ratio_dict maps module -> pruning ratio."""
    ex = torch.randn(1, 3, 32, 32, device=device)
    imp = tp.importance.MagnitudeImportance(p=1)   # L1 filter norm
    ignored = [model.classifier[-1]]               # keep the 10-way output
    pruner = tp.pruner.MagnitudePruner(
        model, ex, importance=imp, pruning_ratio=ratio,
        pruning_ratio_dict=ratio_dict or {}, ignored_layers=ignored,
        global_pruning=global_pruning)
    pruner.step()
    return model


def channel_sensitivity(build_fn, state, val_loader, device, ratios=None):
    """Prune each conv layer's channels alone (no fine-tune) and record val acc."""
    if ratios is None:
        ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    names = conv_names(build_fn())
    curves = {}
    for name in names:
        curve = []
        for r in ratios:
            model = build_fn().to(device)
            model.load_state_dict(state)
            module = dict(model.named_modules())[name]
            _prune_model(model, device, ratio_dict={module: r}, global_pruning=False)
            curve.append((r, evaluate(model, val_loader, device)[1]))
        curves[name] = curve
    return curves


def profile_from_curves(curves, base_acc, tol=0.05):
    """Per-layer robustness profile, normalised to mean 1 (relative aggressiveness)."""
    prof = {}
    for name, curve in curves.items():
        best = 0.0
        for r, a in curve:
            if a >= base_acc - tol:
                best = r
        prof[name] = best
    mean = sum(prof.values()) / max(len(prof), 1)
    if mean <= 0:
        return {k: 1.0 for k in prof}
    return {k: v / mean for k, v in prof.items()}


def prune_and_finetune(build_fn, state, method, scale, profile, device,
                       train_loader, val_loader, ft_epochs, tag=""):
    """Prune channels with the given method+scale, fine-tune, return metrics."""
    model = build_fn().to(device)
    model.load_state_dict(state)
    if method == "uniform":
        _prune_model(model, device, ratio=scale, global_pruning=False)
    elif method == "global":
        _prune_model(model, device, ratio=scale, global_pruning=True)
    else:  # sensitivity: per-layer ratios shaped by the profile
        name2mod = dict(model.named_modules())
        ratio_dict = {name2mod[n]: float(min(0.9, scale * profile[n])) for n in profile}
        _prune_model(model, device, ratio_dict=ratio_dict, global_pruning=False)

    macs, params = macs_params(model, device)
    _, _, state_ft = train(model, train_loader, val_loader, device,
                           epochs=ft_epochs, lr=0.01, log_prefix=f"[{tag}] ")
    model.load_state_dict(state_ft)
    val = evaluate(model, val_loader, device)[1]
    return {"method": method, "scale": scale, "macs": macs, "params": params,
            "val": val, "model": model}
