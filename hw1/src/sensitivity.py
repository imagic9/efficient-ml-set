"""Bonus: per-layer sensitivity analysis for smarter (non-uniform) pruning.

Idea: prune one layer at a time (leaving the rest intact), measure how much
validation accuracy drops at each sparsity level. Layers whose accuracy collapses
early are "sensitive" and should be pruned less; robust layers can absorb more.
"""
import copy
import torch

from .prune import _layer_mask, prunable_layers
from .engine import evaluate


@torch.no_grad()
def scan(model, val_loader, device, ratios=None):
    """Return {layer_name: [(ratio, val_acc), ...]} probing each layer alone.

    No fine-tuning -- this measures the raw fragility of every layer.
    """
    if ratios is None:
        ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    layers = dict(prunable_layers(model))
    curves = {}
    for name, module in layers.items():
        original = module.weight.data.clone()
        curve = []
        for r in ratios:
            mask = _layer_mask(original, r).to(original.device)
            module.weight.data = original * mask
            _, acc = evaluate(model, val_loader, device)
            curve.append((r, acc))
        module.weight.data = original  # restore before moving on
        curves[name] = curve
    return curves


def select_sparsity_dict(curves, baseline_acc, tolerance=0.02):
    """Pick, per layer, the highest sparsity whose accuracy stays within
    `tolerance` of the baseline. Produces a non-uniform per-layer budget.
    """
    chosen = {}
    for name, curve in curves.items():
        best_ratio = 0.0
        for ratio, acc in curve:
            if acc >= baseline_acc - tolerance:
                best_ratio = ratio
        chosen[name] = best_ratio
    return chosen


def _overall_sparsity(model, sparsity_dict):
    """Parameter-weighted overall sparsity implied by a per-layer dict."""
    sizes = {name: m.weight.numel() for name, m in prunable_layers(model)}
    total = sum(sizes.values())
    return sum(sparsity_dict.get(n, 0.0) * s for n, s in sizes.items()) / total


def select_for_target(model, curves, baseline_acc, target_sparsity):
    """Grow the allowed accuracy drop until the sensitivity-based per-layer
    budget reaches (roughly) `target_sparsity` overall.

    This lets us compare a sensitivity-guided budget against a uniform one at the
    *same* overall sparsity -- the only fair comparison.
    Returns (sparsity_dict, achieved_overall, tolerance_used).
    """
    chosen = select_sparsity_dict(curves, baseline_acc, 0.0)
    achieved, used = _overall_sparsity(model, chosen), 0.0
    for i in range(1, 61):
        tol = round(0.005 * i, 3)
        cand = select_sparsity_dict(curves, baseline_acc, tol)
        chosen, achieved, used = cand, _overall_sparsity(model, cand), tol
        if achieved >= target_sparsity:
            break
    return chosen, achieved, used


def select_scaled(model, curves, baseline_acc, target_sparsity, tol=0.10):
    """Sensitivity-shaped per-layer budget that hits `target_sparsity` EXACTLY.

    Steps:
    1. Take a per-layer "robustness" profile: the largest ratio each layer
       tolerates within `tol` accuracy of the baseline (bigger = more robust).
    2. Scale the whole profile by a single factor `alpha` (found by bisection) so
       that the parameter-weighted overall sparsity equals the target. Robust
       layers still get pruned harder, sensitive ones less -- but the total
       budget matches the uniform run precisely, so the comparison is fair.

    Returns (sparsity_dict, achieved_overall, alpha).
    """
    profile = select_sparsity_dict(curves, baseline_acc, tol)  # relative shape
    sizes = {name: m.weight.numel() for name, m in prunable_layers(model)}
    total = sum(sizes.values())

    def overall(alpha):
        return sum(min(0.99, alpha * profile[n]) * sizes[n]
                   for n in profile) / total

    lo, hi = 0.0, 50.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if overall(mid) < target_sparsity:
            lo = mid
        else:
            hi = mid
    alpha = (lo + hi) / 2.0
    chosen = {n: min(0.99, alpha * profile[n]) for n in profile}
    return chosen, overall(alpha), alpha
