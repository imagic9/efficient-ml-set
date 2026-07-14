"""Fine-grained (unstructured) magnitude pruning with persistent masks.

We zero out the weights with the smallest absolute value. This granularity can
reach very high sparsity (~80%) with a small accuracy cost, unlike structured
channel pruning at the same ratio. The trade-off: unstructured sparsity needs a
sparse kernel / special hardware to actually speed inference up -- discussed in
the report.
"""
import torch
import torch.nn as nn


def prunable_layers(model):
    """Conv2d and Linear layers, whose `weight` tensors we prune."""
    return [(name, m) for name, m in model.named_modules()
            if isinstance(m, (nn.Conv2d, nn.Linear))]


def _layer_mask(weight: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Keep the (1 - sparsity) fraction of weights with the largest magnitude."""
    if sparsity <= 0:
        return torch.ones_like(weight)
    n = weight.numel()
    k = int(round(sparsity * n))  # number of weights to zero out
    if k <= 0:
        return torch.ones_like(weight)
    if k >= n:
        return torch.zeros_like(weight)
    threshold = torch.kthvalue(weight.abs().flatten(), k).values
    return (weight.abs() > threshold).float()


class FineGrainedPruner:
    """Holds one binary mask per prunable layer and re-applies them on demand."""

    def __init__(self, masks: dict):
        self.masks = masks

    # --- constructors -----------------------------------------------------
    @classmethod
    def from_uniform(cls, model, sparsity: float):
        """Same target sparsity for every layer (magnitude threshold per layer)."""
        masks = {name: _layer_mask(m.weight.data, sparsity)
                 for name, m in prunable_layers(model)}
        return cls(masks)

    @classmethod
    def from_dict(cls, model, sparsity_dict: dict):
        """Per-layer target sparsity (used with sensitivity analysis)."""
        masks = {name: _layer_mask(m.weight.data, sparsity_dict.get(name, 0.0))
                 for name, m in prunable_layers(model)}
        return cls(masks)

    @classmethod
    def from_global(cls, model, sparsity: float):
        """One global magnitude threshold across all prunable weights.

        Layers keep more or fewer weights depending on their weight magnitudes,
        which usually beats a uniform per-layer ratio.
        """
        all_weights = torch.cat([m.weight.data.abs().flatten()
                                 for _, m in prunable_layers(model)])
        n = all_weights.numel()
        k = int(round(sparsity * n))
        if k <= 0:
            threshold = -1.0
        elif k >= n:
            threshold = float("inf")
        else:
            threshold = torch.kthvalue(all_weights, k).values.item()
        masks = {name: (m.weight.data.abs() > threshold).float()
                 for name, m in prunable_layers(model)}
        return cls(masks)

    # --- usage ------------------------------------------------------------
    @torch.no_grad()
    def apply(self, model):
        layers = dict(prunable_layers(model))
        for name, mask in self.masks.items():
            w = layers[name].weight
            w.data.mul_(mask.to(w.device))


@torch.no_grad()
def model_sparsity(model) -> float:
    """Fraction of zero weights among all Conv/Linear weight tensors."""
    zeros = total = 0
    for _, m in prunable_layers(model):
        w = m.weight.data
        zeros += (w == 0).sum().item()
        total += w.numel()
    return zeros / max(total, 1)


@torch.no_grad()
def layerwise_sparsity(model) -> dict:
    out = {}
    for name, m in prunable_layers(model):
        w = m.weight.data
        out[name] = (w == 0).sum().item() / w.numel()
    return out


def geometric_schedule(final_sparsity: float, steps: int):
    """Cumulative sparsity targets that prune a constant fraction of the
    *remaining* weights each step, reaching `final_sparsity` after `steps`.

    e.g. final=0.8, steps=5 -> [0.275, 0.475, 0.620, 0.725, 0.800]
    """
    keep = (1.0 - final_sparsity) ** (1.0 / steps)  # fraction kept per step
    return [round(1.0 - keep ** t, 4) for t in range(1, steps + 1)]
