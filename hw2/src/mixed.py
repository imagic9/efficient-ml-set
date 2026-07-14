"""Bonus: per-layer bit-width sensitivity -> mixed-precision allocation.

Idea (same spirit as HW1's pruning sensitivity): some layers tolerate 2-bit
clustering with barely a scratch, others fall apart. So spending a flat bit-width
everywhere is wasteful -- give the fragile layers more centroids and the robust
ones fewer, at the SAME average bit budget as the uniform baseline. Only an
equal-size comparison tells us whether the allocation actually helped.
"""
import torch

from .engine import evaluate
from .kmeans_quant import KMeansQuantizer, quantizable_layers


@torch.no_grad()
def bit_sensitivity_scan(model_builder, state, device, val_loader,
                         bit_levels=(2, 3, 4), kmeans_iters=20,
                         keep_pruned_zeros=False):
    """For each layer, quantize ONLY that layer to each bit level (rest fp32)
    and record validation accuracy. Returns {layer: [(bits, val_acc), ...]}.

    keep_pruned_zeros: scan a pruned model by clustering only its non-zero weights
    (zeros stay pruned), so the sensitivity reflects the post-pruning network.
    """
    curves = {}
    names = [n for n, _ in quantizable_layers(model_builder().to(device))]
    for name in names:
        curve = []
        for b in bit_levels:
            model = model_builder().to(device)
            model.load_state_dict(state)
            q = KMeansQuantizer.quantize(model, {name: b}, iters=kmeans_iters,
                                         keep_pruned_zeros=keep_pruned_zeros)
            q.to(device)
            _, acc = evaluate(model, val_loader, device)
            curve.append((b, acc))
        curves[name] = curve
    return curves


def allocate_mixed_bits(model, curves, target_avg_bits,
                        bit_levels=(2, 3, 4)):
    """Greedy sensitivity-driven allocation at a fixed average bit budget.

    Start every layer at the lowest bit-width, then repeatedly upgrade the most
    sensitive layer (largest accuracy drop at the lowest bit-width) by one level,
    as long as the parameter-weighted average bit-width stays <= target. This
    spends the bit budget where it recovers the most accuracy.

    Returns (bits_dict, achieved_avg_bits).
    """
    levels = sorted(bit_levels)
    sizes = {name: m.weight.numel() for name, m in quantizable_layers(model)}
    total = sum(sizes.values())

    # sensitivity score = accuracy drop at the LOWEST bit-width (bigger = fragile)
    lo = levels[0]
    drop = {}
    for name, curve in curves.items():
        acc_lo = dict(curve)[lo]
        acc_hi = dict(curve)[levels[-1]]
        drop[name] = acc_hi - acc_lo         # how much this layer suffers at lo bits

    bits = {name: lo for name in sizes}
    avg = lambda: sum(bits[n] * sizes[n] for n in sizes) / total

    # candidates ordered most-sensitive first; upgrade until budget is spent
    order = sorted(sizes, key=lambda n: drop.get(n, 0.0), reverse=True)
    improved = True
    while improved:
        improved = False
        for name in order:
            i = levels.index(bits[name])
            if i + 1 >= len(levels):
                continue
            trial = dict(bits)
            trial[name] = levels[i + 1]
            trial_avg = sum(trial[n] * sizes[n] for n in sizes) / total
            if trial_avg <= target_avg_bits + 1e-9:
                bits = trial
                improved = True
    return bits, avg()
