"""Unit tests for the two easy-to-get-wrong pieces of the quantizer:

  * gradient pooling by cluster index (scatter_add, sum vs mean);
  * zeros staying pinned through reconstruct() when quantizing a pruned model.

Run with pytest, or directly:  python tests/test_kmeans_quant.py
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.kmeans_quant import KMeansQuantizer, quantizable_layers  # noqa: E402


def _layer_name(model):
    return quantizable_layers(model)[0][0]


def test_pool_gradients_matches_scatter_add():
    """codebook.grad must equal the per-cluster sum (mode='sum') or mean of the
    weight gradients, computed independently with scatter_add."""
    torch.manual_seed(0)
    lin = nn.Linear(8, 4, bias=False)          # 32 weights
    model = nn.Sequential(lin)
    name = _layer_name(model)
    q = KMeansQuantizer.quantize(model, {name: 2}, iters=30)   # K=4 centroids
    idx = q.indices[name]
    k = q.codebooks[name].numel()

    grad = torch.randn_like(lin.weight)
    flat_idx = idx.reshape(-1)

    expected_sum = torch.zeros(k)
    expected_sum.scatter_add_(0, flat_idx, grad.reshape(-1))
    counts = torch.zeros(k)
    counts.scatter_add_(0, flat_idx, torch.ones(flat_idx.numel()))
    expected_mean = expected_sum / counts.clamp(min=1.0)

    lin.weight.grad = grad.clone()
    q.pool_gradients(model, mode="sum")
    assert torch.allclose(q.codebooks[name].grad, expected_sum, atol=1e-5), "sum pooling wrong"

    lin.weight.grad = grad.clone()
    q.pool_gradients(model, mode="mean")
    assert torch.allclose(q.codebooks[name].grad, expected_mean, atol=1e-5), "mean pooling wrong"


def test_reconstruct_preserves_pruned_zeros():
    """When a pruned layer is quantized with keep_pruned_zeros, the originally-zero
    weights must stay exactly zero -- both right after quantize() and after the
    codebook is perturbed and reconstruct() is called again."""
    torch.manual_seed(0)
    lin = nn.Linear(10, 6, bias=False)
    model = nn.Sequential(lin)
    with torch.no_grad():                       # prune ~half by magnitude
        thr = lin.weight.abs().median()
        lin.weight[lin.weight.abs() <= thr] = 0.0
    zero_mask = lin.weight == 0
    assert zero_mask.any() and not zero_mask.all()

    name = _layer_name(model)
    q = KMeansQuantizer.quantize(model, {name: 2}, iters=30, keep_pruned_zeros=True)
    assert torch.all(lin.weight[zero_mask] == 0), "zeros not preserved after quantize"

    q.codebooks[name].data += 5.0               # shove every centroid far from 0
    q.reconstruct(model)
    assert torch.all(lin.weight[zero_mask] == 0), "zeros not preserved after reconstruct"
    # the surviving non-zeros should now be non-zero (they follow the shifted codebook)
    assert torch.all(lin.weight[~zero_mask] != 0), "non-zero weights unexpectedly zero"


if __name__ == "__main__":
    test_pool_gradients_matches_scatter_add()
    test_reconstruct_preserves_pruned_zeros()
    print("all tests passed")
