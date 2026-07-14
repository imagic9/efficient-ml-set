"""K-Means (weight-sharing) quantization + centroid fine-tuning (QAT).

This is the "Deep Compression" style of quantization (Han et al., 2016). Instead
of a uniform grid, we cluster each layer's weights into K = 2**bits centroids and
replace every weight by its cluster centroid. We then store two things per layer:

  * a codebook  -- K float32 centroid values,
  * an index map -- one log2(K)-bit index per weight, pointing into the codebook.

So a 2-bit layer keeps only 4 distinct weight values; a 4-bit layer keeps 16.

QAT (the slide-36 diagram): the cluster assignment (index map) is frozen after the
initial k-means. During fine-tuning we run a normal forward/backward, then *pool*
the per-weight gradients by cluster index (sum of gradients of all weights sharing
a centroid) and take a step on the centroids themselves. The weights are then
re-materialised from the updated codebook. Only the K centroids per layer move --
this is what lets a 2-bit model claw back accuracy.

Note on autograd: reparametrising weight = codebook[index] would make PyTorch do
the exact same gradient pooling for free (gather's backward is scatter_add). We
implement the pooling explicitly instead, so the mechanism from the diagram is
visible and auditable rather than hidden inside autograd.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def quantizable_layers(model):
    """Conv2d and Linear layers -- the ones whose weights we cluster."""
    return [(name, m) for name, m in model.named_modules()
            if isinstance(m, (nn.Conv2d, nn.Linear))]


# --------------------------------------------------------------------------- #
# 1-D k-means over a layer's weight values                                    #
# --------------------------------------------------------------------------- #
def _linear_init(values: torch.Tensor, k: int) -> torch.Tensor:
    """K centroids spread linearly from min to max weight.

    Deep Compression found linear initialisation beats density/random init:
    large-magnitude weights matter most but are rare, so a density-based init
    starves them of centroids. Linear init keeps centroids on the wide tails.
    """
    lo, hi = values.min(), values.max()
    if torch.isclose(lo, hi):
        return lo.repeat(k).clone()
    return torch.linspace(lo.item(), hi.item(), k, device=values.device)


@torch.no_grad()
def kmeans_1d(values: torch.Tensor, k: int, iters: int = 30,
              nonzero_only: bool = False):
    """Cluster a flat tensor of weights into k centroids (Lloyd's algorithm).

    Returns (centroids[k], labels[N]) where labels index into centroids for every
    input value. When `nonzero_only` is set, exactly-zero weights (e.g. pruned
    ones) are excluded from clustering and reported with label -1 so the caller
    can keep them pinned at zero.
    """
    flat = values.flatten()
    if nonzero_only:
        active = flat != 0
    else:
        active = torch.ones_like(flat, dtype=torch.bool)

    labels = flat.new_full(flat.shape, -1, dtype=torch.long)
    pts = flat[active]
    if pts.numel() == 0:                       # whole layer pruned away
        return flat.new_zeros(k), labels.view_as(values)

    k = min(k, pts.numel())                    # can't have more clusters than points
    centroids = _linear_init(pts, k).clone()

    for _ in range(iters):
        # assign: nearest centroid per point
        dist = (pts.unsqueeze(1) - centroids.unsqueeze(0)).abs()
        lab = dist.argmin(dim=1)
        # update: mean of the points in each cluster (empty clusters stay put)
        new_centroids = centroids.clone()
        for c in range(k):
            sel = lab == c
            if sel.any():
                new_centroids[c] = pts[sel].mean()
        if torch.allclose(new_centroids, centroids):
            centroids = new_centroids
            break
        centroids = new_centroids

    # final assignment, written back to the full-length label vector
    dist = (pts.unsqueeze(1) - centroids.unsqueeze(0)).abs()
    labels[active] = dist.argmin(dim=1)
    return centroids, labels.view_as(values)


# --------------------------------------------------------------------------- #
# The quantizer: holds one codebook + index map per layer                     #
# --------------------------------------------------------------------------- #
class KMeansQuantizer:
    """Weight-sharing quantizer with explicit gradient-pooling QAT.

    `bits_dict` maps layer name -> bit-width. A layer absent from the dict (or
    with bits >= 32) is left in full precision. After `quantize()` the model's
    weights are already the clustered values; call `reconstruct()` before each
    forward during QAT and `pool_gradients()` + optimiser step after backward.
    """

    def __init__(self, codebooks: dict, indices: dict, bits: dict):
        self.codebooks = codebooks   # name -> nn.Parameter[K]  (trainable centroids)
        self.indices = indices       # name -> LongTensor, weight-shaped (-1 = pinned 0)
        self.bits = bits             # name -> int
        self.counts = {}             # name -> FloatTensor[K], weights per cluster (cached)
        for name, idx in indices.items():
            k = codebooks[name].numel()
            valid = idx[idx >= 0].reshape(-1)
            cnt = torch.zeros(k, device=valid.device)
            cnt.scatter_add_(0, valid, torch.ones_like(valid, dtype=cnt.dtype))
            self.counts[name] = cnt.clamp(min=1.0)          # avoid div-by-zero

    # --- construction ------------------------------------------------------ #
    @classmethod
    @torch.no_grad()
    def quantize(cls, model, bits_dict: dict, iters: int = 30,
                 keep_pruned_zeros: bool = False, default_bits: int = 32):
        """Fit a codebook per quantizable layer and write clustered weights back.

        keep_pruned_zeros: cluster only the non-zero weights and keep zeros pinned
        (used when quantizing an already-pruned model -- the Deep Compression
        pipeline). Otherwise every weight participates in clustering.
        """
        codebooks, indices, bits = {}, {}, {}
        for name, module in quantizable_layers(model):
            b = bits_dict.get(name, default_bits)
            if b >= 32:                        # leave this layer in fp32
                continue
            k = 2 ** b
            w = module.weight.data
            centroids, labels = kmeans_1d(w, k, iters=iters,
                                          nonzero_only=keep_pruned_zeros)
            codebooks[name] = nn.Parameter(centroids.clone())
            indices[name] = labels
            bits[name] = b
        q = cls(codebooks, indices, bits)
        q.reconstruct(model)
        return q

    # --- QAT plumbing ------------------------------------------------------ #
    @torch.no_grad()
    def reconstruct(self, model):
        """Write weight = codebook[index] into every quantized layer.

        Pinned entries (index == -1, i.e. pruned weights) stay exactly zero.
        """
        layers = dict(quantizable_layers(model))
        for name, idx in self.indices.items():
            cb = self.codebooks[name].data.to(idx.device)
            idx_safe = idx.clamp(min=0)                    # -1 -> 0, masked next
            recon = cb[idx_safe]
            recon = torch.where(idx < 0, torch.zeros_like(recon), recon)
            layers[name].weight.data.copy_(recon)

    @torch.no_grad()
    def pool_gradients(self, model, mode: str = "mean"):
        """Slide-36 step: pool per-weight grads into per-centroid grads.

        For a centroid c shared by the weights in cluster S_c:

            sum : g_c = Σ_{i in S_c} dL/dW_i          (the literal diagram)
            mean: g_c = (1/|S_c|) Σ_{i in S_c} dL/dW_i (what we use)

        The `sum` form is the exact gradient of the reparametrisation W_i = C_c, so
        it is "correct". But cluster sizes span orders of magnitude (a 2-bit conv
        layer packs ~500k weights into 4 clusters, the classifier only ~1k), so
        with `sum` a single learning rate makes each centroid's step scale with its
        cluster size -- big clusters overshoot, small ones stall (plain SGD diverges
        within a couple of epochs; see the ablation). `mean` normalises that out --
        same descent direction, well-conditioned step regardless of cluster size --
        so it is robust across optimisers. Pinned (pruned) weights carry index -1
        and are skipped. Result lands in each codebook Parameter's .grad.
        """
        layers = dict(quantizable_layers(model))
        for name, idx in self.indices.items():
            wgrad = layers[name].weight.grad
            cb = self.codebooks[name]
            if wgrad is None:
                cb.grad = None
                continue
            k = cb.numel()
            pooled = torch.zeros(k, device=cb.device, dtype=cb.dtype)
            valid = idx >= 0
            pooled.scatter_add_(0, idx[valid].reshape(-1).to(cb.device),
                                wgrad[valid].reshape(-1).to(cb.dtype))
            if mode == "mean":
                pooled = pooled / self.counts[name].to(cb.device)
            cb.grad = pooled

    def centroid_parameters(self):
        """The trainable centroids -- hand these to the optimiser for QAT."""
        return list(self.codebooks.values())

    def to(self, device):
        for cb in self.codebooks.values():
            cb.data = cb.data.to(device)
        self.indices = {n: i.to(device) for n, i in self.indices.items()}
        self.counts = {n: c.to(device) for n, c in self.counts.items()}
        return self


# --------------------------------------------------------------------------- #
# Model-size accounting                                                        #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def model_size_bits(model, quantizer: KMeansQuantizer | None = None) -> int:
    """Total storage in bits.

    fp32 baseline: every parameter costs 32 bits. For a quantized layer we count
    the codebook (K * 32 bits) plus the index map (numel * bits). Pruned zeros are
    conservatively still counted as indices here (a real sparse format would store
    only non-zeros + positions; we note that separately where it matters).
    """
    q_names = set(quantizer.bits) if quantizer else set()
    total = 0
    layers = dict(quantizable_layers(model))
    for name, m in model.named_parameters():
        total += m.numel() * 32                        # start: everything fp32
    for name in q_names:
        w = layers[name].weight
        total -= w.numel() * 32                         # undo the fp32 count...
        b = quantizer.bits[name]
        total += (2 ** b) * 32 + w.numel() * b          # ...replace with codebook + indices
    return total


@torch.no_grad()
def sparse_quant_size_bits(model, quantizer: KMeansQuantizer) -> int:
    """Storage in bits for a PRUNED + quantized model (Deep Compression format).

    A quantized layer that also contains pruned zeros is stored as: the codebook
    (K * 32 bits) + one index (`bits`) per NON-ZERO weight + a 1-bit-per-weight
    position bitmap so the decoder knows where the non-zeros sit. This rewards the
    combination of pruning and quantization, unlike the dense `model_size_bits`.
    """
    layers = dict(quantizable_layers(model))
    total = 0
    q_names = set(quantizer.bits)
    for name, m in model.named_parameters():
        total += m.numel() * 32
    for name in q_names:
        w = layers[name].weight
        n = w.numel()
        nnz = int((w != 0).sum().item())
        b = quantizer.bits[name]
        total -= n * 32                               # undo fp32 count
        total += (2 ** b) * 32 + nnz * b + n * 1      # codebook + nz indices + bitmap
    return total


@torch.no_grad()
def compression_report(model, quantizer: KMeansQuantizer):
    """Human-readable size summary vs the fp32 model."""
    fp32 = sum(p.numel() for p in model.parameters()) * 32
    q = model_size_bits(model, quantizer)
    avg_bits = _avg_bits(model, quantizer)
    return {
        "fp32_MB": fp32 / 8 / 1e6,
        "quant_MB": q / 8 / 1e6,
        "compression_x": fp32 / q,
        "avg_bits_per_quantized_weight": avg_bits,
    }


@torch.no_grad()
def _avg_bits(model, quantizer: KMeansQuantizer) -> float:
    """Parameter-weighted average bit-width over the quantized layers only."""
    layers = dict(quantizable_layers(model))
    num = den = 0
    for name, b in quantizer.bits.items():
        n = layers[name].weight.numel()
        num += b * n
        den += n
    return num / max(den, 1)
