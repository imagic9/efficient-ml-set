# HW2 — K-Means Quantization-Aware Training on VGG11 / CIFAR-10

## Task

Implement **K-Means (weight-sharing) quantization**: cluster each layer's weights
into K = 2^bits centroids (sub-8-bit, e.g. 2-bit → 4 centroids) and store a codebook
plus one small index per weight. Then implement **quantization-aware training (QAT)**
following the slide-36 diagram — run the forward pass with the clustered weights,
**pool the gradients by cluster index**, and update the centroids directly.

**Bonus (up to 3 pt):** a sensitivity analysis to apply **mixed-precision**
quantization, and combine quantization with the iterative pruning from HW1.

## Approach

- **Model:** the exact HW1 fp32 VGG11 baseline (90.71% test) is loaded, not
  retrained — a shared starting point for every model.
- **K-Means quantizer** (`src/kmeans_quant.py`): per-layer 1-D k-means (linear
  centroid init, as in Deep Compression), a frozen index map, and a `reconstruct()`
  that rebuilds `weight = codebook[index]`.
- **QAT** (`src/qat.py`): the centroids are the only trainable weights. After
  `loss.backward()` fills each weight's gradient, `pool_gradients()` pools those
  gradients by cluster index (mean per cluster) into the codebook gradient (the
  slide-36 step); Adam then steps the centroids. The conv/linear weights sit outside
  the optimizer, so their grads are zeroed each batch to stop them accumulating, and
  BatchNorm stats are recalibrated to the shifted weights before every validation.
- **Methodology:** the test set is measured once per final model; PTQ vs QAT
  curves, the bit-sensitivity scan and the mixed-vs-uniform comparison all use a
  held-out validation split. Since uniform quantization only exists at integer
  bit-widths, mixed precision is judged **against the uniform accuracy-vs-size Pareto
  line** — a mixed point counts as a win only if it sits above that line.
- **Bonus A — mixed precision:** a per-layer bit-width sensitivity scan drives a
  greedy allocation (`src/mixed.py`) at a deliberately lossy ~2.5-bit budget (where
  uniform still loses accuracy); the mixed model reaches uniform-3bit accuracy at a
  ~2.5-bit size — above the uniform Pareto line.
- **Bonus B — improve the pruning result:** take HW1's 80%-sparse model and quantize
  its surviving non-zeros with mixed precision; it beats uniform-precision quant of
  the same pruned net and gives the best overall point (≈90.5% at ~19× smaller),
  sizes reported in a sparse+quantized format.

## Reproduce

```bash
# core: bit-width sweep {2,3,4} with PTQ vs QAT
python run_kmeans.py   --baseline ../hw1/results/baseline.pt --data-dir ./data --out results

# bonus: mixed-precision (Pareto) + pruning+quantization pipeline
python run_mixed.py    --baseline ../hw1/results/baseline.pt \
    --pruned ../hw1/results/iterative_final.pt --data-dir ./data --out results

# ablation (pooling sum/mean under SGD & Adam, adapt_extras F/T)
python run_ablation.py --baseline ../hw1/results/baseline.pt --data-dir ./data --out results

# error bars: 3-seed mean±std for the mixed-vs-uniform claim
python run_seeds.py    --seeds 0 1 2 --baseline ../hw1/results/baseline.pt \
    --pruned ../hw1/results/iterative_final.pt --data-dir ./data --out results

python build_notebook.py results     # assemble the notebook + REPORT
python -m pytest tests/              # unit tests (pooling + zero-preservation)

# quick wiring check (tiny epochs):  add --smoke to any runner
```

## Files

- `HW2_KMeans_Quantization.ipynb` (+ `.html`) — main notebook (Ukrainian).
- `REPORT.md` — short written report.
- `src/` — `kmeans_quant`, `qat`, `mixed`, `plots` (new) + `data`, `model`, `engine`,
  `prune`, `sensitivity`, `utils` (reused from HW1).
- `run_kmeans.py`, `run_mixed.py`, `run_ablation.py`, `run_seeds.py`, `build_notebook.py`
  — runners + notebook generator.
- `tests/` — unit tests (scatter_add pooling, pruned-zero preservation).
- `requirements.txt` — package versions (gx10 CUDA build noted inside).
- `results/` — metrics (JSON) and figures (PNG). Checkpoints are not committed.
- `task/` — instructor materials (gitignored, local only).
