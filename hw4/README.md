# HW4 — Neural Architecture Search with Hyperopt (VGG-style CNN / CIFAR-10)

## Task

Use [Hyperopt](https://github.com/hyperopt/hyperopt) to search a **discrete** space
of neural architectures for CIFAR-10:

- **Search space** over block operation, width multiplier, and activation function.
- **Search with TPE**, recording the best architecture (using **weight sharing**).
- **Plots:** running-best loss vs. trial number; accuracy vs. parameter count.
- **Retrain standalone:** train the best-found architecture from scratch and compare
  test accuracy and parameter count to a baseline.
- **Report:** what worked, what didn't, how informative the one-shot proxy was, and
  how to improve.

**Bonus:** a rank-correlation study that answers "how informative was the one-shot
proxy?" with a number (Kendall τ / Spearman ρ), not a hunch.

## Approach

- **Search space** (`src/search_space.py`) — a fixed 4-stage macro-skeleton (each
  stage = one searchable block + 2×2 max-pool), searched along exactly the three
  requested axes:
  - **block op** per stage ∈ {`conv3x3`, `dwsep` (MobileNet-v1 separable), `mbconv`
    (MobileNet-v2 inverted residual, expand ×3)},
  - **width multiplier** ∈ {0.5, 0.75, 1.0, 1.25} (one global channel scale),
  - **activation** ∈ {ReLU, ReLU6, SiLU, GELU, LeakyReLU} (one global choice).

  Space size = 3⁴ × 4 × 5 = **1620** architectures.

- **Weight sharing — one-shot supernet** (`src/supernet.py`) — the key to a cheap
  search. We train **one** over-parameterised network whose weights are shared across
  the whole space (the slimmable-net trick from HW3's NetAug, extended to a choice of
  operation per stage). A narrower design is the channel slice `W[:out, :in]` of the
  shared weight; every stage holds all three ops as parallel branches. Training uses
  **Single-Path One-Shot** (Guo et al., ECCV 2020): each step trains one uniformly
  random path, so no design is favoured a priori. Evaluating a candidate is then just
  "select its sub-path, recalibrate its BatchNorm on a few batches, score on val" —
  **no per-candidate training**.

- **Search — TPE** (`src/nas_search.py`) — Hyperopt's Tree-structured Parzen
  Estimator over the discrete `hp.choice` space; objective = one-shot proxy validation
  loss. Every trial logs #params + proxy val-accuracy (for the scatter) and the
  running-best loss (for the convergence curve). Revisited architectures are cached.

- **Retrain** (`run_retrain.py`) — the proxy only *ranks*; the deliverable number
  comes from training the selected design from scratch as an ordinary network. We
  compare test accuracy + parameter count against **two** baselines: the frozen HW1
  VGG11 (the same baseline used across HW1–HW3), and an in-space "default" design
  (all `conv3x3`, width 1.0, ReLU) trained with the identical recipe — so the gain is
  attributed to the search, not to a different architecture family.

- **Methodology** — the whole search and every intermediate number are on a held-out
  **validation** split; the **test** set is measured exactly once per final model;
  `inference_mode` in eval. Single-seed numbers are labelled as such.

## Reproduce

```bash
# 1-3: train the one-shot supernet, then TPE-search it (validation only)
python run_search.py     --data-dir ./data --supernet-epochs 100 --evals 250

# 4: retrain the best-found design from scratch; compare to baselines (test once)
python run_retrain.py    --baseline ../hw1/results/baseline.pt --epochs 120

# bonus: is the one-shot proxy trustworthy? (rank correlation, validation only)
python run_proxy_corr.py --top-k 8 --short-epochs 15

# quick wiring check (tiny configs, seconds)
python run_search.py --smoke && python run_retrain.py --smoke && python run_proxy_corr.py --smoke

# unit tests
python tests/test_search_space.py      # or: pytest tests/
python tests/test_supernet.py
```

Trained on gx10 (NVIDIA GB10, CUDA 13.0); the code is CPU-compatible. See
`requirements.txt`.

## Layout

```
src/search_space.py   discrete space + StandaloneNet builder + analytic param count
src/supernet.py       weight-sharing one-shot supernet (SPOS) + BN recalibration
src/nas_search.py     Hyperopt TPE search over the supernet + top-K helper
src/{model,data,engine,utils}.py   reused from HW1-HW3
src/plots.py          convergence / acc-vs-params / proxy-correlation plots
run_search.py         train supernet + TPE search (+ the two required plots)
run_retrain.py        retrain best design from scratch vs. baselines (test once)
run_proxy_corr.py     bonus: one-shot proxy vs. from-scratch rank correlation
tests/                search-space + supernet unit tests
results/              JSON + PNG (the supernet checkpoint is not committed)
```
