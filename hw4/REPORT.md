# HW4 — Neural Architecture Search with Hyperopt · REPORT

**Course:** Efficient ML, SET University · **Dataset:** CIFAR-10

## Task
Search a discrete space of architectures (block op × width × activation) with
Hyperopt **TPE**, using **weight sharing**; plot running-best loss vs. trial and
accuracy vs. #params; retrain the best design from scratch vs. a baseline; report
how informative the one-shot proxy was.

## Setup
- **Search space** (`src/search_space.py`): 4 stages, each a block op ∈
  {conv3x3, dwsep, mbconv}; global width ∈ {0.5, 0.75, 1.0, 1.25}; global
  activation ∈ {ReLU, ReLU6, SiLU, GELU, LeakyReLU} → **1620** archs.
- **One-shot supernet** (`src/supernet.py`): shared weights at max width, all ops as
  parallel branches, trained with Single-Path One-Shot (uniform random path/step,
  100 epochs). Per-candidate BN recalibrated before scoring.
- **Search** (`src/nas_search.py`): Hyperopt TPE, objective = one-shot proxy val loss,
  250 trials (139 unique archs, 111 cached revisits).
- **Retrain** (120 epochs, seed 42): best design
  from scratch vs. frozen HW1 VGG11 and an in-space default (all conv3x3/1.0×/ReLU).
- **Methodology:** all search/intermediate numbers on validation; test measured once
  per final model; single-seed (labelled as such).

## Headline results (test — once per model)

| Model | Params | vs baseline | Test acc |
|---|---|---|---|
| Baseline VGG11 (HW1) | 9.49M | 1.0× | 90.71% |
| In-space default (conv3x3/1.0×/ReLU) | 0.96M | 9.8× smaller | 90.26% |
| **Found by search** (conv3x3/mbconv/conv3x3/mbconv, 1.0×, relu6) | **0.74M** | **12.8× smaller** | **90.69%** |

The searched design reaches **90.69%** — essentially baseline accuracy
(-0.02 pp) at **12.8× fewer parameters**, and +0.43 pp over
the in-space default.

## What the search found
TPE converged on a clear structural motif (top-15 distinct designs by proxy):
stage 0 → conv3x3 (conv3x3×15), stage 1 → mbconv (mbconv×15),
stage 2 → conv3x3×10, mbconv×5, stage 3 → mbconv×10, conv3x3×5; width
1.0×9, 1.25×4, 0.75×2; activation relu6×7, relu×4, gelu×3, leakyrelu×1. The worst designs (all-dwsep,
0.5× width) collapse to ~67% proxy accuracy and are reliably rejected.

## How informative was the one-shot proxy? (bonus)
Top-8 by proxy, each trained 15
epochs from scratch: **Kendall τ = 0.11, Spearman ρ = 0.19** (weak positive).
Among the top designs the proxy spans only 0.3 pp while real
short-training spans 4.1 pp — under shared weights the best
architectures are nearly indistinguishable, so fine ranking is noise.

**Two-part answer:** the proxy is **very** informative for *coarse filtering* (it nails
the good region and discards weak ops/widths) but **weak** for *picking the single
winner* (τ=0.11). Best used as a filter (1620 → a dozen), with the final choice
decided by short honest from-scratch training.

## What worked / what didn't / how to improve
- **Worked:** 12.8× smaller net at baseline accuracy; weight sharing made the
  search cheap; TPE + proxy find the good region confidently.
- **Didn't:** fine proxy ranking is weak (co-adaptation of shared weights flattens
  strong candidates); absolute proxy accuracy (~73%) is far below real (84–91%).
- **Improve:** train the supernet longer with fairness tricks (FairNAS / sandwich
  rule); larger val subset + more BN recal batches; widen the space (depth / kernel /
  stride); multi-seed mean±std (all numbers here are single-seed).

## Reproduce
```bash
python run_search.py     --supernet-epochs 100 --evals 250
python run_retrain.py    --baseline ../hw1/results/baseline.pt --epochs 120
python run_proxy_corr.py --top-k 8 --short-epochs 15
python tests/test_search_space.py && python tests/test_supernet.py
```
