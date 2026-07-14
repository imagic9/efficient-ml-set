# HW1 — VGG11 Iterative Pruning on CIFAR-10

## Task

Take VGG11 from `torchvision`, train it on CIFAR-10, then remove ~80% of the
weights while keeping accuracy as high as possible. Compare pruning everything at
once (one-shot) against gradual iterative pruning with fine-tuning between steps.
Bonus: sensitivity analysis to improve the results.

## Approach

- **Model:** `vgg11_bn` from torchvision, convolutional backbone unchanged, with a
  compact classifier head for CIFAR-10 (32×32, 10 classes).
- **Fine-grained pruning:** unstructured magnitude pruning with persistent masks
  re-applied after every optimizer step.
- **Methodology:** the test set is measured once per final model; all intermediate
  curves and method comparisons use a held-out validation split.
- **Bonus 1 (unstructured):** sweep comparing three per-layer allocation
  strategies — uniform-per-layer, global-magnitude, sensitivity-guided.
- **Bonus 2 (structured):** channel pruning with `torch-pruning` that physically
  removes filters (real MAC/param reduction), uniform vs sensitivity-guided.

## Results (test accuracy)

| Model | Sparsity | Test acc |
|-------|----------|----------|
| Baseline (dense) | 0% | 90.71% |
| One-shot 80% (after fine-tuning) | 80% | 90.72% |
| Iterative 80% (global magnitude) | 80% | 90.79% |
| Bonus @95%: uniform-per-layer | 95% | 88.13% |
| Bonus @95%: sensitivity-guided | 95% | 89.94% |

At moderate sparsity the global-magnitude threshold is near-optimal for
unstructured pruning; sensitivity-guided allocation pays off at aggressive
sparsity (95%), beating the naive uniform baseline by ~1.8 pp. For structured
pruning, an accuracy-only sensitivity budget is MAC-blind — a MAC-aware allocation
is the natural next step (and directly relevant to the final project).

## Reproduce

```bash
python run_all.py --data-dir ./data --out results        # baseline, one-shot, iterative, sweep
python run_structured.py --data-dir ./data --out results  # structured (channel) pruning bonus
python run_confirm.py --data-dir ./data --out results --target 0.95  # test-once confirmation
python build_notebook.py results                          # assemble the notebook + REPORT
```

## Files

- `HW1_VGG11_Pruning.ipynb` (+ `.html`) — main notebook (Ukrainian).
- `REPORT.md` — short written report.
- `src/` — `data`, `model`, `engine`, `prune`, `sensitivity`, `structured`, `plots`, `utils`.
- `run_*.py`, `build_notebook.py` — experiment runners and notebook generator.
- `results/` — metrics (JSON) and figures (PNG). Checkpoints are not committed.
