# Experiment — few-shot "add a new target animal" module

> An optional extension to the Core project. It answers a product question the Core does not:
> **can a new target animal be added to the deployed system without retraining the backbone?**
> Nothing here changes the Core (`final_project/`) or the `v1.0-final` release; it is additive
> and evaluated independently.

## Question

The Core ships a 16-class MobileNetV2 (final model **M2**, INT8 QAT) whose selectable targets
are fixed at training time (bobcat is index 3). This experiment asks whether a **new** animal —
**tiger** — can be added by reusing the **frozen** backbone as a feature extractor, at negligible
on-device cost, and how much supervision that takes.

Backbone: the deployed **M0** MobileNetV2, frozen. Embedding = `features → global-average-pool →
1280-d`, L2-normalised. Reuses the exact Core preprocessing (256×192 letterbox, ImageNet
normalisation) so the feature space matches the deployed pipeline.

## Data

- **Positives** — **ATRW** (Amur Tiger Re-identification in the Wild; the dataset from *TigerNet*,
  arXiv 1909.01122): 2762 detection whole-frames (1920×1080) and 3392 re-id crops.
- **Negatives** — Core **`cis_val_clean`** (3214 CCT camera-trap frames, per-class labels), i.e.
  the background the device normally sees. Held-out test opened once; thresholds calibrated on a
  validation split (Core methodology: val for tuning, test once).

## Three ways to produce the new "tiger" head

All three yield the **same artefact** — one extra row (1280 weights + bias) for the classifier —
so on-device integration is identical; they differ only in how the row is computed offline.

| ID | Method | Supervision | Script |
|---|---|---|---|
| **A1** | Feature-space **prototype** (mean of support embeddings, cosine score) | ~K tiger images, **no training** | `code/approach1_prototype.py` |
| **A2** | **Linear head** (logistic regression on frozen embeddings) | ~K tiger images + background | `code/approach2_head.py` |
| **A3** | **Distillation** from an ImageNet teacher that already knows "tiger" | **0 manual tiger labels** | `code/approach3_distill.py` |

`code/emb_build.py` builds the embeddings; `code/compare.py` produces the comparison figure and
`results/comparison_summary.json`.

## Results (ATRW detection whole-frames vs CCT background; held-out test)

| Approach | Manual tiger labels | ROC-AUC (K=10) | Recall @ 5% false-fire | Pi marginal cost |
|---|---:|---:|---:|---:|
| A1 prototype | ~10 imgs | 0.974 | 0.886 | 2.14 µs/frame |
| **A2 linear head** | ~10 imgs + bg | **0.997** | **0.987** | 1.07 µs/frame |
| A3 distillation | **0** | 0.995 | 0.995 | 1.07 µs/frame |

- **A2** is best at matched few-shot K; **A3** needs no manual labels and the distilled student
  (0.995) beats its own teacher (zero-shot 0.957); **A1** needs no training at all.
- **Domain-vs-appearance probe** (A1, K=10): tiger vs CCT felids (cat + bobcat) AUC **0.952** — the
  signal is genuine tiger appearance, not an ATRW-vs-CCT domain shortcut.

## On-device latency (real Raspberry Pi CM5)

`code/bench_head.c` (static `-mcpu=cortex-a76`, governor `performance`, `throttled=0x0`, one core,
200k iters × 3): the marginal cost of adding a target is **1–2 µs/frame** — ~0.005–0.01 % of the
21.61 ms Core inference. A 64-target catalogue costs 68.5 µs (0.32 %). No FPS impact.

## Recommended integration

Embed the **A2 linear-head form** as a **parallel FP32 side-head** reading the already-computed
1280-d embedding, leaving the INT8 16-class path (and all its parity/frozen-test/Pi evidence)
bit-identical. Populate the head via A1 (instant), A2 (with a few labels), or A3 (no labels).
Two caveats before shipping: (1) recompute the head on **M2's** INT8 embedding space, not M0 FP32;
(2) re-run the P1–P4 parity gates for the new graph output.

## Honest limits

- Tiger is a **visually easy** target; a camouflaged animal (like the Core's own bobcat,
  trans-domain F2 ≈ 0.38) would score far lower — the **mechanism** generalises, not the 0.99.
- Partial ATRW(zoo)-vs-CCT(US traps) domain gap remains (the felid probe mitigates, not eliminates).
- Within-clip frame correlation in the ATRW random split may slightly inflate numbers.
- Latency figures are the marginal head only (float32 upper bound); the full pipeline is 21.61 ms.

## Reproduce (gx10)

Datasets + embeddings live on gx10 under `~/efficientml/atrw/` (ATRW tarballs, `emb/*.npz`); the
frozen backbone is `results/training/c2/c2_m0_fp32_seed42_*/best.pt`. Run `emb_build.py` →
`approach{1,2,3}_*.py` → `compare.py`. The `code/bench_head.c` binary runs on the Pi via
`ssh cm5-pi` (from gx10).
