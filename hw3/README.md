# HW3 — Self-Distillation with Compressed Models (VGG11 / CIFAR-10)

## Task

Implement **Knowledge Distillation (KD)** and use it to recover the accuracy that
compression takes away:

- **KD loss** — hard labels (cross-entropy) + soft labels (KL-divergence with
  temperature scaling): `L = α·CE + (1−α)·T²·KL(teacher_T ‖ student_T)`.
- **Self-distillation** — the **teacher** is our own uncompressed fp32 VGG11 from
  HW1/HW2 (same architecture, just not compressed).
- **Compressed student** — the same VGG11 compressed with the HW1/HW2 tooling
  (pruning, quantization, or both).
- **Iterative fine-tuning** — recover the student with the teacher's soft targets.

**Bonus (up to 3 pt):** a **NetAug** (Network Augmentation) training pipeline,
integrated with KD.

## Approach

- **Teacher:** the exact HW1 fp32 VGG11 baseline (90.71% test) is loaded and frozen,
  not retrained — the same starting point for every student.
- **KD loss** (`src/distill.py`): the Hinton (2015) combination. The T² factor
  rescales the soft-loss gradient (which otherwise shrinks like 1/T²) so a single
  learning rate works for any temperature. `α=1` collapses to pure CE, so the KD
  run and the CE-only baseline share **one code path** — the only thing that changes
  between them is the loss.
- **Central comparison — KD vs CE-only, matched:** for every compressed student we
  fine-tune **twice from the same compressed init**, with identical optimiser,
  schedule, epochs and seed. One run is plain cross-entropy recovery (the honest
  baseline), the other adds the teacher's soft targets. Any accuracy gap is then
  attributable to distillation alone.
- **Three student regimes** (`run_distill.py`):
  - **pruned 80%** — global unstructured magnitude pruning (HW1), pruner masks
    re-applied every step; SGD + cosine recovery.
  - **2-bit K-Means** — the hardest case (largest compression gap), recovered with
    centroid QAT (HW2) under a KD loss instead of CE.
  - **prune 80% + 4-bit** — the Deep Compression student: prune, then quantize the
    surviving non-zeros, recovered with centroid QAT + KD.
- **Ablations** (`run_kd_ablation.py`, the 2-bit student): temperature `T∈{1,2,4,8}`
  and mixing `α∈{0,…,1}` sweeps — `α=1` is the pure-CE reference line.
- **Methodology:** the teacher is loaded not retrained; every intermediate number
  (recovery curves, ablation sweeps) is on a held-out **validation** split; the
  **test** set is measured exactly once per final model (CE and KD are each one
  final model); `inference_mode` in eval.

## Reproduce

```bash
# core: KD vs CE-only recovery for the three compressed students
python run_distill.py     --baseline ../hw1/results/baseline.pt --data-dir ./data --out results

# ablations: temperature + alpha sweeps on the 2-bit student (validation only)
python run_kd_ablation.py --baseline ../hw1/results/baseline.pt --data-dir ./data --out results

# quick wiring check
python run_distill.py --smoke ...

# unit tests
python tests/test_distill.py      # or: pytest tests/
```

Trained on gx10 (NVIDIA GB10, CUDA 13.0); the code is CPU-compatible. See
`requirements.txt`.

## Layout

```
src/distill.py        KD loss (DistillLoss) + KD-aware training loop (kd_train)
src/qat.py            HW2 centroid QAT, extended with optional teacher/distill
src/{model,data,engine,prune,kmeans_quant,utils,plots}.py   reused from HW1/HW2
run_distill.py        core: KD vs CE-only for the three student regimes
run_kd_ablation.py    temperature + alpha sweeps (2-bit student, validation only)
tests/test_distill.py KD-loss unit tests
results/              JSON + PNG (checkpoints are not committed)
```
