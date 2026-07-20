# Final optimized model decision (Pi Day 3 / PLAN F3, DESIGN §8.4–§8.5)

**Decision: the final optimized model is `M2` (INT8 QAT).** Frozen deployment config:
`threads=1`, full JPEG decode, `ORT_ENABLE_ALL`, CPU arena on, fused preprocess, 256×192.

Decided 2026-07-20 from **real Raspberry Pi CM5** validation evidence (F2) plus the frozen
validation-accuracy / MACs / size record (`comparison.jsonl`, `pre_pi_shortlist.md`). Test
labels stayed sealed (DESIGN §5.4) — nothing below reads a test manifest.

## The candidates (validation, deployment ORT; Pi latency from F2 @ performance governor)

| model | transform | selection_score (mean bobcat F2 @0.5) | cis F2 @op | trans F2 @op | MACs | size | Pi p50 @t1 | Pi p50 @t3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| M0 | FP32 baseline | 0.3663 | 0.6355 | 0.0965 | 293.4 M | 8.95 MB | 48.23 ms | 34.73 ms |
| **M2** | **INT8 QAT** | **0.3832** | 0.640 | 0.0976 | 293.4 M | 2.54 MB | 21.30 ms | 20.28 ms |
| M4 | pruned + QAT INT8 | 0.3730 | 0.6615 | 0.0901 | 205.6 M | 2.01 MB | 17.54 ms | 13.81 ms |

M1 (PTQ, 0.3527) and M3 (pruned-FP32, 0.3583) were dropped pre-Pi (dominated; `pre_pi_shortlist.md`).

## Applying the pre-registered rule (DESIGN §8.4)

> *"If M4 is slower or less accurate than M2, M2 remains the final model. A more complicated
> stack does not win by default."*

The only real contest is **M2 vs M4** (both INT8, both far past real-time on the Pi):

- **Speed:** M4 is faster (17.54 vs 21.30 ms @ t1; 13.81 vs 20.28 ms @ t3) and smaller
  (2.01 vs 2.54 MB, 205.6 vs 293.4 M MACs). M4 wins this axis.
- **Accuracy (the pre-registered metric):** the primary accuracy figure is the **mean bobcat
  F2 selection score** (DESIGN §8.5 step 3 / §6 checkpoint score). **M2 = 0.3832 > M4 = 0.3730.**
  M4 is *less accurate* on the pre-registered metric.

§8.4 fires on **"slower OR less accurate."** M4 is not slower, but it **is less accurate** on the
governing metric, so the rule keeps **M2**. §8.5 step 5 agrees independently: the two are within
the C1a bootstrap noise band (~0.035 F2), i.e. **statistically tied on accuracy**, and when tied
the simpler transformation is preferred — **QAT alone (M2) over pruning+QAT (M4)**. A more
complicated stack does not win by default, and M4's speed/size edge does not buy an accuracy gain
to justify the extra pruning stage.

**Why not chase M4's speed anyway.** Both models clear real-time on the Pi with large margins
(M2 47 FPS, M4 57–72 FPS vs the DESIGN §11 aspirational ≥10 FPS / ≤100 ms p95 target). Latency is
not the binding constraint here; catching the animal is. So spending the pre-registered accuracy
tiebreak on the simpler model is the disciplined call, and M4's aggressive result is still fully
reported (§ below) rather than discarded.

## Per-domain nuance (recorded, not decisive)

At the calibrated operating points M4 is actually a hair better on the **cis** (in-distribution)
domain (capture 0.86 vs 0.84, F2 0.6615 vs 0.640) and M2 a hair better on **trans** (capture 0.17
vs 0.155, F2 0.0976 vs 0.0901). These offset and both sit inside the ±noise band; neither overturns
the pre-registered selection_score ranking. All models remain `recall_floor_infeasible` (no
threshold meets the 90 % sequence-balanced bobcat recall floor within the 5 % per-domain false-fire
budget) — honestly reported, unchanged by this choice (DESIGN §6.3).

## What ships (the freeze)

- **Model:** `M2` — `results/optimize/m2_qat/lr5e-5/model.onnx`
  (sha256 `499bc3ec…45ecc`, 2.54 MB, INT8 QDQ, opset 17).
- **Policy:** `bobcat_m2_qat_lr5e-5_v1` (threshold 0.6504, status `recall_floor_infeasible`),
  bound by sha256 to the class map and model.
- **Runtime:** `threads=1`, full decode, `ORT_ENABLE_ALL`, arena on, fused preprocess, 256×192 —
  the exact config every P1–P4 parity gate validated, so the F4 Pi-vs-gx10 parity check is
  config-matched.
- **Baseline for the comparison:** `M0` FP32 at the identical `threads=1` config.

## Headline result

**M0 FP32 baseline → M2 INT8 QAT, both threads=1, on the Raspberry Pi CM5:
48.23 → 21.30 ms end-to-end (20.7 → 47.0 FPS) = 2.26× faster, model 3.5× smaller
(8.95 → 2.54 MB), at statistically-equivalent validation accuracy.**

## Measured-but-not-selected (reported in full for the write-up)

- **M4 (pruned + QAT)** reaches 17.54 ms / 57 FPS @ t1 and **13.81 ms / 72.4 FPS @ threads=3**
  (3.50× over the baseline) at 2.01 MB / 205.6 M MACs — the most aggressive optimization we
  produced. Not selected only because it did not improve accuracy over M2 (§8.4). Its numbers are
  the project's demonstration that structured pruning + QAT compounds.
- **Threading** on the 4-core A76 helps every model (threads=3 optimal); it is parity-safe for the
  INT8 winner (int32 accumulation is thread-invariant) but was **not folded into the frozen
  artifact** — the freeze stays at the parity-validated threads=1 so the F4 comparison is clean.
  The threaded numbers are reported as an available inference optimization (F2).

## Post-freeze action

Confirmation seeds **17 / 73** retrain the selected transformation (M2 QAT from the frozen M0
checkpoint, lr 5e-5, 6 epochs) on gx10 in the background to measure seed variability. They are
**non-gating** (DESIGN §8.5 / PLAN F3): they never replace the seed-42 deployment artifact and must
not gate this freeze, any trial day, or Gate F; they must finish before Gate G.
