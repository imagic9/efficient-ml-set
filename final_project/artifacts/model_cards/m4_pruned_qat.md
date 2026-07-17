# Model card — M4, structured-pruned + INT8 QAT (c30 + lr5e-5)

**One sentence:** the M3 c30 pruned architecture with the validated M2 QAT
recipe applied — the **smallest artifact on the whole ladder** (2,014,806 B), a
real INT8 graph over a 30%-fewer-MAC architecture, and the combined
pruning+quantization payoff DESIGN §8 set out to measure.

| | |
|---|---|
| Architecture | the M3 c30 pruned MobileNetV2 (14 expansion groups narrowed), INT8 QAT: per-channel S8 weights, per-tensor S8 activations, QDQ, BN folded at init |
| Starts from | M3 c30 checkpoint `89bbb9e7…` (D4 epoch 2) — never M0, never M2 (DESIGN §8.4) |
| Input | `[1, 3, 192, 256]` NCHW float32, the frozen §5.5 preprocessing — unchanged |
| Deployable artifact | `results/optimize/m4_qat/lr5e-5/model.onnx`, opset 17, **2,014,806 B** (ladder-smallest; M2 2,536,267; M3 FP32 7,035,950) |
| QAT run | best epoch **3** of 6 by the frozen §7.2 rule (run dir `d5_m2_qat_lr5e-5_20260717T060301Z` — the run-name stem is a pre-fix cosmetic artifact, see note; the candidate id `d5_m4_qat_lr5e-5` is correct everywhere downstream) |
| Params / MACs | 1,761,720 / 205,614,080 — **identical to M3** (QAT preserves shape), copied from the M3 row |
| Observer calibration | the frozen D1 manifest (`9fab904c…`) — shared with M1/M2/M3 by registration |
| Policy | `artifacts/policies/bobcat_m4_qat_lr5e-5_v1.json`, threshold **0.543913**, status **`recall_floor_infeasible`** |

## How M4 was produced (D5, DESIGN §8.4)

Registered before any number (`results/optimize/m4_qat/registration.md`):
**the validated M2 recipe applied to the selected M3 checkpoint, no new
search.** M2's LR sweep already chose 5e-5; §8.4 says "apply the validated QAT
procedure", so M4 is one arm at lr5e-5, 6 epochs, observers frozen after epoch
1, seed 42 — M2's recipe verbatim, from the c30 init. The trainer rebuilds the
pruned architecture (`prune.apply_widths`) before loading c30 and before
inserting the shape-agnostic QAT structure; the export folds Q/DQ into INT8
initializers (the D2 bitwise-proven fold), so M4 is a real INT8 graph
(`integer_execution=True`), not a float graph carrying rounded weights.

## The DESIGN §8.4 verdict — M4 does not win by default, and did not need to

The registration committed to recording whichever happened. What happened:

- **M4 does not dominate M2, and M2 does not dominate M4.** M2 wins primary
  (0.3832 vs 0.373); M4 wins **both MACs** (205.6M vs 293.4M) **and size**
  (2.01 vs 2.54 MB). They are **non-dominated** → both go to the D6 shortlist,
  and **Pi latency (F-phase) settles which ships**. M4 is the MAC-and-size
  champion of the ladder at a ~2.7% primary cost.
- **M4 dominates its own FP32 parent M3**: same architecture and MACs, higher
  primary (0.373 vs 0.3583) *and* a 3.5× smaller file. For deployment, M4
  strictly supersedes M3 — the QAT fine-tune recovered accuracy the pruning
  cost *and* shrank the file. M3 remains on record as the FP32 pruning result;
  M4 is what a pruned model should actually ship as.

## Metrics

Validation, deployment ORT (CPU EP, batch 1); test sealed. Reference: M0's
ONNX through the same evaluator (primary 0.3667).

### At the 0.5 selection yardstick

| | M4 cis | M2 cis | M3 cis | M4 trans | M2 trans | M3 trans |
|---|---:|---:|---:|---:|---:|---:|
| bobcat F2 | 0.6529 | 0.6499 | 0.5879 | 0.0930 | 0.1166 | 0.1287 |
| frame recall | 0.7708 | — | — | 0.0769 | — | — |
| precision | 0.4051 | — | — | 0.5701 | — | — |
| false-fire | 0.0531 | — | — | 0.0494 | — | — |
| average precision | 0.527 | 0.5218 | 0.4946 | 0.587 | 0.5218 | 0.5967 |

Primary (mean F2@0.5): **0.373** vs M2 0.3832, M3 0.3583, M0-ref 0.3667.

**Finding — QAT sharpened cis at trans's expense.** M4's cis F2 (0.6529) is the
ladder's best and beats its own parent M3 (0.5879); its trans F2 (0.0930)
*fell* below M3's FP32 0.1287. The pruned architecture has less capacity, and
the QAT fine-tune spent it on the easier seen-camera domain. So M4 is the
cis-and-efficiency candidate; M3-FP32 held the best trans F2 on the ladder
(0.1287), which is a real reason to keep M3's numbers on record even though
M4 dominates it on the primary.

### At the calibrated operating point 0.543913 (`bobcat_m4_qat_lr5e-5_v1`)

| | cis-val-clean | trans-val |
|---|---:|---:|
| bobcat F2 | 0.6615 | 0.0901 |
| frame recall | 0.7708 | 0.0744 |
| sequence-balanced recall | 0.7667 | 0.0742 |
| false-fire | 0.0495 | 0.0461 |
| event capture | 0.86 | 0.1547 |

**The primary rule is NOT met** — `recall_floor_infeasible`, as everywhere on
the ladder. Bootstrap threshold [0.4445, 0.7355] — the tightest INT8 interval
of the ladder (M1 aside), but the trans-domain recall at the operating point
(0.074) is the ladder's weakest; the cis event-capture (0.86) is its strongest.
D6 should weigh both.

## Intended use and limitations

Same intended use and out-of-scope list as [m0_fp32.md](m0_fp32.md); same
INT8 caveats as [m1_int8_ptq.md](m1_int8_ptq.md); same pruned-capacity caveat
as [m3_pruned_fp32.md](m3_pruned_fp32.md). M4's value is that it is small on
both axes at once (MACs from pruning, bytes from INT8) — whether that buys Pi
latency over M2 is exactly what D6/F-phase measures. It is **not** assumed to
be the final model.

## Parity and deployment evidence (P3/P4)

| gate | result |
|---|---|
| Coverage (P3.1) | integer execution confirmed on ARM64; no float Conv/Gemm/MatMul survived |
| Metrics reproduce (P3.2) | full validation re-run equals the recorded candidate exactly |
| ORT py↔cpp fixtures (P3.3) | clean under the registered gates |
| Binding (P3.4) | every C++ infer record names this artifact hash and policy id |
| C++ dataset parity (P4) | both full splits, 4,939 frames, worst gap 5.96e-08, 1 trans decision within the 1e-4 threshold carve-out (listed), matrices otherwise equal |

Evidence: `results/optimize/m4_qat/lr5e-5/p3_quantized.json`,
`results/optimize/m4_qat/lr5e-5/p4_dataset_parity.json`.

## Hashes

| artifact | sha256 |
|---|---|
| `model.onnx` (deployable, folded INT8) | `2c9d53b41fcf88c3a1df1a986a8ceccc5fe8f96965bcf8a46d4f2a7d9bc770da` |
| source M3 c30 checkpoint | `89bbb9e7e30b5938c0826ecdc73be80a80c28d37ff91690af944dc15ab881b88` |
| observer-calibration manifest | `9fab904c6eb3a5501011fdd3277b3ee98655a906d6301da44364ef35d8a863c4` |

The `.onnx` and QAT checkpoints live on gx10 until the G5 release; every hash
above is committed.

## Note on the run-name stem

`qat_train.train_arm` recorded the run directory as `d5_m2_qat_lr5e-5_…`: the
run-name stem was hard-coded to `m2_qat` when M2 was the only QAT candidate.
D5 parameterized it (`run_name_stem`) so future runs read `d5_m4_qat_…`; this
run predates that one-line fix. The **candidate id** (`d5_m4_qat_lr5e-5`), the
policy, the comparison row, and every hash are correct — only the internal
run-directory name carries the old stem, and nothing downstream keys on it.

## License

Same as [m0_fp32.md](m0_fp32.md).

## Machine-readable row

`results/model_selection/comparison.jsonl`, `model_id: M4`, `kind: pruned_qat`
— params/MACs copied from the committed M3 row (QAT preserves the pruned
shape), everything else from M4's own artifact and evaluation.
