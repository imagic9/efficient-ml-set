# Model card — M2, INT8 QAT (lr 5e-5, 6 epochs from M0)

**One sentence:** M0's MobileNetV2 fine-tuned for 6 epochs under quantization
noise (direct output-side QDQ fake-quant, the P0-pinned path), exported as a
real INT8 graph — the smallest artifact on the ladder (2,536,267 B, 3.53x under
M0) and the only quantized candidate whose validation primary **exceeds** the
FP32 deployment reference.

| | |
|---|---|
| Architecture | identical graph shape to M0/M1; weights INT8 per-channel symmetric, activations INT8 per-tensor affine, QDQ; BN folded from the M0 checkpoint at init |
| Starts from | M0 checkpoint `42079c36…` (epoch 11) — never M1, per DESIGN §8.2 |
| Input | `[1, 3, 192, 256]` NCHW float32, the frozen §5.5 preprocessing — unchanged |
| Deployable artifact | `results/optimize/m2_qat/lr5e-5/model.onnx`, opset 17, **2,536,267 B** (M0 8,950,645 → **3.53x**; M1 2,620,130) |
| QAT run | `d2_m2_qat_lr5e-5_20260716T210550Z`, best epoch **5** of 6 by the frozen §7.2 rule |
| Observer calibration | the frozen D1 manifest (`9fab904c…`) — shared corpus with M1, by registration |
| Policy | `artifacts/policies/bobcat_m2_qat_lr5e-5_v1.json`, threshold **0.650390**, status **`recall_floor_infeasible`** |
| Selection | `results/optimize/m2_qat/selection.{json,md}` — mechanical, pre-registered (`preregistration.md`, committed before any arm ran) |

## How M2 was produced (D2, DESIGN §8.2)

1. **Pre-registration first** (PR #45): init pinned by hash, BN folded at init
   (the registered reading of §8.2's freeze bullet), observers frozen after
   epoch 1, AMP off (a recorded deviation — fake-quant in fp16 trains scales
   against arithmetic the deployed graph never performs), 6 epochs fixed,
   arms 1e-5/3e-5/5e-5 and nothing outside the documented range.
2. Three arms trained under the frozen §7.2 data/loss contract; per-epoch
   epoch selection by the frozen §7.2 rule under active fake-quant.
3. Each arm scored through its own ORT CPU outputs and calibrated its §6.3
   policy; `select_ptq` applied the D1 rule across arms.
4. **The export-size finding and its fix** (PRs #47–#49): the fake-quant
   export stored FP32 weights behind Q/DQ — integer execution, float storage,
   9,096,154 B. `optimize.fold_qdq` (ORT Basic-level constant folding only)
   turned them into INT8 initializers with **proven bitwise-identical
   outputs**; every arm was re-derived end to end over the folded artifacts
   and every metric matched to every recorded digit. Both evidence sets are
   committed, so that claim is checkable.

**Findings for the record.** (a) The LR search was non-monotonic: 5e-5 > 1e-5
> 3e-5 on the primary — a reminder that three points sample a curve, they do
not map it. (b) QAT at 5e-5 *improved on the FP32 reference in both domains*:
six extra low-LR epochs under quantization noise acted as useful additional
fine-tuning, not merely damage control.

## Metrics

All numbers are validation, scored through deployment ORT (CPU EP, batch 1);
test stays sealed. References: M0's ONNX through the same evaluator, and M1
(percentile PTQ) from its committed row.

### At the 0.5 selection yardstick

| | M2 cis | M0-ref cis | M1 cis | M2 trans | M0-ref trans | M1 trans |
|---|---:|---:|---:|---:|---:|---:|
| bobcat F2 | **0.6499** | 0.6280 | 0.6015 | **0.1166** | 0.1054 | 0.1039 |
| frame recall | 0.7708 | — | — | 0.0971 | — | — |
| precision | 0.3993 | — | — | 0.5878 | — | — |
| false-fire | 0.0544 | — | — | 0.0579 | — | — |
| average precision | 0.5810 | 0.5948 | 0.5702 | 0.5218 | 0.5256 | 0.5280 |
| event capture | 0.8400 | — | — | 0.1887 | — | — |

Primary (mean F2@0.5): **0.3832** vs reference 0.3667 → ratio **1.0451** —
above the pre-registered *fully recovers* line (≥ 1.0); M1's primary was
0.3527. Per the registered reading, **QAT recovers PTQ accuracy and exceeds
the FP32 reference** on this validation data. AP is fractionally below the
reference in both domains — the F2 gains live at the yardstick and operating
region, not uniformly across all thresholds.

### At the calibrated operating point 0.650390 (`bobcat_m2_qat_lr5e-5_v1`)

| | cis-val-clean | trans-val |
|---|---:|---:|
| bobcat F2 | 0.6400 | 0.0976 |
| frame recall | 0.7431 | 0.0807 |
| precision | 0.4115 | 0.5981 |
| sequence-balanced recall | 0.7400 | 0.0805 |
| false-fire | 0.0498 | 0.0461 |
| event capture | 0.8400 | 0.1698 |

**The primary rule is NOT met** — the same registered status as M0 and M1
(`recall_floor_infeasible`); QAT improved scores but no threshold inside the
5% per-domain false-fire budget reaches the 90% sequence-recall floor on both
domains. Bootstrap (1,000 replicates): threshold in **[0.4970, 0.9144]** — a
wide interval; the operating point is less stable across resamples than M0/M1's
and D6 should weigh that.

## Intended use and limitations

Same intended use and out-of-scope list as [m0_fp32.md](m0_fp32.md); same
INT8-specific caveats as [m1_int8_ptq.md](m1_int8_ptq.md) (near-threshold
coarseness, calibration-corpus dependence for activation scales). At the
operating point M2 catches ~84% of bobcat visits at seen cameras and ~17% at
the unseen camera — the unseen-camera weakness is a property of the recipe
(issue #18's verdict) and QAT did not change that story.

## Parity and deployment evidence (P3/P4, gates registered before measurement)

| gate | result |
|---|---|
| Weight fold | bitwise-identical outputs on seeded probes; INT8 initializers; ai.onnx only |
| Coverage (P3.1) | integer execution on ARM64; no float Conv/Gemm/MatMul survived |
| Metrics reproduce (P3.2) | full validation re-run equals the recorded candidate exactly |
| ORT py↔cpp fixtures (P3.3) | clean under the registered gates (≤1e-4, identical argmax/decisions) |
| Binding (P3.4) | every C++ infer record names this artifact hash and policy id |
| C++ dataset parity (P4) | both full splits: worst gap 5.96e-08, 0 decision diffs, matrices equal |

Evidence: `results/optimize/m2_qat/lr5e-5/p3_quantized.json`,
`results/optimize/m2_qat/lr5e-5/p4_dataset_parity.json`.

## Hashes

| artifact | sha256 |
|---|---|
| `model.onnx` (deployable, folded) | `499bc3ecb30853b8592fc6077af706d26026fc2eb08ee12512fd2f4706d45ecc` |
| source M0 checkpoint | `42079c362013898c3354a65bbf8ced4524504c0dfc20cb6efaa00dabe9209074` |
| observer-calibration manifest | `9fab904c6eb3a5501011fdd3277b3ee98655a906d6301da44364ef35d8a863c4` |

The `.onnx` and QAT checkpoints live on gx10 until the G5 release; every hash
above is committed.

## License

Same as [m0_fp32.md](m0_fp32.md).

## Machine-readable row

`results/model_selection/comparison.jsonl`, `model_id: M2` — written by
`wildlife_trigger.comparison --candidate` from the candidate, policy, and P3
evidence.
