# Model card — M1, INT8 PTQ (percentile calibration)

**One sentence:** M0's MobileNetV2, post-training-quantized to per-channel S8S8
QDQ INT8 on 1,024 frozen training images — 3.4x smaller than M0 with a 3.8%
relative drop in the selection primary, chosen among three calibration methods
by the pre-registered rule, never by eye.

| | |
|---|---|
| Architecture | identical graph shape to M0 (MobileNetV2, 16-output head); weights INT8 per-channel symmetric, activations INT8 per-tensor affine, QDQ representation |
| Starts from | M0 FP32 ONNX `c3102764…` (never retrained, never re-exported) |
| Input | `[1, 3, 192, 256]` NCHW float32, the frozen §5.5 preprocessing — unchanged |
| Deployable artifact | `results/optimize/m1_ptq/percentile/model.onnx`, opset 17, **2,620,130 B** (M0: 8,950,645 B, **3.42x**) |
| Calibration | `calibration_1024.jsonl` (sha `9fab904c…`), training data only, percentile method; MinMax/Entropy candidates measured and lost |
| Policy | `artifacts/policies/bobcat_m1_ptq_percentile_v1.json`, threshold **0.496375**, status **`recall_floor_infeasible`** |
| Selection | `results/optimize/m1_ptq/selection.{json,md}` — mechanical, pre-registered (`preregistration.md`, committed before any result) |

## How M1 was produced (D1, DESIGN §8.1)

1. **Pre-registration first** — the MobileNetV2 PTQ risk, the selection rule,
   the material-drop triggers and the QOperator rule were committed before any
   candidate existed (PR #38).
2. Three static-quantization candidates (MinMax, Entropy, Percentile) were
   generated from the hash-pinned M0 on the identical frozen manifest in the
   identical order; all three executed as integer on the ARM64 host (no float
   Conv/Gemm/MatMul survived session optimization).
3. Every candidate was scored on validation through **its own ORT CPU batch-1
   outputs** — the deployment arithmetic, not the torch checkpoint — and each
   calibrated its own §6.3 policy from those scores. The M0 reference below
   went through the same path, so the drop is regime-clean (issue #30's
   lesson).
4. `optimize.select_ptq` applied the rule: percentile wins the frozen §7.2
   primary; no debugging trigger fired; QOperator not warranted.

**Findings for the record.** (a) MinMax and Entropy produced **byte-identical**
models (sha `964d1196…`): ORT 1.27's entropy calibrator landed on exactly the
MinMax ranges for every activation — consistent with MobileNetV2's
ReLU6-bounded activations pinning the KL threshold at the observed maximum
(hypothesis; ORT internals not diagnosed). The comparison was therefore
effectively two candidates. (b) The pre-registered depthwise collapse did not
materialize: per-channel weights plus in-domain calibration held the primary
drop to 3.8% relative, inside the 5% line.

## Metrics

All numbers are validation, scored through deployment ORT (CPU EP, batch 1);
test stays sealed. The M0 reference column is M0's ONNX through the *same*
evaluator — its cis F2@0.5 of 0.6280 differs from the model card's 0.6272
(CUDA history value) by exactly the regime gap §6.3's amendment exists for.

### At the 0.5 selection yardstick

| | M1 cis | M0-ref cis | M1 trans | M0-ref trans |
|---|---:|---:|---:|---:|
| bobcat F2 | **0.6015** | 0.6280 | **0.1039** | 0.1054 |
| frame recall | 0.6875 | — | 0.0858 | — |
| precision | 0.4008 | — | 0.6800 | — |
| false-fire | 0.0482 | — | 0.0343 | — |
| average precision | 0.5702 | 0.5948 | 0.5280 | 0.5256 |
| event capture | 0.7600 | — | 0.1811 | — |

Primary (mean F2@0.5): **0.3527** vs reference 0.3667 → ratio **0.9618**
(pre-registered debugging line: 0.95; not triggered). Per-domain relative
drops: cis −4.2%, trans −1.4% (lines: −10%; not triggered).

### At the calibrated operating point 0.496375 (`bobcat_m1_ptq_percentile_v1`)

| | cis-val-clean | trans-val |
|---|---:|---:|
| bobcat F2 | 0.6046 | 0.1054 |
| frame recall | 0.6944 | 0.0870 |
| precision | 0.3984 | 0.6832 |
| sequence-balanced recall | 0.6933 | 0.0868 |
| false-fire | 0.0492 | 0.0343 |
| event capture | 0.7800 | 0.1849 |

**The primary rule is NOT met** — same registered status as M0
(`recall_floor_infeasible`): no threshold inside the 5% per-domain false-fire
budget reaches 90% sequence-balanced recall on both domains. Quantization
neither created nor destroyed a passing trigger. The seq_id-cluster bootstrap
(1,000 replicates) puts the threshold in **[0.3845, 0.6299]**.

## Intended use and limitations

Same intended use, out-of-scope list, and unseen-camera limitation as
[m0_fp32.md](m0_fp32.md) — M1 is M0's arithmetic changed, not its knowledge:
at the operating point it catches ~78% of bobcat visits at seen cameras and
~18% at the unseen camera. Additional M1-specific limitations:

- INT8 scores are *coarser* than FP32 scores; near the threshold this shows up
  as different individual fire decisions than M0 would make, even though the
  aggregate rates match. The policy threshold is meaningful only for exactly
  this artifact (bound by hash).
- The calibration data determines the activation scales; frames statistically
  unlike the calibration manifest (heavy overexposure, unusual IR) may saturate
  activations that the calibrator clipped.

## Parity and deployment evidence (P3/P4, gates registered before measurement)

| gate | result |
|---|---|
| Coverage (P3.1) | integer execution on ARM64: QLinearConv executes, no float Conv/Gemm/MatMul |
| Metrics reproduce (P3.2) | full validation re-run equals the recorded candidate exactly |
| ORT py↔cpp fixtures (P3.3) | see `p3_quantized.json` (logits ≤ 1e-4, identical argmax/decisions) |
| Binding (P3.4) | every C++ infer record names this model hash and this policy id |
| C++ dataset parity (P4) | see `p4_dataset_parity.json` — ordered ids, labels, scores ≤ 1e-4, decisions, confusion matrices on both full validation splits |

Evidence: `results/optimize/m1_ptq/percentile/p3_quantized.json`,
`results/optimize/m1_ptq/percentile/p4_dataset_parity.json`.

## Hashes

| artifact | sha256 |
|---|---|
| `model.onnx` (deployable, percentile) | `faf54dde23e4418b2cfb3b79563883624c97e536968d50aee9ee98746e1345c8` |
| source M0 ONNX | `c3102764b6ab3bf9f5c8984b7cbdb80b581458db1e668d1e6d9e88df25ce5154` |
| calibration manifest | `9fab904c6eb3a5501011fdd3277b3ee98655a906d6301da44364ef35d8a863c4` |
| rejected minmax ≡ entropy artifact | `964d119600145f04…` (byte-identical pair, see selection.md) |

The `.onnx` lives on gx10 until the G5 release; the policy binds the hash, so
any copy can be verified.

## License

Same as [m0_fp32.md](m0_fp32.md): CCT-20 per LILA's CDLA-permissive terms,
torchvision ImageNet weights, course-work code with the G-phase license
deliverable.

## Machine-readable row

`results/model_selection/comparison.jsonl`, `model_id: M1` — written by
`wildlife_trigger.comparison --candidate` from the candidate, policy, and P3
evidence.
