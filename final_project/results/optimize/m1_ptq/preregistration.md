# D1 pre-registration — M1 INT8 PTQ (written before any candidate result was viewed)

Date: 2026-07-16. Commit order is the proof: this file lands before any PTQ
candidate is calibrated or scored, per PLAN D1 ("record the pre-registered
MobileNetV2 PTQ risk before viewing results") and DESIGN §8.1.

## 1. The registered risk

MobileNetV2 is a known-hard PTQ target, and the mechanism is specific enough to
name in advance:

- **Depthwise convolutions have too few weights per channel** for their ranges
  to average out: per-layer (per-tensor) weight quantization mixes channels whose
  ranges differ by orders of magnitude, and accuracy collapses (the classic
  measurement is Krishnamoorthi 2018, and DFQ/Nagel et al. 2019 report naive
  per-tensor MobileNetV2 ImageNet top-1 near 0.1%). **Per-channel signed INT8
  weights are therefore the floor here, not an option** — DESIGN §8.1 fixes them.
- **Inverted-residual expansions produce wide, outlier-heavy activations**
  (ReLU6 caps them at 6 in training, but the pre-projection linear bottleneck
  outputs are unbounded). Camera-trap data aggravates this: IR night frames and
  blown-out daylight frames sit far from the ImageNet statistics the backbone
  was pre-trained on. This is exactly the failure MinMax calibration is most
  vulnerable to (one outlier sets the scale for everything), and the reason
  Entropy and Percentile are in the comparison.
- Where the damage should concentrate, if it appears: the depthwise 3x3 layers
  and the final 1x1 projections of late blocks, and — for the trigger metric —
  the near-threshold score band, because quantization noise on the order of the
  score gap moves decisions there first.

Registered expectation for **this** configuration (per-channel weights, S8S8
QDQ, 1,024-image in-domain calibration): a **modest** drop on validation bobcat
F2 relative to the FP32 ORT reference — not a collapse. A measured collapse is a
finding about the recipe (calibration method or data), not a reason to touch the
test protocol. QAT (D2) is the planned recovery path; a PTQ failure here remains
a reported negative result (DESIGN §8.1).

## 2. The registered reference

The FP32 reference for every drop computation is **M0's ONNX scored through the
same ORT CPU evaluation path as the candidates** (batch 1, deployment
arithmetic, the §6.3 amendment regime) — never `history.json`'s CUDA numbers:
issue #30 measured the CUDA-TF32-vs-deployment gap at up to 7.25e-3, which is
material in the near-threshold band. Reference primary at the fixed 0.5
yardstick, computed once, before any candidate score is looked at.

## 3. The registered selection rule (validation only)

Among candidates that pass the coverage gate (`integer_execution == true` in
`validate.ort_coverage`'s verdict — at least one integer kernel executed AND no
float Conv/Gemm/MatMul survived):

1. **primary**: mean bobcat frame F2 at threshold 0.5 across cis-val-clean and
   trans-val — `metrics.PRIMARY_METRIC`, the same frozen §7.2 yardstick every
   other selection in this project uses;
2. tie-breaks, in §7.2 order: mean sequence-balanced recall, then support-aware
   macro F1;
3. final tie-break: smaller ONNX bytes.

Candidates failing coverage are excluded from selection regardless of F2 — a
"quantized" model that runs float convolutions is not an INT8 candidate, it is
mislabeled M0. Test labels are not read anywhere in D1.

## 4. The registered "material drop" trigger for quantization debugging

Run ORT quantization debugging (per-layer FP32-vs-QDQ activation comparison on
calibration data) and record per-layer findings **before accepting the verdict**
if, for the selected candidate:

- primary < **0.95 x** the M0 ORT reference primary (relative drop > 5%), **or**
- either domain's bobcat frame F2 at 0.5 falls by more than **10% relative** to
  the same reference domain value.

Below both lines, PTQ is declared non-material and debugging is recorded as
"not triggered". The lines are chosen now, before any number exists; that is
their entire value.

## 5. The registered QOperator rule

S8S8 QDQ is the primary representation (ARM64 SDOT kernels are signed; U8S8 is
the x86 VNNI recommendation). A QOperator-format candidate is generated and
measured **only if** the QDQ coverage verdict on the deployment-representative
ARM64 host (gx10, aarch64, ORT 1.27.0 CPU EP) shows float Conv/Gemm/MatMul
surviving optimization, or the ORT profile shows the QDQ graph executing
predominantly non-integer kernels. Otherwise "QOperator not warranted" is
recorded alongside the coverage evidence. If generated, it is an explicitly
named extra candidate (`qoperator_<method>`), never a silent substitution.

## 6. Calibration data (fixed before use)

The 1,024-image manifest is built once by
`wildlife_trigger.optimize.calibration_manifest` from **training data only**
(`train.jsonl` + `cct_empty_train_v1.jsonl`), stratified by (source, class)
with a floor per stratum, seed 20260716, and committed with its sha256 in
`calibration_manifest.json`. All three calibration methods read the identical
manifest in the identical order. Validation and test images never enter it —
the builder refuses manifests whose names look like val/test splits.

## 7. What this pre-registration does not permit

- No re-selection after seeing P3/P4 or any test-set number.
- No threshold-rule changes: policies come from the frozen §6.3 rule
  (5% per-domain false-fire budget, 90% sequence-balanced recall floor,
  verbatim status recording) applied to each candidate's own ORT validation
  scores.
- No retraining or re-export of M0: M1 starts from the committed M0 ONNX
  (sha256 `c3102764…`, the artifact bound to `bobcat_v1.json`).
- Amending this file after results exist requires recording the amendment and
  the trigger measurement in this same file, per the project's standing
  practice (see the §6.3 amendment history in DESIGN §10).
