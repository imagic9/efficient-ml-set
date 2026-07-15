# Final Project — Executable Design Specification

Status: **Core approved; implementation not started.**

This document is the source of truth for the Efficient ML final project. It is
written as an execution contract for an AI coding agent: implement the phases in
order, satisfy every gate, preserve the raw evidence, and prepare the complete
submission package. Do not add features or change the experimental question
without updating this document first.

The old two-model cascade is no longer part of Core. It depended on an empty-frame
prevalence that does not hold in CCT-20 and added a second model without being
required by the assignment. The only permitted Stretch is crop-teacher knowledge
distillation, and it may start only after the Core Definition of Done passes.

---

## 1. Project in one sentence

> A CPU-only C++ application on a Raspberry Pi 5 analyzes a wildlife-camera frame
> and emits an emulated shutter signal when the target animal, a bobcat, is present.

The course project implements and measures the decision core. A future physical
product could connect the same decision to a PIR sensor, camera, and GPIO shutter
interface, but no live camera or GPIO is required for this submission.

### Demonstration behavior

```text
frame_000412.jpg  predicted=bobcat  score=0.94  -> SHUTTER_TRIGGER=1
frame_000413.jpg  predicted=coyote  score=0.03  -> SHUTTER_TRIGGER=0
frame_000414.jpg  predicted=empty   score=0.01  -> SHUTTER_TRIGGER=0
```

The target in CCT-20 is **bobcat (`Lynx rufus`)**, sometimes called the red lynx.
It is not the Eurasian lynx (`Lynx lynx`). All code, metrics, report text, and
slides must use the unambiguous label `bobcat`.

---

## 2. The falsifiable engineering question

> How much latency, throughput, memory, and model-size improvement can we obtain
> on a Raspberry Pi 5 by applying structured pruning and affine INT8
> quantization to a full-frame wildlife classifier, while preserving bobcat
> recall at the selected operating point?

This question maps directly to the assignment:

| Rubric area | Evidence produced by this project |
|---|---|
| Model training and optimization, 15 pts | MobileNetV2 justification; FP32 baseline; PTQ; QAT; structured pruning; accuracy/compute trade-offs |
| C++ inference, 15 pts | Correct preprocessing, ONNX Runtime inference, target policy, signal emulation, dataset runner, benchmark harness, tests |
| Benchmarking and metrics, 10 pts | Baseline and optimized models measured on the same Pi 5 with latency, FPS, resource use, and accuracy |
| Analysis and presentation, 10 pts | Reproducible tables and plots; what worked; what failed; bottlenecks; concrete next steps |

The project is successful even if an optimization fails to improve speed or
accuracy, provided the failure is measured correctly and explained.

---

## 3. Scope

### 3.1 Core — required for submission

1. Prepare a leakage-safe wildlife dataset with CCT-20 cis/trans splits.
2. Train one FP32 full-frame MobileNetV2 classifier.
3. Calibrate a primary bobcat shutter threshold using validation data only and
   implement a generic policy that can select any subset of the 15 known animals.
4. Produce and evaluate:
   - FP32 baseline;
   - INT8 PTQ;
   - INT8 QAT;
   - structured-pruned FP32;
   - structured-pruned + QAT candidate.
5. Export deployable ONNX models and prove numerical/decision parity.
6. Implement the inference application in C++.
7. Run baseline and optimized benchmarks on a rented Raspberry Pi 5.
8. Produce the public repository, reproducibility assets, report, and slides.

### 3.2 Stretch — locked until Core is complete

The only Stretch is **crop-teacher knowledge distillation**:

- teacher: a stronger classifier trained on ground-truth animal crops;
- student: the same full-frame MobileNetV2 used by Core;
- control: crop augmentation without KD;
- success criterion: KD must beat crop augmentation, not merely the naive FP32
  baseline.

Stretch must not modify, replace, or delay the frozen Core result. It is reported
as an additional experiment.

### 3.3 Explicitly out of scope

- physical camera, PIR, GPIO, or power meter;
- on-device object detector;
- gate/cascade model;
- person/vehicle/illegal-logging detector;
- elephant or regional model packs;
- separate per-species neural networks or downloadable model packs;
- multi-label recognition of simultaneous species in one frame;
- custom inference engine;
- Hailo/AI HAT/NPU;
- battery-life claims;
- training on any test split;
- XNNPACK as a requirement. Core uses ONNX Runtime CPU Execution Provider; any
  other backend may be mentioned only as future work.

---

## 4. System architecture

```text
saved JPEG frame
      |
      v
C++ decode + deterministic full-frame preprocessing
      |
      v
MobileNetV2, 16 logits: 15 animal classes + empty
      |
      v
softmax -> configured animal scores
      |
      +-- any selected score >= its threshold --> SHUTTER_TRIGGER=1
      |
      `-- otherwise ----------------------------> SHUTTER_TRIGGER=0
```

There is exactly one neural-network inference per frame. The same C++ application
loads every baseline/optimized ONNX model so model comparisons are not confounded
by different application code.

### Runtime policy

The Raspberry Pi is assumed to be powered and the inference process resident.
The project measures model loading separately but does not claim boot-from-off or
suspend-to-inference latency. The emulated shutter signal is a console/JSON event,
not a physical GPIO transition.

### Configurable target policy

The neural network is shared by every target selection. A policy is a small
configuration file, not a separate model. It selects one or more of the 15 animal
outputs and fires when any selected class passes its own threshold:

```yaml
schema_version: 1
policy_id: bobcat_coyote_v1
model_sha256: MODEL_SHA256
class_map_sha256: CLASS_MAP_SHA256
mode: any
targets:
  - class: bobcat
    threshold: 0.42
  - class: coyote
    threshold: 0.55
```

The numbers above illustrate the numeric schema only; generated policies must
replace them with measured validation-calibrated values.

`empty` is never a valid target. `mode: any` is the only Core combination mode.
The loader rejects an empty target list, duplicate or unknown classes, thresholds
outside `[0, 1]`, unsupported modes, and model/class-map hash mismatches.

The submission's primary evaluated policy remains the bobcat-only
`bobcat_v1.yaml`. The same final model must also accept a multi-target example
policy without another inference or model reload. This adds configuration-level
modularity without adding networks or changing the benchmark question.

The calibration pipeline produces a model-bound threshold catalog for all 15
animal classes from validation data. For a class represented in both validation
domains, apply the two-domain rule from section 6.3. If a class has no positives
in one domain or too little support for that rule, use pooled validation F2 only
as a fallback and record the missing-domain/low-support limitation. Never invent
a threshold or claim validated recall without positive examples. A generated
multi-target policy must store its combined validation trigger metrics because
false-fire rates increase when multiple targets are joined with `any`.

A new species outside the 15-class map requires labelled data, classifier-head
adaptation, fine-tuning, export, and recalibration, but not training from random
initialization. Simultaneous identification of multiple different species in one
frame would require a multi-label classifier or detector and remains out of Core.

### Execution environments

`gx10` is the dedicated primary development and compute host for the complete
project until submission. It is an NVIDIA GB10 ARM64 system with CUDA; A0 must
capture the exact current hardware, OS, CUDA, compiler, and package versions
rather than trusting this description. Use `gx10` for:

- source development and repository automation;
- all data downloads, preparation, audits, and artifact storage;
- GPU model training, PTQ, QAT, pruning, export, and Python evaluation;
- CPU-only C++ compilation and inference with ONNX Runtime CPU EP;
- shutter-signal emulation, unit/integration/parity tests, profiling, notebooks,
  report/slide generation, and packaging;
- the complete pre-Pi ARM64 dry run.

No separate local development machine or local ARM64 environment is part of the
execution plan. Use isolated, pinned environments on `gx10`, and keep long jobs
restartable with checkpoints and persistent logs.

ARM64 instruction-set compatibility alone does not prove Raspberry Pi OS binary
compatibility. The pre-Pi C++ build and bundle-install test must therefore run on
`gx10` inside a clean target-compatible ARM64 container matching the rented Pi's
planned OS, glibc, compiler, OpenCV, and ONNX Runtime constraints as closely as
possible. If a portable binary cannot be proven, the deployment bundle must
include pinned source/build automation and compile on the Pi during provisioning.

The rented Raspberry Pi 5 is the only second execution environment. It is used
only for Phase F provisioning, target-hardware smoke/parity checks, and the final
CPU performance/resource/accuracy measurements. Results from `gx10` may be used
for correctness and candidate screening, but never reported as Pi latency, FPS,
memory, temperature, or throttling evidence. During Phase F, `gx10` remains the
control and evidence host: it drives remote commands where practical and receives
an immediate copy of every raw Pi result.

---

## 5. Data contract

### 5.1 Primary benchmark: CCT-20

Use the official Caltech Camera Traps-20 benchmark metadata and downsized images.
The downloadable split files contain **57,864 images** in total:

| Split | Images | Use |
|---|---:|---|
| train | 13,553 | Model training only |
| cis-val | 3,484 | Validation at known camera locations |
| cis-test | 15,827 | Final test at known camera locations |
| trans-val | 1,725 | Validation at one unseen camera location |
| trans-test | 23,275 | Final test at nine unseen camera locations |

The ECCV paper states 57,868 images, while the current downloadable split JSONs
sum to 57,864. The pipeline must record the exact downloaded file hashes and use
the JSON contents, not a number copied from the paper, as executable truth.

CCT-20 contains 15 animal categories plus `empty` (16 total). The category order
in the model is defined once in `configs/data/classes.yaml` and must never be
inferred from dictionary iteration order.

### 5.2 Required empty-frame supplement

The current official CCT-20 `train_annotations.json` contains no empty-labelled
training images, although validation and test do. A production-valid classifier
therefore needs a small empty training supplement.

Build `cct_empty_train_v1` from full CCT metadata using these rules:

1. Collect the set of all 20 CCT-20 location IDs.
2. Candidate image label must be exactly `empty`.
3. Candidate location must be disjoint from all 20 CCT-20 locations.
4. Select **5,000 images**, stratified across locations and sequences, with fixed
   seed `42`. Avoid letting one camera dominate the sample.
5. Download only selected images through the LILA per-image cloud paths; do not
   download the full 105 GB archive.
6. Save a manifest containing image ID, location, sequence ID, source URL,
   relative path, label, and checksum.
7. Use this supplement for training only. Do not create a new test set from it.

This changes the training recipe relative to the paper, so the report must say so.
All project models use the same augmented training dataset, preserving a fair
baseline-versus-optimized comparison.

### 5.3 Dataset assertions — hard data gate

Before training, automated tests must prove:

- split image counts match the table above;
- category set is exactly the configured 16 classes;
- every image ID is unique within a split;
- no image occurs in more than one split;
- no sequence spans train and either validation split;
- train locations are disjoint from trans-val and trans-test locations;
- supplemental-empty locations are disjoint from every CCT-20 location;
- every manifest path exists and every checksum matches;
- class and location distributions are written to `results/data_audit/`.

If any assertion fails, stop. Do not train around a split problem.

### 5.4 Test-set discipline

`cis-test` and `trans-test` are sealed until models, thresholds, code, runtime
configuration, and thread count are frozen. All development decisions use only
train, cis-val, trans-val, and an unlabeled validation benchmark manifest.

Final test labels may be used only by the final evaluation job. Rerunning an
unchanged frozen artifact for reproducibility is allowed; changing anything after
viewing test results is not.

### 5.5 Preprocessing

The network must see the complete frame; do not use a center crop that can remove
a small animal.

Canonical preprocessing:

1. Decode JPEG as 8-bit BGR with OpenCV.
2. Convert BGR to RGB.
3. Resize while preserving aspect ratio so the longer side is 224 pixels.
4. Center-pad the shorter side to 224 using RGB value `(114, 114, 114)`.
5. Convert to float32 and divide by 255.
6. Normalize with ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation
   `(0.229, 0.224, 0.225)`.
7. Convert HWC RGB to contiguous NCHW tensor `[1, 3, 224, 224]`.

The Python and C++ implementations must share golden fixtures and match within a
documented tolerance.

Training-only augmentation, applied before normalization:

- horizontal flip, probability 0.5;
- mild brightness/contrast/saturation jitter;
- random grayscale, probability 0.15, to simulate IR appearance;
- mild Gaussian blur, probability 0.10;
- no crop that can exclude the labelled animal.

Record every transform and parameter in configuration. Validation/test are fully
deterministic.

---

## 6. Learning task and operating point

### 6.1 Model output

The model performs 16-way single-label classification. The application action is
binary for a configured target set `T`:

```text
fire(frame, T) = any(softmax(logits)[class] >= threshold[class] for class in T)
```

Do not equate top-1 accuracy with product quality. A configured class may correctly
trigger even when another logit is slightly larger, provided the policy is
explicitly defined and evaluated consistently. The Core scientific evaluation
uses `T = {bobcat}`; generic target selection is a product/configuration feature.

### 6.2 Loss and class imbalance

Use weighted cross-entropy. Compute class weights from the frozen training
manifest using the effective-number-of-samples method, cap extreme weights, and
store the final numeric vector in the run metadata. The weighting scheme is part
of the baseline training recipe and remains identical across compression runs.

Do not use test frequencies for weighting or sampling.

### 6.3 Primary bobcat threshold calibration

Calibrate the bobcat threshold on `cis-val + trans-val` only. Treat the two
validation domains separately so the larger bobcat count in trans-val does not
hide a cis-val failure.

Primary rule:

1. Search all unique bobcat scores.
2. Choose the largest threshold for which bobcat recall is at least 90% on both
   cis-val and trans-val.
3. If no non-trivial threshold meets both constraints, choose the threshold
   maximizing the mean of cis-val and trans-val F2 and record which recall
   constraint was not met.
4. Save the threshold, calibration metrics, score distribution, and dataset hash
   to `artifacts/policies/bobcat_v1.yaml`.

The threshold is calibrated separately for each candidate model because
quantization changes score distributions. For the final baseline-versus-optimized
comparison, report both:

- each model at its own calibrated operating point;
- every model at the FP32 baseline threshold, to expose calibration drift.

After the final model is selected, run the same calibrator for every animal class
to produce the generic threshold catalog described in section 4. Classes lacking
adequate validation support must be explicitly flagged. These secondary thresholds
do not affect model selection or replace the bobcat evaluation.

### 6.4 Primary accuracy metrics

Report separately for cis-test and trans-test:

- bobcat recall — primary product metric;
- bobcat precision;
- bobcat F2;
- false-fire rate = false triggers / non-bobcat frames;
- fire rate = all triggers / all frames;
- macro F1 and per-class recall — secondary model-quality metrics;
- confusion matrix;
- 95% confidence intervals by bootstrap over `seq_id`, not individual frames.

Also report per-location bobcat recall on trans-test. Do not claim a universally
calibrated probability from softmax scores.

---

## 7. Baseline model

### 7.1 Architecture

**MobileNetV2, width multiplier 1.0, input 224x224, ImageNet-pretrained.** Replace
the classifier with a 16-output linear layer.

Reasons:

- designed for resource-constrained inference;
- mature pretrained weights;
- stable ONNX export and CPU operator support;
- depthwise-separable convolutions provide a meaningful structured-pruning case;
- small enough for repeated Pi benchmarks;
- the course container already proves the MobileNetV2/ONNX/C++ toolchain.

Core does not perform neural architecture search. Architecture search would add a
new experimental axis without earning a separate rubric category.

### 7.2 Training recipe

Default recipe; all deviations must be recorded in the run config:

- seeds: primary `42`; confirmation seeds `17` and `73` for baseline and final;
- phase A: train classifier head for 5 epochs, backbone frozen;
- phase B: fine-tune the full network for at most 30 epochs;
- optimizer: AdamW;
- learning rate: `1e-3` for head, `3e-4` for full fine-tuning;
- weight decay: `1e-4`;
- cosine learning-rate decay;
- mixed precision allowed on gx10, never on Pi inference;
- early stopping patience: 6 epochs;
- checkpoint selection score: mean of cis-val and trans-val bobcat F2, with
  bobcat recall used as the first tie-break and macro F1 as the second;
- save last and best checkpoints plus full optimizer/scheduler state.

Do not tune on overall accuracy alone.

### 7.3 Baseline artifacts

The baseline is not complete until all exist:

- PyTorch checkpoint;
- FP32 ONNX export;
- training history JSON/CSV;
- validation predictions and logits;
- calibrated policy YAML;
- model card;
- parity report;
- Pi benchmark row.

---

## 8. Optimization ladder

Every row uses the same input contract, labels, metric code, C++ application, and
benchmark images. Each transformation starts from the artifact named below; do
not create a misleading sequential chain.

| ID | Candidate | Starts from | Question |
|---|---|---|---|
| M0 | FP32 baseline | ImageNet MobileNetV2 | Reference accuracy and Pi performance |
| M1 | INT8 PTQ | M0 FP32 ONNX | Is calibration-only quantization sufficient? |
| M2 | INT8 QAT | M0 FP32 checkpoint | Does QAT recover PTQ accuracy while retaining INT8 speed/size? |
| M3 | Structured-pruned FP32 | M0 FP32 checkpoint | Does real channel/MAC removal improve Pi latency? |
| M4 | Structured-pruned + QAT | selected M3 checkpoint | Is the combined candidate Pareto-superior? |

No candidate is assumed to win. The final model is chosen from M1-M4 on the
validation Pareto frontier, then measured on test and Pi.

### 8.1 PTQ — M1

Use ONNX Runtime static quantization for the CNN.

- calibration data: 1,024 training images, stratified by class and source,
  including supplemental empty images;
- test MinMax, Entropy, and Percentile calibration on validation only;
- prefer per-channel signed INT8 weights and supported activation format;
- export QDQ and, if CPU EP support warrants it, QOperator as an explicitly named
  alternative;
- run ORT quantization debugging if accuracy drops materially;
- record which nodes remain FP32 and why.

Choose the PTQ configuration by validation bobcat F2 subject to measured model
size and operator coverage. Do not select it using Pi test labels.

### 8.2 QAT — M2

QAT initializes from the FP32 M0 checkpoint, not from the PTQ model.

- fake-quantize weights per channel and activations per tensor according to the
  deployable ORT INT8 representation;
- fine-tune 5-10 epochs with low learning rate (`1e-5` to `5e-5` search on val);
- freeze BatchNorm statistics after the initial stabilization epoch unless
  validation proves otherwise;
- export a real INT8 ONNX graph, not a float graph carrying rounded weights;
- verify operator coverage and actual integer kernels in the ORT profile.

Quantization APIs change over time. Phase E0 must prove one end-to-end QAT -> ONNX
-> ORT C++ path and pin the exact compatible PyTorch/torchao/ONNX/ORT versions in
the lockfile before long training begins. Do not silently fall back to a different
quantization meaning.

### 8.3 Structured pruning — M3

Use dependency-aware structured channel pruning. The existing homework
`hw1/src/structured.py` is a starting point, not drop-in code: update the input
shape, ignored layers, objective metric, residual/depthwise dependency handling,
and output classifier.

Procedure:

1. Profile the unpruned model's MACs and parameter count.
2. Run sensitivity analysis using validation bobcat F2/recall, not generic
   accuracy.
3. Create candidates targeting approximately 15%, 30%, and 45% MAC reduction.
4. Physically remove channels; zero masks alone do not qualify as structured
   pruning.
5. Fine-tune each candidate with the same data and loss contract.
6. Export each candidate and confirm that ONNX MACs and tensor shapes changed.
7. Select one M3 candidate on the validation Pareto frontier for QAT.

The report must distinguish parameter reduction, MAC reduction, file size, and
measured latency; they are not interchangeable.

### 8.4 Pruned + QAT — M4

Apply the validated QAT procedure to the selected pruned FP32 model. If M4 is
slower or less accurate than M2, M2 remains the final model. A more complicated
stack does not win by default.

### 8.5 Final-model selection

Before opening test labels, choose the final deploy candidate using validation:

1. Reject candidates that fail export/parity/correctness gates.
2. Prefer candidates satisfying the 90% validation bobcat-recall operating rule.
3. Among them, choose the Pareto-best trade-off between validation bobcat F2,
   ARM64 ORT microbenchmark latency from the E0-compatible harness, and model
   size.
4. If multiple remain, choose the simpler transformation.

Write the decision and rejected alternatives to
`results/model_selection/decision.md` before final test evaluation.

---

## 9. Software and reproducibility contract

### 9.1 Proposed command-line interfaces

Python stages must be runnable as modules, not only notebook cells:

```bash
python -m wildlife_trigger.data.prepare --config configs/data/cct20.yaml
python -m wildlife_trigger.data.audit --config configs/data/cct20.yaml
python -m wildlife_trigger.train --config configs/train/m0_fp32.yaml
python -m wildlife_trigger.optimize.ptq --config configs/optimize/m1_ptq.yaml
python -m wildlife_trigger.optimize.qat --config configs/optimize/m2_qat.yaml
python -m wildlife_trigger.optimize.prune --config configs/optimize/m3_prune.yaml
python -m wildlife_trigger.export --run-id RUN_ID
python -m wildlife_trigger.validate.parity --run-id RUN_ID
python -m wildlife_trigger.evaluate --run-id RUN_ID --split val
python -m wildlife_trigger.calibrate --run-id RUN_ID --target bobcat
```

The C++ application must expose:

```bash
./wildlife_trigger infer --model model.onnx --policy bobcat_v1.yaml --image x.jpg
./wildlife_trigger run-dataset --model model.onnx --policy bobcat_v1.yaml --manifest val.jsonl --output predictions.jsonl
./wildlife_trigger benchmark --model model.onnx --manifest benchmark.jsonl --threads 1 --warmup 100 --iterations 1000
./wildlife_trigger self-test --fixtures tests/fixtures/
```

Exact flag names may evolve, but equivalent non-interactive commands and `--help`
are required.

### 9.2 Configuration and provenance

Every run receives an immutable run ID and writes:

- resolved YAML configuration;
- git commit and dirty/clean status;
- command line;
- UTC timestamp;
- hostname and platform;
- Python/package versions;
- CUDA/GPU details for `gx10` training, or an explicit CPU-only marker for C++
  and ORT correctness runs;
- random seeds;
- dataset manifest hashes;
- checkpoint/model hashes;
- metrics and raw predictions;
- elapsed training time.

No slide number may be reconstructed from memory. Slides and report tables are
generated from versioned result files.

### 9.3 Notebooks

Notebooks are deliverables for inspection and analysis, not the execution engine.

- `notebooks/01_data_audit.ipynb`: class/location/split distributions and data
  examples, reading frozen manifests.
- `notebooks/02_results_analysis.ipynb`: tables, plots, confidence intervals, and
  failure examples, reading raw results.

Training, conversion, evaluation, and benchmarking must remain scriptable from a
clean environment without manually executing notebook cells.

---

## 10. ONNX export and parity gates

Export is part of correctness, not packaging.

### E0 — toolchain spike

Before full training:

1. Export an ImageNet-pretrained MobileNetV2 FP32 model.
2. Produce a small PTQ model.
3. Produce a one-epoch QAT model.
4. On `gx10`, load all three with the exact planned ONNX Runtime C++ build in
   the target-compatible ARM64 environment.
5. Run one image and capture an ORT profile.
6. Pin compatible versions only after this succeeds.

### E1 — preprocessing parity

For at least 20 fixtures covering landscape, portrait, grayscale-looking IR, and
odd dimensions:

- save the canonical Python input tensor;
- compare it with reference C++ preprocessing;
- compare it with fused C++ preprocessing;
- require matching shapes and documented maximum/mean absolute error.

This catches BGR/RGB, interpolation, padding, layout, and normalization bugs.

### E2 — FP32 model parity

PyTorch and ORT FP32 must match on a fixed validation fixture set:

- logits within numeric tolerance;
- identical top-1 class;
- identical bobcat fire/no-fire decisions except samples explicitly identified as
  lying within the numeric tolerance of the threshold.

### E3 — quantized-model validation

For PTQ/QAT, compare the correct framework reference with ORT and report numeric
drift. Exact FP32 equality is not expected, but the model must satisfy:

- valid quantized graph with intended operator coverage;
- validation metrics equal to the recorded quantized candidate;
- ORT Python and ORT C++ predictions/decisions identical on fixtures;
- no silent fallback to an unintended model or preprocessing path.

### E4 — C++ dataset parity

On a validation manifest, Python evaluation and the C++ dataset runner must emit
the same ordered image IDs, labels, target scores within tolerance, trigger
decisions, and confusion matrix.

No model reaching a failed gate may be deployed to the Pi.

---

## 11. C++ application design

Use C++17, CMake, OpenCV, and ONNX Runtime CPU Execution Provider. Pin the ORT
version; do not retain the course container's obsolete hard-coded 1.14.1 package.

### Required components

1. `ModelSession`
   - validates model input/output names, shapes, and types;
   - owns ORT environment/session through RAII;
   - supports configured intra-op thread count;
   - exposes model metadata and timing.
2. `Preprocessor`
   - correct reference implementation;
   - fused single-pass implementation;
   - reusable preallocated input buffer;
   - exact contract from section 5.5.
3. `Policy`
   - loads `mode: any` plus one or more class/threshold entries from YAML/JSON;
   - validates schema, non-empty/unique known animal targets, threshold bounds,
     class mapping, and model hash;
   - emits the shutter decision plus the selected class scores and passing targets.
4. `DatasetRunner`
   - consumes a manifest in deterministic order;
   - writes JSONL predictions and stage timings;
   - continues or fails according to explicit corrupt-image policy.
5. `BenchmarkRunner`
   - warm-up and measured phases;
   - p50/p95/p99 and FPS;
   - decode/preprocess/inference/policy/end-to-end timing;
   - machine-readable output.
6. `SystemMonitor`
   - peak RSS and CPU utilization;
   - temperature/frequency/throttling where exposed;
   - records unavailable sensors rather than inventing values.
7. CLI and logging
   - non-interactive operation;
   - concise human output plus complete JSON/JSONL evidence;
   - non-zero exit codes on invalid input/config/model.

### Required tests

- unit tests for class mapping, single/multi-target `any`, exact threshold
  boundaries, invalid/duplicate/`empty` targets, percentile calculation, and
  manifest parsing;
- preprocessing golden-fixture tests;
- corrupt/missing image behavior;
- wrong model shape and wrong class-map rejection;
- deterministic dataset order;
- integration test: fixture image -> known score/decision;
- repeated benchmark output schema validation.

### C++ optimization experiment

Measure reference versus fused preprocessing while holding the model constant.
The fused path is retained only after parity passes. Report its isolated latency
effect; do not attribute total model speedup to preprocessing.

---

## 12. Raspberry Pi 5 benchmark protocol

### 12.1 Hardware scope

- rented Raspberry Pi 5, BCM2712, CPU-only;
- no physical camera/GPIO/power meter;
- batch size 1;
- exact OS, kernel, compiler, OpenCV, ORT, CPU governor, cooling exposure, and
  available sensors recorded in `results/pi/environment.json`.

### 12.2 Benchmark datasets

Prepare before renting the Pi:

- `benchmark_val_1000.jsonl`: fixed stratified validation subset for performance
  tuning and dry runs;
- full cis-test and trans-test manifests for final accuracy evaluation;
- all images copied or downloadable before the trial begins.

The same ordered benchmark manifest is used for every model.

### 12.3 Timing protocol

For each candidate and thread count `1, 2, 4`:

1. Load model and report load time separately.
2. Warm up 100 inferences.
3. Measure at least 1,000 frames, batch 1.
4. Repeat three times in separate processes.
5. Report p50/p95/p99 for:
   - decode;
   - preprocessing;
   - ORT inference;
   - policy;
   - end-to-end.
6. Report:
   - inference FPS = `N / summed inference time`;
   - end-to-end FPS = `N / wall-clock dataset time`.
7. Log temperature/frequency/throttling before, during, and after each run.
8. Record peak RSS, average CPU utilization, and actual thread configuration.

Do not mix cold-start/model-load latency with resident per-frame latency. Report
both separately.

### 12.4 Fair comparison rules

- same C++ binary and compiler flags;
- same images and order;
- same preprocessing mode;
- same ORT graph optimization level;
- same thread count when comparing models;
- no other heavy workload;
- retain raw per-frame timings;
- report all repetitions, not only the fastest.

Latency is not called energy. Without a power meter, the report may say that
lower active compute time is relevant to energy, but it may not claim measured
joules or battery life.

### 12.5 Five-day trial schedule

| Day | Allowed work |
|---|---|
| 1 | Provision, record environment, install pinned artifacts, smoke test only |
| 2 | Validation benchmark, thread sweep, profiling, final safe runtime fixes |
| 3 | Re-run validation, freeze git commit/models/policies/configuration |
| 4 | Final cis-test/trans-test accuracy and full baseline/optimized benchmark |
| 5 | Exact unchanged repeat, artifact backup, no tuning from test results |

If Core is not dry-run complete before Day 1, do not start the trial.

---

## 13. Execution phases and gates

The implementing agent follows this order.

### Phase A — repository and environment

- create package/application structure from section 14;
- establish Python and C++ test runners;
- create pinned `gx10` training and target-compatible ARM64/C++ environments;
- create lockfiles and environment capture;
- run E0 toolchain spike.

**Gate A:** FP32, PTQ, and minimal QAT models load in ARM64 C++ ORT.

### Phase B — data

- download metadata and required images;
- build frozen manifests and empty supplement;
- implement audit assertions;
- create data-audit notebook and report.

**Gate B:** all D1 assertions pass; manifests and hashes are committed.

### Phase C — FP32 baseline

- implement dataset/transforms/model/training/evaluation;
- train seed 42 baseline;
- calibrate threshold;
- export and pass E1-E4;
- run confirmation seeds 17 and 73.

**Gate C:** reproducible M0 metrics, policy, ONNX, parity, and model card exist.

### Phase D — Core optimizations

- M1 PTQ;
- M2 QAT;
- M3 structured pruning candidates;
- M4 selected pruning + QAT;
- update a single machine-readable comparison table after each candidate.

**Gate D:** final model decision written before test labels are opened.

### Phase E — C++ application

- complete application components and tests;
- reference/fused preprocessing experiment;
- validation dataset parity;
- on `gx10`, run the target-compatible ARM64 build, clean bundle-install test,
  shutter emulation, and dry run of the exact planned Pi commands.

**Gate E:** one command runs the full benchmark unattended and produces schema-
validated results.

### Phase F — Pi trial

- follow the five-day schedule;
- measure M0 plus all deployable Core candidates;
- back up raw data and environment details.

**Gate F:** baseline-vs-optimized Pi evidence contains latency, FPS, resource use,
accuracy, and three repetitions.

### Phase G — analysis and submission

- generate figures/tables from raw results;
- complete report and slides;
- publish code and model artifacts;
- run the submission checklist.

**Gate G:** Core Definition of Done passes.

### Phase S — optional Stretch KD

May begin only after Gate G. It must be a separate experiment/config/result block
and must not rewrite the Core headline result.

---

## 14. Repository and deliverable layout

The public repository should converge on this structure:

```text
final_project/
  DESIGN.md                         # this execution specification
  README.md                         # quick start, results summary, public links
  SUBMISSION.md                     # final checklist and canonical URLs
  LICENSE
  CITATION.cff
  pyproject.toml
  requirements.lock                # or uv.lock/conda lock, exact versions
  configs/
    data/cct20.yaml
    train/m0_fp32.yaml
    optimize/m1_ptq.yaml
    optimize/m2_qat.yaml
    optimize/m3_prune.yaml
    optimize/m4_pruned_qat.yaml
    runtime/pi.yaml
  src/wildlife_trigger/
    data/
    models/
    train.py
    evaluate.py
    calibrate.py
    export.py
    optimize/
    validate/
    reporting/
  cpp/
    CMakeLists.txt
    include/
    src/
    tests/
  scripts/
    setup_gx10.sh
    download_data.sh
    run_core_pipeline.sh
    build_cpp.sh
    package_pi.sh
    run_pi_benchmarks.sh
    generate_submission.sh
  tests/
    python/
    fixtures/
  notebooks/
    01_data_audit.ipynb
    02_results_analysis.ipynb
  data/
    manifests/                      # committed small metadata, no image archive
    README.md
  artifacts/
    policies/
    manifests/
    model_cards/
    checksums.sha256
    README.md                       # model download/GitHub Release links
  deploy/
    pi/
      install.sh
      run_demo.sh
      manifest.json
      README.md
  results/
    data_audit/
    training/
    evaluation/
    parity/
    model_selection/
    pi/
    figures/
  report/
    final_report.md
    final_report.pdf
  slides/
    final_presentation.pptx
    final_presentation.pdf
  demo/
    README.md
    sample_output.txt
```

Large datasets are never committed. ONNX/checkpoint files should be published via
GitHub Releases or Git LFS, with hashes and stable download links in
`artifacts/README.md`.

### 14.1 Mandatory submission deliverables

The assignment explicitly requires a codebase and formal slide deck. The project
will submit the following complete package:

1. **Public GitHub repository**
   - clean tagged release, e.g. `v1.0-final`;
   - reproducible setup and commands;
   - repository URL included in README, report, first slide, and final slide;
   - exact commit hash recorded in all benchmark environment files.
2. **Source code**
   - Python training/conversion/evaluation;
   - C++ inference application and tests;
   - configs and automation scripts.
3. **Models and policies**
   - FP32 baseline ONNX;
   - final optimized ONNX;
   - optional intermediate deployable models;
   - primary bobcat policy YAML, final-model threshold catalog, and one validated
     multi-target example policy;
   - checksums and model cards.
4. **Raspberry Pi deployment bundle**
   - release archive containing the C++ executable, required shared libraries or
     reproducible installer, final ONNX model, class map, policy, sample manifest,
     `run_demo.sh`, and checksums;
   - bundle must install and run on a clean compatible ARM64 environment without
     accessing the training machine.
5. **Raw evidence**
   - training histories;
   - validation/test predictions;
   - parity reports;
   - raw Pi timings and system logs;
   - generated tables and figures.
6. **Formal slide deck**
   - editable `.pptx`;
   - exported `.pdf`;
   - repository link and QR code.
7. **Final report**
   - Markdown source and PDF;
   - methods, reproducibility, results, critical analysis, limitations, references;
   - public repository and release links.
8. **Two notebooks**
   - data audit;
   - results analysis;
   - both execute from frozen artifacts without hidden state.
9. **Submission manifest**
   - `SUBMISSION.md` lists every link/file, the final commit, release, model hashes,
     headline metrics, and reproduction commands.

### 14.2 Recommended optional deliverable

A 30-60 second terminal demo recording is useful for the presentation but is not
required. It must show the rented Pi hostname/platform, C++ executable, several
frames, trigger decisions, and summary timings. Do not let video production delay
mandatory artifacts.

---

## 15. Report specification

Target length: approximately 8-12 pages excluding appendices. Generate tables and
plots from raw result files.

Required sections:

1. Problem and product narrative.
2. Course objective and engineering question.
3. Dataset, splits, empty supplement, and leakage controls.
4. MobileNetV2 baseline and training recipe.
5. PTQ, QAT, structured pruning, and theoretical expectations.
6. C++ application and correctness/parity tests.
7. Raspberry Pi benchmark protocol.
8. Accuracy and system-performance results.
9. Ablation and Pareto analysis.
10. What worked well.
11. What did not work as expected.
12. Hardware/software bottlenecks.
13. Limitations: remote Pi, no camera/GPIO/power measurement, CCT domain.
14. Concrete next steps, including physical device integration and optional KD.
15. Reproducibility statement, public repository link, release tag, and commit.
16. References.

The report must separate measured, published, and estimated values. Estimates do
not appear in the headline results table.

---

## 16. Slide-deck specification

Suggested 10-12 slide narrative:

1. Title, one-line product, author, public repository/QR.
2. Why Edge AI: offline target-species shutter decision.
3. Assignment mapping and engineering question.
4. CCT-20, bobcat target, cis/trans, leakage controls.
5. Baseline model and C++ deployment pipeline.
6. Optimization ladder: PTQ, QAT, structured pruning.
7. Correctness and reproducibility gates.
8. Accuracy results at the bobcat operating point.
9. Pi baseline-vs-optimized latency/FPS/resource results.
10. Pareto chart and bottleneck/profile evidence.
11. What worked, what failed, limitations.
12. Final demo/result, next steps, repository/QR.

The presenter must be able to explain every C++ preprocessing step, quantization
choice, pruning result, threshold, and benchmark number.

---

## 17. Required result tables and figures

At minimum generate:

1. Dataset split/class/location audit table.
2. Training curves for M0 and final optimized model.
3. Validation model-selection table.
4. Cis/trans target metrics with sequence-bootstrap confidence intervals.
5. Per-class confusion matrices for M0 and final.
6. Accuracy vs model size/MACs table.
7. Pi table: model, precision, file size, threads, p50/p95/p99 inference,
   end-to-end FPS, peak RSS, CPU utilization, temperature/throttling.
8. Pareto plot: trans bobcat recall or F2 vs Pi p95 latency.
9. Reference vs fused C++ preprocessing latency.
10. At least six representative failure cases with scores and explanations.

All plots include units, sample counts, split, model ID, and commit/run ID.

---

## 18. Risk register and decision rules

| Risk | Decision rule |
|---|---|
| Empty supplement introduces bias | Keep it location-disjoint, deterministic, documented, and identical across all models |
| Trans bobcat recall is poor | Report honestly; do not train on trans-test. Stretch KD is allowed only after Core completion |
| PTQ loses accuracy | Use QAT; use quantization debugging; keep PTQ as a negative result |
| QAT export/runtime path is unstable | E0 before training; pin compatible versions; fail early rather than improvise during Pi trial |
| Structured pruning does not speed MobileNetV2 | Show real MAC reduction and measured lack of speedup; final model may be unpruned QAT |
| C++ preprocessing silently differs | E1 golden tensor fixtures block deployment |
| ORT C++ differs from Python | E3/E4 block deployment |
| Remote Pi hides sensors/governor | Record unavailable values; use exposed `/proc`/`sysfs`; do not fabricate resource/energy claims |
| Five-day trial is spent debugging | Exact ARM64 dry run and one-command benchmark are prerequisites |
| Scope creep | Only Core until Gate G; only crop-teacher KD afterward |
| Public repo cannot store large models | GitHub Release/LFS plus hashes and download script |

---

## 19. Core Definition of Done

Core is complete only when every item is true:

### Data and ML

- [ ] Data manifests, hashes, distributions, and leakage assertions pass.
- [ ] Empty supplement is location-disjoint and reproducible.
- [ ] M0, M1, M2, M3, and M4 results exist or a technically justified failed
      candidate is preserved and documented.
- [ ] Thresholds use validation only.
- [ ] Final model decision predates test evaluation.
- [ ] Cis/trans metrics and confidence intervals exist.

### Deployment and C++

- [ ] Baseline and final ONNX models pass preprocessing/model/C++ parity.
- [ ] C++ CLI, dataset runner, benchmark harness, and self-tests pass.
- [ ] The same final model passes bobcat-only and multi-target policy tests without
      another model inference per frame.
- [ ] ARM64 dry run succeeds from a clean environment.
- [ ] Pi baseline and optimized runs use the same application and protocol.
- [ ] Latency, FPS, RSS, CPU utilization, model size, and available thermal data
      are recorded with raw evidence.

### Submission

- [ ] Public repository is clean, tagged, and accessible.
- [ ] Model release links and checksums work.
- [ ] Raspberry Pi deployment bundle installs and runs from its documented entry
      point.
- [ ] README reproduces setup, training, export, C++ build, demo, and benchmarks.
- [ ] Data-audit and results notebooks execute cleanly.
- [ ] Final report Markdown/PDF includes the public repository URL.
- [ ] Final slides PPTX/PDF include the repository URL/QR.
- [ ] `SUBMISSION.md` points to every canonical artifact.
- [ ] Placeholder `REPO_URL` values are replaced by the public repository and
      tagged-release URLs.
- [ ] All headline numbers trace back to machine-readable raw results.
- [ ] Slides explicitly answer what worked, what failed, bottlenecks, and next
      steps.

Only after all boxes are checked may Stretch KD begin.

---

## 20. Stretch specification — crop-teacher KD only

This section is dormant until Core completion.

### Experiment

| Row | Student input | Extra signal |
|---|---|---|
| S0 | Full frame | Core FP32 baseline |
| S1 | Full frames plus GT crops as training augmentation | No teacher |
| S2 | Full frame | Soft animal-class distribution from crop teacher |

The crop teacher trains only on the 15 animal classes because empty frames have no
box. If the Core student retains 16 outputs, KD is applied only on non-empty
samples and compares the teacher distribution with the student's animal logits
after excluding and renormalizing the `empty` dimension.

Required controls:

- same train split, optimizer budget, augmentations, seed, and student
  initialization for S0/S1/S2;
- explicit rule for multiple boxes: default to a padded union box; record and test
  alternatives only if needed;
- CE remains active alongside KD;
- tune KD temperature and weight on validation only;
- S2 must beat S1 to justify the teacher.

If S2 fails to beat S1, report the null result and keep the Core final model.

---

## 21. References and authoritative sources

- Course assignment: `Final Project TASK.docx` in this directory.
- Beery, Van Horn, Perona, *Recognition in Terra Incognita*, ECCV 2018:
  https://openaccess.thecvf.com/content_ECCV_2018/papers/Beery_Recognition_in_Terra_ECCV_2018_paper.pdf
- CCT and CCT-20 downloads, metadata, and license:
  https://lila.science/datasets/caltech-camera-traps
- CCT dataset description:
  https://beerys.github.io/CaltechCameraTraps/
- MobileNetV2:
  https://arxiv.org/abs/1801.04381
- ONNX Runtime quantization:
  https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- ONNX Runtime C/C++ API:
  https://onnxruntime.ai/docs/get-started/with-c.html

Any implementation-time documentation lookup must prefer primary/official
sources and record the versions actually used.
