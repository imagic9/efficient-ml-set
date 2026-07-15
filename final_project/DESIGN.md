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
| Benchmarking and metrics, 10 pts | Baseline and optimized latency/FPS/resources on the same Pi 5; frozen full-test C++ accuracy plus target-hardware parity |
| Analysis and presentation, 10 pts | Reproducible tables and plots; what worked; what failed; bottlenecks; concrete next steps |

The project is successful even if an optimization fails to improve speed or
accuracy, provided the failure is measured correctly and explained.

---

## 3. Scope

### 3.1 Core — required for submission

1. Prepare a leakage-safe wildlife dataset with CCT-20 cis/trans splits.
2. Train one FP32 full-frame MobileNetV2 classifier.
3. Calibrate a primary bobcat shutter threshold using validation data only and
   implement a generic policy over the known animal outputs. The final catalog
   determines which classes have enough validation support to be selectable.
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
MobileNetV2, 16 logits: 14 animals + car + empty
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
configuration file, not a separate model. It selects one or more animal outputs
that carry a usable threshold in the final 14-entry catalog and fires when any
selected class passes its own threshold. Runtime policies use JSON so the C++
bundle can vendor one pinned header-only parser instead of relying on a system YAML
library:

```json
{
  "schema_version": 1,
  "policy_id": "bobcat_coyote_v1",
  "model_sha256": "MODEL_SHA256",
  "class_map_sha256": "CLASS_MAP_SHA256",
  "mode": "any",
  "targets": [
    {"class": "bobcat", "threshold": 0.42},
    {"class": "coyote", "threshold": 0.55}
  ]
}
```

The numbers above illustrate the numeric schema only; generated policies must
replace them with measured validation-calibrated values.

`car` and `empty` are model classes but are not selectable wildlife targets.
`mode: any` is the only Core combination mode. The loader rejects an empty target
list, duplicate/unknown/non-animal classes, classes carrying no calibrated
threshold in the model's catalog, thresholds outside `[0, 1]`, unsupported modes,
and model/class-map hash mismatches.

Vendor a pinned `nlohmann/json` single header plus its license under
`cpp/third_party/`; record version and SHA-256 in the dependency manifest. Policy
parsing happens once at startup, never in the per-frame hot path. No network fetch
or system JSON/YAML development package may be required to build the release
bundle.

The submission's primary evaluated policy remains the bobcat-only
`bobcat_v1.json`. The same final model must also accept a multi-target example
policy without another inference or model reload. This adds configuration-level
modularity without adding networks or changing the benchmark question.

The calibration pipeline produces a model-bound catalog entry for every one of the
14 animal classes. A catalog entry always records support and status, but it
contains a selectable numeric threshold only when validation support can define a
defensible operating point. Measured positive support on
`cis-val-clean + trans-val`:

| Class | cis-val-clean (img/seq) | trans-val (img/seq) | Catalog status |
|---|---:|---:|---|
| bobcat | 144 / 50 | 793 / 265 | `two_domain_calibrated` — primary target |
| opossum | 474 / 158 | 444 / 148 | `two_domain_calibrated` |
| rabbit | 392 / 132 | 70 / 25 | `two_domain_calibrated` |
| coyote | 258 / 102 | 51 / 17 | `two_domain_calibrated` |
| raccoon | 165 / 55 | 129 / 43 | `two_domain_calibrated` |
| cat | 189 / 63 | 72 / 24 | `two_domain_calibrated` |
| dog | 127 / 43 | 96 / 32 | `two_domain_calibrated` |
| skunk | 28 / 12 | 63 / 21 | `two_domain_calibrated` |
| bird | 60 / 22 | 7 / 3 | `two_domain_calibrated` — weak trans support |
| squirrel | 205 / 69 | **0 / 0** | `single_domain_fallback` — cis only |
| rodent | 135 / 45 | **0 / 0** | `single_domain_fallback` — cis only |
| badger | 1 / 1 | **0 / 0** | **`unavailable_low_support`** |
| deer | **0 / 0** | **0 / 0** | **`unavailable_no_support`** |
| fox | **0 / 0** | **0 / 0** | **`unavailable_no_support`** |

`deer` and `fox` have zero validation positives in both domains, so no threshold
exists for them at any confidence level. `badger` has one image in one sequence,
which cannot define an operating point. The catalog therefore contains 14 status
entries but only **11 selectable targets**: nine two-domain thresholds and two
explicit single-domain fallbacks. `badger`, `deer`, and `fox` have
`threshold: null`; the policy loader rejects them rather than silently inventing
or accepting an unvalidated threshold.

For a class represented in both validation domains, apply the two-domain rule from
section 6.3. For `squirrel` and `rodent`, optimize pooled-validation F2 so negative
frames from both domains still constrain false fires, and record the result as a
single-domain-positive fallback; do not call its recall trans-validated. Too little
total support, as for `badger`, produces no threshold. Never invent a threshold or
claim validated recall without positive examples. A generated multi-target policy
must store its combined validation trigger metrics because false-fire rates
increase when multiple targets are joined with `any`.

Catalog status is one of `two_domain_calibrated`, `single_domain_fallback`,
`unavailable_low_support`, or `unavailable_no_support`. Selectable entries carry a
numeric threshold plus calibration-domain/support/metric metadata; unavailable
entries carry `threshold: null` and a reason. The catalog is bound to the final
model and class map by hash, and the loader accepts targets only from the two
selectable statuses. Persist it as
`artifacts/policies/threshold_catalog.json`; generated policies may reference only
entries from that exact model-bound catalog.

A new species outside the 16-class map requires labelled data, classifier-head
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
`gx10` inside a clean target-compatible ARM64 container. Record the target
`/etc/os-release`, `uname -a`, compiler, and `ldd --version`; pin the container base
image by digest to the same distro release and a glibc no newer than the target.
Before packaging, run `ldd` plus ELF/required-`GLIBC_*` symbol inspection on the
executable and every bundled shared library, then execute the clean install/smoke
test. If exact compatibility cannot be proved before rental, the deployment bundle
must include pinned source/build automation and compile on the Pi during
provisioning.

The rented Raspberry Pi target (Pi 5 preferred, RPi 4 contingency) is the only
second execution environment. It is used only for Phase F provisioning,
target-hardware smoke/parity checks, and final CPU
performance/resource measurements. Frozen full-test accuracy runs on `gx10` with
the exact C++/ORT artifacts after the Pi validation freeze. Results from `gx10` may be used
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

#### Required downloads and their resolution

| Source | Size | Contents |
|---|---:|---|
| `eccv_18_all_images_sm.tar.gz` | 6 GB | all 57,864 CCT-20 frames, **downsized to a maximum of 1024 px on a side** |
| `eccv_18_annotations.tar.gz` | 3 MB | CCT-20 split metadata |
| `caltech_camera_traps.json.zip` | 9 MB | full-CCT image-level metadata, used only to select the empty supplement |
| per-image CCT paths | ~2.1 GB | the 5,000 selected empty frames, served **at original resolution only** |

Total data acquisition is approximately **8.1 GB**. The 105 GB `cct_images.tar.gz`
archive is never downloaded. `caltech_bboxes_20200316.json` (35 MB) is required
only if the Stretch KD experiment in section 20 is ever unlocked.

**The benchmark images and the supplement therefore arrive at different
resolutions.** CCT-20 is capped at 1024 px per side; per-image downloads are not.
Section 5.2 mandates the correction, and section 5.5 states the input geometry
against the downsized frames the pipeline actually consumes. B0 must record the
observed dimension distribution of every downloaded split rather than inheriting a
number from the paper or from this document.

CCT-20 contains **14 animal categories, `car`, and `empty`** (16 total). `car` is
kept as a distractor class but is not a selectable wildlife target. The exact
category IDs are non-contiguous, so the model order is defined once in
`configs/data/classes.yaml` and must never be inferred from dictionary iteration
order.

### 5.1.1 Known official sequence overlap and cleaned development split

The official CCT-20 metadata contains one known sequence-level overlap:

- train and cis-val share exactly **224 `seq_id` values**;
- those sequences account for **270 cis-val images** (7.7%);
- **10 of the 270** are bobcat images;
- every other cross-split sequence intersection is zero.

Never rewrite the official split. Produce an immutable derived manifest
`cis_val_clean.jsonl` by removing from cis-val every image whose `seq_id` occurs
in train. Its expected size is **3,214 images**, including **144 bobcat images**.
Use `cis-val-clean`, not official cis-val, for checkpoint selection, preprocessing
selection, pruning/quantization choices, and threshold calibration. Report both
clean and official cis-val results so the leakage effect remains visible.

### 5.2 Required empty-frame supplement

The current official CCT-20 `train_annotations.json` contains no empty-labelled
training images, although validation and test do. A production-valid classifier
therefore needs a small empty training supplement.

Build `cct_empty_train_v1` from full CCT metadata using these rules:

1. Collect the set of all 20 CCT-20 location IDs.
2. Candidate image label must be exactly `empty`.
3. Candidate location must be disjoint from all 20 CCT-20 locations.
4. Candidate image ID and `seq_id` must be absent from every CCT-20 split.
5. Select **5,000 images**, stratified across locations and sequences, with fixed
   seed `42`. Avoid letting one camera dominate the sample.
6. Download only selected images through the LILA per-image cloud paths; do not
   download the full 105 GB archive.
7. **Downsize every selected image to a maximum of 1024 px on a side**, matching
   the CCT-20 `_sm` archive, before it enters the training set. Record the
   resampling filter and JPEG quality in the data config, and store both the
   original and downsized checksums.
8. Save a manifest containing image ID, location, sequence ID, source URL,
   relative path, label, original dimensions, downsized dimensions, and both
   checksums.
9. Use this supplement for training only. Do not create a new test set from it.

Step 7 is not cosmetic. Per-image CCT downloads are served at original resolution
(~2048x1494) while every CCT-20 split is capped at 1024 px per side. Skipping it
makes `empty` the only training class carrying double resolution and its own JPEG
recompression signature — a feature perfectly correlated with the label.

That failure is silent and it corrupts the headline number. Validation and test
contain **only** `_sm` frames, so the shortcut is absent at evaluation time: a model
that learned "2048-px artifacts mean empty" cannot recognize `empty` on
`cis-val-clean` or `trans-val`, and the bobcat false-fire rate is inflated exactly
where it is measured. The `A-empty-5k` ablation would then read as "the supplement
barely helped" and the cause would be misattributed to the location-disjoint rule.

Because the supplement is already location-disjoint by necessity (rule 3), the
model sees `empty` only on unfamiliar backgrounds. That is a second feature
correlated with the same label, and it is unavoidable: within the 10 cis
locations, every `empty` frame in full CCT is already spent in cis-val and
cis-test, so a background-matched supplement does not exist. Rule 7 removes the
one confound that *is* removable; the report must state the remaining one.

**Shortcut probe.** Before training, train a small binary classifier to separate
supplement frames from CCT-20 `_sm` frames. Near-chance accuracy means the
resolution/encoding confound is closed. High accuracy means it is live: record the
value, fix the downsizing procedure, and do not proceed on the assumption that the
supplement is clean.

This changes the training recipe relative to the paper, so the report must say so.
All project models use the same augmented training dataset, preserving a fair
baseline-versus-optimized comparison.

#### Empty-supplement ablation

Before freezing M0, run one matched data ablation:

- `A-empty-0`: a 15-output non-empty classifier (14 animals + `car`) trained on
  official train only;
- `A-empty-5k`: the planned 16-output classifier trained with
  `cct_empty_train_v1`.

Use the same backbone initialization, augmentations, seed, and validation protocol.

**This ablation cannot change the deployed head.** It tests whether the empty
supplement earns its place in the training data; the Core model is 16-output in
every case. `A-empty-0` is a diagnostic arm, not a deployment candidate: a shutter
trigger that has no `empty` output must resolve every empty frame onto some animal
or `car` logit, and CCT-20's own splits are 3.5-28.6% empty. The 15-output head
exists only because a 16th output trained on zero positives would be degenerate and
would confound the data question with a dead-logit question.

If `A-empty-0` wins, the finding is "the 5,000-image supplement does not pay for
its animal-exposure cost at fixed compute" — and the response is to revisit the
supplement size or sampling, not to ship a 15-output model. Every other statement
of 16 outputs in this document stands unconditionally.

**Match the budget in optimizer steps, not epochs.** The supplement changes the
training set from 13,546 to 18,546 images (+36.9% steps per epoch), so an
epoch-matched comparison would confound "empty data helps" with "this arm simply
trained 37% longer" rather than isolate the data effect under fixed compute. Fix
the total step count, and record steps, effective epochs, total images-seen, and
non-empty images-seen for both arms. The last value makes the lower animal
exposure in the compute-matched supplement arm explicit; an exposure-matched
sensitivity run is optional only if the primary result is ambiguous.

Compare bobcat recall/F2, bobcat false-fire rate on cis and trans empty frames
separately, calibration stability, and empty top-1 recall where defined. This
ablation tests the supplement assumption; it does not add a new deployment
candidate or alter the M0-M4 compression ladder.

### 5.3 Dataset assertions — hard data gate

Before training, automated tests must prove:

- split image counts match the table above;
- category set is exactly the configured 16 classes;
- category names are exactly 14 animals plus `car` and `empty`;
- every image ID is unique within a split;
- no image occurs in more than one split;
- train/cis-val overlap is exactly the known 224 sequences / 270 images / 10
  bobcat images, and `cis-val-clean` contains 3,214 images / 144 bobcat images;
- all other required cross-split sequence intersections are zero, including
  train/cis-test, train/trans-val, train/trans-test, and cis-val/cis-test;
- train locations are disjoint from trans-val and trans-test locations;
- supplemental-empty IDs, sequences, and locations are disjoint from CCT-20;
- every supplement image is at most 1024 px on its long side after section 5.2
  step 7, and its manifest records original and downsized dimensions plus both
  checksums;
- the observed image-dimension distribution of every split is recorded, and the
  supplement's distribution is consistent with the CCT-20 `_sm` splits;
- the supplement-versus-CCT-20 shortcut probe scores at or near chance; a
  materially higher score is recorded and blocks training;
- every manifest stores the complete ordered label set for each image;
- observed distinct-class multi-label image counts match 7 / 0 / 1 / 61 / 9 for
  train / cis-val / cis-test / trans-val / trans-test;
- the animal classes with zero validation positives on `cis-val-clean + trans-val`
  are exactly `deer` and `fox`; `badger` support is exactly one image / one
  sequence; all three are recorded with null thresholds and unavailable statuses;
- every manifest path exists and every checksum matches;
- class and location distributions are written to `results/data_audit/`.

If any assertion fails, stop. Do not train around a split problem.

These counts are fingerprints of a specific upstream download, not invariants of
the universe. If one fails, first check the recorded source hashes from section
5.1: a hash change means LILA republished the metadata and the expected numbers
must be re-derived and re-reviewed. Never edit an expected number to make a
failing assertion pass.

### 5.4 Test-set discipline

`cis-test` and `trans-test` model results are sealed until models, thresholds,
code, runtime configuration, and thread count are frozen. Mechanical data-audit
code may parse labels to verify schema/counts/multi-label rules and then write
sealed manifests, but it must not compute model metrics or expose predictions for
development. All model/runtime decisions use only train, cis-val-clean, trans-val,
and an unlabeled validation benchmark manifest.

Beyond the mechanical audit above, final test labels may be used only by the final
evaluation job. Rerunning an unchanged frozen artifact for reproducibility is
allowed; changing anything after viewing test results is not.

### 5.4.1 Multi-label annotation rule

The model remains a single-label softmax classifier, but the source annotations
contain images with more than one distinct class. The executable rule is:

1. Manifests store `labels` as the complete sorted set of distinct classes.
2. Images with exactly one distinct class receive that class as `primary_label`.
3. The seven multi-label train images are excluded from weighted-CE training and
   recorded in the data audit; do not choose an arbitrary primary label.
4. Multi-label validation/test images remain in every target-presence metric. An
   image is positive for bobcat whenever `bobcat` is in `labels`, regardless of
   the other labels.
5. Single-label confusion matrices and macro F1 exclude multi-label images and
   state the excluded count. Per-target recall/precision/F2 and policy metrics do
   not exclude them.

This rule makes product recall unambiguous without claiming simultaneous
multi-species recognition from a softmax model.

### 5.5 Preprocessing

The network must see the complete frame; do not use a center crop that can remove
a small animal.

The provisional Core input is **256x192 (width x height)**. This is nearly the
same pixel budget as 224x224 but matches the dominant CCT aspect ratio much more
closely. The dominant original CCT frame is 2048x1494 (91% of all images); the
`_sm` archive this pipeline consumes caps the long side at 1024, so the dominant
frame we actually decode is **1024x747**. Measured on that frame:

| Input | Real content | Pixel utilisation | Grey padding | Tensor px | Linear scale |
|---|---|---:|---:|---:|---:|
| 224x224 | 224x163 | 72.8% | **27.2%** | 50,176 | 0.2188 |
| 256x192 | 256x187 | **97.4%** | 2.6% | 49,152 | **0.2500** |

Utilisation, padding, and tensor pixels depend only on the aspect ratio, so
downsizing leaves them unchanged and the comparison is unaffected — only the
absolute scale factors double relative to the original frames. B0 must confirm the
observed dimension distribution before this table is treated as measured.

The square letterbox spends over a quarter of every inference on grey bars that
carry no information. 256x192 buys **14% more linear resolution on the animal with
2% fewer input pixels** and should reduce spatial MACs by approximately the same
amount; exported-model MACs remain the authoritative compute measurement. Both
dimensions stay divisible by 32, so MobileNetV2's five downsampling stages still
produce a clean 8x6 feature map.

256x192 also aligns with the JPEG decoder, which is a second and independent
argument for it. Libjpeg can scale during decode at 1/2, 1/4, and 1/8 for close to
free, and `1024 / 4 = 256` while `747 / 4 = 186.75`, so a 1/4 reduced decode emits
**256x187 — the network input, with no resize step at all**. Only the 5-row grey
pad remains. The square 224x224 input has no such alignment: it would decode to
256x187 and then pay a downscale to 224x163. Section 11 treats reduced decode as a
first-class latency candidate on this basis.

Before M0 is frozen, run the matched input-shape control described below; the
winning fixed shape becomes part of the immutable model/preprocessing contract for
every M0-M4 candidate.

Canonical preprocessing for a configured fixed `(width, height)`:

1. Decode JPEG as 8-bit BGR with OpenCV.
2. Convert BGR to RGB.
3. Resize while preserving aspect ratio to fit inside the configured width/height.
4. Center-pad the remaining rows/columns using RGB value `(114, 114, 114)`.
5. Convert to float32 and divide by 255.
6. Normalize with ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation
   `(0.229, 0.224, 0.225)`.
7. Convert HWC RGB to contiguous NCHW; provisional shape is `[1, 3, 192, 256]`.

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

#### Input-shape control before M0

Run the controls sequentially so they require three training runs, not a full 2x2
factorial experiment:

1. run both empty-supplement arms at provisional 256x192 and select the data/head
   contract;
2. reuse the winning 256x192 run as the landscape reference;
3. train one additional 224x224 run with that same winning data/head contract.

The resulting shape comparison contains exactly two candidates under the same
seed and training budget:

- `I-square`: 224x224 aspect-preserving letterbox;
- `I-landscape`: 256x192 aspect-preserving letterbox.

Select on cis-val-clean/trans-val bobcat F2 and sequence-balanced recall, with
MACs and real-pixel utilization reported. Prefer 256x192 when the target metrics
are statistically indistinguishable; fall back to 224x224 if the landscape input
materially hurts them. Only the selected shape proceeds to M0-M4. A 320x240
candidate is permitted only as a documented contingency if neither candidate can
meet the validation bobcat-recall operating rule; it is not a default search axis.

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

Calibrate the bobcat threshold on `cis-val-clean + trans-val` only. Treat the two
validation domains separately so the larger bobcat count in trans-val does not
hide a cis failure. Trans-val contains only one camera location, so it is evidence
for that unseen location, not a universal estimate across unseen cameras.

Bursts contain correlated near-duplicate frames. For threshold selection, compute
**sequence-balanced recall**: calculate frame recall inside each positive
`seq_id`, then average those sequence values with equal weight. Keep ordinary
frame-level recall as the primary reported product metric.

Do not exclude short positive sequences or down-weight them by burst length: a
one-frame visit is a real event the trigger must not silently ignore. To expose
metric stability without changing the registered threshold rule, also report the
positive-sequence length distribution, recall for length strata `1-2`, `3-5`, and
`>5` frames where supported, and **event capture rate** = positive sequences with
at least one trigger / all positive sequences. Sequence-cluster bootstrap intervals
remain the uncertainty measure.

Primary rule:

1. Search all unique bobcat scores.
2. Choose the largest threshold for which sequence-balanced bobcat recall is at
   least 90% on both cis-val-clean and trans-val.
3. If no non-trivial threshold meets both constraints, choose the threshold
   maximizing the mean frame-level F2 across cis-val-clean and trans-val and
   record which sequence-balanced recall constraint was not met.
4. Save the threshold, calibration metrics, score distribution, and dataset hash
   to `artifacts/policies/bobcat_v1.json`.
5. Bootstrap complete `seq_id` clusters within each validation domain and save the
   95% interval/distribution of the selected threshold. The deployed point
   threshold still comes from the full cleaned validation data, not a bootstrap
   replicate.

The threshold is calibrated separately for each candidate model because
quantization changes score distributions. For the final baseline-versus-optimized
comparison, report both:

- each model at its own calibrated operating point;
- every model at the FP32 baseline threshold, to expose calibration drift.

After the final model is selected, run the catalog builder for all 14 animal
classes as described in section 4. It emits a threshold for the 11 selectable
classes and an explicit null-threshold status for `badger`, `deer`, and `fox`.
These secondary thresholds do not affect model selection or replace the bobcat
evaluation.

### 6.4 Primary accuracy metrics

Report separately for cis-test and trans-test:

- bobcat recall — primary product metric;
- bobcat precision;
- bobcat F2;
- false-fire rate = false triggers / non-bobcat frames;
- fire rate = all triggers / all frames;
- per-class support, recall, precision, and F1 for every class with positives;
- event capture rate and positive-sequence recall by the registered length strata;
- support-aware macro F1 over classes with at least 20 positive images and at
  least 5 positive `seq_id` values in that split; list the included classes;
- confusion matrix;
- 95% confidence intervals by bootstrap over `seq_id`, not individual frames.

Report frame-level and sequence-balanced bobcat recall, threshold uncertainty,
and per-location bobcat recall on trans-test. Multi-label images follow section
5.4.1. Do not claim a universally calibrated probability from softmax scores.

---

## 7. Baseline model

### 7.1 Architecture

**MobileNetV2, width multiplier 1.0, ImageNet-pretrained.** Replace the classifier
with a 16-output linear layer. The provisional fixed input is 256x192; the
224x224-vs-256x192 control in section 5.5 freezes the final Core input before M0.

Reasons:

- designed for resource-constrained inference;
- mature pretrained weights;
- stable ONNX export and CPU operator support;
- depthwise-separable convolutions provide a meaningful structured-pruning case;
- small enough for repeated Pi benchmarks;
- the course container already proves the MobileNetV2/ONNX/C++ toolchain.

Core does not perform neural architecture search. HW4 remains useful background:
MobileNetV2's inverted residual blocks are an efficient search-space choice, and
its validation-only search discipline, weight-sharing caveats, and proxy-rank
analysis inform this project. A new supernet is still excluded because real Pi
latency is unavailable until the short rental window, `gx10` latency is not a
reliable Cortex-A76 ranking proxy, and weight-sharing proxy noise would add a
second model-selection problem before the required pruning/quantization/C++ work
is complete. After Core, a bounded width/input-shape study could reuse HW4 ideas;
it must not replace the frozen Core result.

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
- checkpoint selection score: mean of cis-val-clean and trans-val bobcat F2,
  with sequence-balanced bobcat recall used as the first tie-break and
  support-aware macro F1 as the second;
- save last and best checkpoints plus full optimizer/scheduler state.

Do not tune on overall accuracy alone.

### 7.3 Baseline artifacts

The baseline is not complete until all exist:

- PyTorch checkpoint;
- FP32 ONNX export;
- training history JSON/CSV;
- validation predictions and logits;
- calibrated policy JSON;
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

No candidate is assumed to win. Validation produces a deployable shortlist from
M1-M4; real Pi validation latency selects the final optimized model before any
test labels are opened.

The provisional export contract is ONNX **opset 17** for M0-M4. P0 must verify
that the pinned PyTorch exporter, ONNX checker, quantization path, and C++ ORT
runtime all accept it. Use one opset across the comparison; change it only if P0
proves a concrete incompatibility, then document and re-run every export/parity
fixture. Opset 9 from the legacy course spike is forbidden.

### 8.1 PTQ — M1

Use ONNX Runtime static quantization for the CNN.

Pre-registered expectation: MobileNetV2 PTQ may lose more accuracy than a
standard-convolution model because depthwise layers and activation ranges are
quantization-sensitive. A measured PTQ failure is evidence, not an emergency
reason to change the test protocol; QAT is the planned recovery path.

- calibration data: 1,024 training images, stratified by class and source,
  including supplemental empty images;
- test MinMax, Entropy, and Percentile calibration on validation only;
- prefer per-channel signed INT8 weights and supported activation format;
- use S8S8 QDQ as the primary static-quantization representation and, if measured
  ARM CPU EP support warrants it, QOperator as an explicitly named alternative;
- run ORT quantization debugging if accuracy drops materially;
- save the quantized and session-optimized graphs, ORT profile, operator/data-type
  coverage, and record which nodes remain FP32 and why. Do not rely on one
  version-specific kernel name such as `QLinearConv` as the sole proof of INT8
  execution.

Choose the PTQ configuration by validation bobcat F2 subject to measured model
size and operator coverage. Do not select it using test labels.

### 8.2 QAT — M2

QAT initializes from the FP32 M0 checkpoint, not from the PTQ model.

- fake-quantize weights per channel and activations per tensor according to the
  deployable ORT INT8 representation;
- fine-tune 5-10 epochs with low learning rate (`1e-5` to `5e-5` search on val);
- freeze BatchNorm statistics after the initial stabilization epoch unless
  validation proves otherwise;
- export a real INT8 ONNX graph, not a float graph carrying rounded weights;
- verify integer execution using the exported/optimized graphs, operator/data-type
  coverage, ORT profile, and target latency together.

Quantization APIs change over time, and this document deliberately does **not**
pre-select the QAT library. The tool is an output of parity gate P0, not an input
to it: P0 must prove one end-to-end QAT -> ONNX -> ORT C++ path on MobileNetV2 and
only then pin the exact compatible PyTorch/QAT-library/ONNX/ORT versions in the
lockfile, before long training begins.

P0 evaluates, in this order, and stops at the first that produces a QDQ graph ORT
executes as integer:

1. QDQ fake-quant modules inserted directly and exported with `torch.onnx.export` —
   fewest moving parts, and the export semantics are ours rather than a library's;
2. NVIDIA `pytorch-quantization` — TensorRT-oriented but emits QDQ that ORT accepts;
3. `torchao`, only if it demonstrably works here.

Ordering matters because the CNN -> ONNX QDQ export path is where these libraries
diverge most; several target LLM workloads and carry weak or deprecated ONNX
export for convolutional models. Record which candidates failed and why — a
rejected path is P0 evidence and belongs in the report's "what did not work"
section. Do not silently fall back to a different quantization meaning.

### 8.3 Structured pruning — M3

Use dependency-aware structured channel pruning. The existing homework
`hw1/src/structured.py` is a starting point, not drop-in code: update the input
shape, ignored layers, objective metric, residual/depthwise dependency handling,
and output classifier.

Core pruning roots are limited to the **expansion channels** inside MobileNetV2
inverted residual blocks. A removal must be one verified dependency group spanning
the expansion `1x1` output and BatchNorm, the corresponding depthwise-convolution
input/output channels and groups, and the projection `1x1` input channels. Keep
projection output widths, residual/add widths, the stem, final convolution, and
16-output classifier fixed. Expansion channels are not independently removable:
Torch-Pruning (or an equivalently tested dependency solver) must emit and validate
the complete coupled group before mutation. Do not broaden Core to residual-width
pruning unless the expansion-only search cannot create a valid candidate and the
broader dependency contract is separately reviewed.

Procedure:

1. Profile the unpruned model's MACs and parameter count.
2. Run sensitivity analysis using validation bobcat F2/recall, not generic
   accuracy.
3. Create candidates targeting approximately 15%, 30%, and 45% MAC reduction.
4. Round every surviving channel count to a multiple of **8** (`round_to=8` in the
   dependency solver). Record the requested and realized MAC reduction separately.
5. Physically remove channels; zero masks alone do not qualify as structured
   pruning.
6. After every pruning step, assert equality of depthwise `groups`, `in_channels`,
   and `out_channels`; residual-add shape equality; forward/backward execution;
   and ONNX exportability.
7. Fine-tune each candidate with the same data and loss contract.
8. Export each candidate and confirm that ONNX MACs and tensor shapes changed.
9. Select one M3 candidate on the validation Pareto frontier for QAT.

Step 4 protects the conclusion, not the accuracy. ORT/MLAS kernels on ARM
vectorize over 8- and 16-channel groups, so an unaligned width such as 403 is
processed as if it were 408 and the tail lanes are wasted. Pruning 576 -> 403 would
then show roughly 30% fewer MACs and almost no latency change, and the project
would report "structured pruning does not speed up MobileNetV2" while blaming
depthwise separable convolutions — when the real cause was the channel count. The
same trap is worse for M4, where INT8 kernels are more alignment-sensitive than
FP32. With rounding in place, a null pruning result is a measurement of the
architecture instead of an artifact of the solver. `hw1/src/structured.py:33`
already constructs the pruner, so this is one argument.

The report must distinguish parameter reduction, MAC reduction, file size, and
measured latency; they are not interchangeable.

### 8.4 Pruned + QAT — M4

Apply the validated QAT procedure to the selected pruned FP32 model. If M4 is
slower or less accurate than M2, M2 remains the final model. A more complicated
stack does not win by default.

### 8.5 Pre-Pi shortlist and final-model selection

Before renting the Pi, build a deployable shortlist using validation:

1. Reject candidates that fail export/parity/correctness gates.
2. Retain candidates satisfying the 90% sequence-balanced validation bobcat-recall
   operating rule; if none do, retain the best documented fallback candidates.
3. Remove candidates dominated on validation bobcat F2, MACs, and model size.
4. Use `gx10` latency only to detect pathologies such as float fallback; never use
   it to rank Cortex-A76 performance.
5. Prefer a simpler transformation when two candidates are otherwise equivalent,
   but package every non-dominated deployable candidate for Pi validation.

Write the shortlist and rejected alternatives to
`results/model_selection/pre_pi_shortlist.md`. On Pi Day 2, measure the shortlist
on the fixed validation benchmark. On Day 3, choose the final optimized model from
the validation-accuracy/Pi-latency/size Pareto evidence, write
`results/model_selection/final_decision.md`, freeze model/policy/runtime, and only
then allow final test evaluation.

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
./wildlife_trigger infer --model model.onnx --policy bobcat_v1.json --image x.jpg
./wildlife_trigger run-dataset --model model.onnx --policy bobcat_v1.json --manifest val.jsonl --output predictions.jsonl
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

### P0 — toolchain spike

Before full training:

1. Export an ImageNet-pretrained MobileNetV2 FP32 model at provisional opset 17;
   reject the legacy opset-9 path.
2. Produce a small PTQ model.
3. Produce a one-epoch QAT model.
4. On `gx10`, load all three with the exact planned ONNX Runtime C++ build in
   the target-compatible ARM64 environment.
5. Create sessions with `ORT_ENABLE_ALL`, call the C++
   `SessionOptions::EnableProfiling(prefix)` API, save the session-optimized graph,
   run one image, and capture the profile/operator coverage. Compare
   `ORT_ENABLE_EXTENDED` only as an explicitly named E6 performance candidate.
6. Verify FP32/PTQ/QAT graphs use the same accepted opset and that integer
   execution evidence does not depend on a single fused-node name.
7. Pin compatible versions only after this succeeds.

### P1 — preprocessing parity

For at least 20 fixtures covering landscape, portrait, grayscale-looking IR, and
odd dimensions:

- save the canonical Python input tensor;
- compare it with reference C++ preprocessing;
- compare it with fused C++ preprocessing;
- require matching shapes and documented maximum/mean absolute error.

This catches BGR/RGB, interpolation, padding, layout, and normalization bugs.

### P2 — FP32 model parity

PyTorch and ORT FP32 must match on a fixed validation fixture set:

- logits within numeric tolerance;
- identical top-1 class;
- identical bobcat fire/no-fire decisions except samples explicitly identified as
  lying within the numeric tolerance of the threshold.

### P3 — quantized-model validation

For PTQ/QAT, compare the correct framework reference with ORT and report numeric
drift. Exact FP32 equality is not expected, but the model must satisfy:

- valid quantized graph with intended operator coverage;
- validation metrics equal to the recorded quantized candidate;
- ORT Python and ORT C++ predictions/decisions identical on fixtures;
- no silent fallback to an unintended model or preprocessing path.

### P4 — C++ dataset parity

On a validation manifest, Python evaluation and the C++ dataset runner must emit
the same ordered image IDs, labels, target scores within tolerance, trigger
decisions, and confusion matrix.

No model reaching a failed gate may be deployed to the Pi.

### Mandatory early vertical slice

Before data preparation or full training, the target-compatible ARM64 environment
must run a thin end-to-end slice using a 16-output smoke model:

```text
saved JPEG -> C++ decode/preprocess -> ORT -> generic policy
           -> SHUTTER_TRIGGER JSON -> benchmark JSON + system metrics
```

The slice must also build/install from the provisional deployment bundle and emit
schema-valid results. Accuracy is irrelevant for the smoke model; interface,
correctness plumbing, observability, and packaging are the gate. This makes the
C++/benchmark path an early prerequisite rather than work deferred until after
model optimization.

---

## 11. C++ application design

Use C++17, CMake, OpenCV, and ONNX Runtime CPU Execution Provider. Pin the ORT
version; do not retain the course container's obsolete hard-coded 1.14.1 package.
Use a Release build and target-scoped `-O3`; do not set global
`CMAKE_CXX_FLAGS_RELEASE`. Never use `-march=native` while building on `gx10`,
because that selects the build host rather than the Pi. A Pi-specific build may
use a recorded explicit CPU target (`cortex-a76` for Pi 5 or `cortex-a72` for Pi
4), or `native` only when compilation runs on the same target Pi. Baseline and
optimized models must be benchmarked with the same binary and compiler flags.

### Required components

1. `ModelSession`
   - validates model input/output names, shapes, and types;
   - owns ORT environment/session through RAII;
   - supports configured intra-op thread count;
   - defaults to `ORT_ENABLE_ALL`, supports the registered E6 graph-level
     comparison, and enables profiling with an explicit file prefix;
   - can persist the session-optimized graph for operator/type inspection;
   - exposes model metadata and timing.
2. `Preprocessor`
   - correct reference implementation;
   - fused single-pass implementation;
   - reusable preallocated input buffer;
   - exact contract from section 5.5.
3. `Policy`
   - loads `mode: any` plus one or more class/threshold entries from JSON;
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

Run a bounded, validation-only inference-pipeline matrix:

1. reference versus fused preprocessing;
2. full JPEG decode versus OpenCV/libjpeg-turbo reduced decode at 1/2 and 1/4;
3. supported ORT graph-optimization levels;
4. intra-op threads `1, 2, 4`;
5. ORT memory arena on/off when exposed by the pinned build;
6. CPU affinity only when the remote Pi exposes a stable, documentable control.

Factor 2 is a first-class latency candidate rather than a footnote. On a 1024x747
`_sm` frame, `IMREAD_REDUCED_COLOR_4` decodes straight to 256x187 — the 256x192
network input minus the grey pad — so the aligned path skips the resize entirely
and does strictly less DCT work than a full decode. Decoding a 1024 px JPEG is
plausibly more expensive than MobileNetV2 inference itself on a Cortex-A76, which
would make this the largest single end-to-end win available; the profile decides,
not this paragraph. 1/8 is excluded because 1024/8 = 128 lands below the input and
would require an upscale.

Reference/fused preprocessing must pass P1. Reduced JPEG decode intentionally
changes pixels, so it is an accuracy/decision-drift candidate, not a parity-
equivalent implementation: compare it through P4 and keep it only if validation
bobcat metrics remain within the predeclared tolerance. Its appeal must not
shorten that check — an aligned decode is still a different image. Change one
factor at a time before testing a combined configuration. Report isolated and
combined effects; do not attribute total model speedup to preprocessing alone.

XNNPACK is not a Core requirement. ONNX Runtime CPU EP, bottleneck profiles,
decode reduction, preprocessing fusion, graph settings, threads, and memory
configuration satisfy the inference-level optimization scope without introducing
a second backend/toolchain.

### Performance target

For resident batch-1 inference on saved JPEGs, the primary engineering target is
Pi p95 end-to-end latency at or below **200 ms** (at least 5 end-to-end FPS). The
aspirational target is p95 at or below **100 ms** (about 10 FPS). Model load/cold
start is measured separately. These are project targets, not claims of measured
professional camera-trigger latency.

---

## 12. Raspberry Pi benchmark protocol (Pi 5 preferred)

### 12.1 Hardware scope

- rented Raspberry Pi 5, BCM2712, CPU-only, or documented RPi 4 contingency if
  Pi 5 cannot be provisioned;
- no physical camera/GPIO/power meter;
- batch size 1;
- exact OS, kernel, compiler, OpenCV, ORT, CPU governor, cooling exposure, and
  available sensors recorded in `results/pi/environment.json`.

### 12.2 Benchmark datasets

Prepare before renting the Pi:

- `benchmark_val_1000.jsonl`: fixed stratified validation subset for performance
  tuning, parity, and dry runs, including bobcat, empty, rare, multi-label, and
  preprocessing edge cases where available. **Pi parity on this manifest is
  mandatory** — it is the evidence that licenses evaluating full test accuracy on
  `gx10` instead of on the Pi, and it is the "target-hardware parity" required by
  Gate F. It must therefore be built to test that specific claim:
  - include a dedicated **threshold-adjacent stratum**: frames whose target score
    satisfies `|score - threshold| < eps`, over-sampled well beyond their natural
    frequency. A hardware difference can only change `SHUTTER_TRIGGER` on those
    frames, so a subset that omits them can pass while proving nothing;
  - include **M0-FP32** explicitly, not only the optimized winner. FP32 is the
    exposed case: float accumulation order depends on vector width, and GB10 uses
    SVE2 where the Pi uses NEON. INT8 QDQ convolutions accumulate exactly in int32,
    so i8mm and dotprod return bit-identical results — the intuition that the
    quantized models are riskier is inverted here;
  - record score deltas, not only decision agreement, so a drift that has not yet
    crossed a threshold is still visible;
- full cis-test and trans-test manifests for frozen final accuracy evaluation on
  `gx10` with the exact C++/ORT artifacts;
- an optional small post-freeze *test-split* parity subset for Pi, run only if
  transfer/storage permit. This one strengthens the argument but is not the gate.
  Do not copy the full approximately 6 GB test image set to Pi by default.

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

When measuring an inference-pipeline optimization, vary only that named factor
and run the required validation accuracy/decision-drift check. When comparing
models, use the frozen winning C++ pipeline unchanged.

Latency is not called energy. Without a power meter, the report may say that
lower active compute time is relevant to energy, but it may not claim measured
joules or battery life.

### 12.5 Five-day trial schedule

| Day | Allowed work |
|---|---|
| 1 | Provision, record environment, install pinned artifacts, smoke test only |
| 2 | Validation benchmark of M0 plus shortlist; decode/thread/ORT profiling; safe fixes only |
| 3 | Select final optimized model from Pi validation evidence; freeze everything; launch confirmation seeds asynchronously |
| 4 | Full frozen cis-test/trans-test C++ accuracy on gx10; Pi M0-vs-final performance benchmark and parity subset |
| 5 | Exact unchanged Pi benchmark/parity repeat, artifact backup, no tuning from test results |

If Core is not dry-run complete before Day 1, do not start the trial.

**The rental clock never waits on a GPU job.** Confirmation seeds 17/73 for the
selected transformation can only start once Day 3 names the winner, and retraining
a pruned+QAT candidate twice can take longer than a trial day. They measure
training variability, they do not produce the deployed artifact (seed 42 does),
and they therefore must not gate the Day 3 freeze, any later trial day, or Gate F.
Launch them on `gx10` in the background and report them whenever they land, even
after the trial expires. Because `gx10` remains dedicated to the project, both
confirmation runs must finish before Gate G and the final submission.

If the planned Pi 5 trial is lost — provisioning fails, the instance is withdrawn,
or the hardware proves unusable — immediately try another Pi 5 provider, then an
RPi 4 provider because the assignment explicitly permits either generation. If no
Raspberry Pi can be obtained, preserve the validation/MACs/size selection and all
other evidence, but classify the result as a degraded partial submission: Gate F
fails and Core is not complete because native RPi execution and target-hardware
benchmarks are mandatory. Never substitute `gx10` timings or describe this
contingency as meeting the project requirements.

---

## 13. Execution phases and gates

The implementing agent follows this order.

### Phase A — repository and environment

- create package/application structure from section 14;
- establish Python and C++ test runners;
- create pinned `gx10` training and target-compatible ARM64/C++ environments;
- create lockfiles and environment capture;
- run P0 toolchain spike;
- complete the mandatory 16-output C++ vertical slice, benchmark JSON, system
  metrics, and provisional bundle install.

**Gate A:** FP32, PTQ, and minimal QAT models load in ARM64 C++ ORT, and the thin
saved-JPEG-to-trigger benchmark path works end to end.

### Phase B — data

- download metadata and required images;
- build frozen manifests and empty supplement;
- build and fingerprint `cis-val-clean` while preserving official cis-val;
- implement audit assertions;
- create data-audit notebook and report.

**Gate B:** all section 5.3 assertions pass; manifests and hashes are committed.

### Phase C — FP32 baseline

- implement dataset/transforms/model/training/evaluation;
- resolve the 224x224 versus 256x192 input-shape control;
- run the empty-supplement data ablation;
- train seed 42 baseline;
- calibrate threshold;
- export and pass P1-P4;
- run confirmation seeds 17 and 73.

**Gate C:** reproducible M0 metrics, policy, ONNX, parity, and model card exist.

### Phase D — Core optimizations

- M1 PTQ;
- M2 QAT;
- M3 structured pruning candidates;
- M4 selected pruning + QAT;
- update a single machine-readable comparison table after each candidate.

**Gate D:** a validation/MACs/size-based deployable shortlist is written before
Pi rental; `gx10` latency has not been used to rank Pi candidates.

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
target-hardware parity, and three repetitions; frozen full-test C++ accuracy exists
from `gx10` using the exact model/policy/runtime artifacts.

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
    third_party/
      nlohmann/json.hpp               # pinned single-header runtime policy parser
      nlohmann/LICENSE.MIT
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
      bobcat_v1.json
      bobcat_coyote_v1.json
      threshold_catalog.json
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
   - primary bobcat policy JSON, final-model threshold catalog JSON, and one
     validated multi-target example policy JSON;
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
2. Official-vs-clean cis-val leakage table, empty-supplement ablation, and the
   supplement shortcut-probe score.
3. 224x224-vs-256x192 input-shape control with real-pixel utilization/MACs.
4. Training curves for M0 and final optimized model.
5. Validation shortlist and final Pi-selection table.
6. Cis/trans target metrics with sequence-bootstrap confidence intervals and
   threshold uncertainty.
7. Per-class support/confusion matrices for M0 and final, with multi-label
   exclusions stated.
8. Accuracy vs model size/MACs table.
9. Pi table: model, precision, file size, threads, p50/p95/p99 inference,
   end-to-end FPS, peak RSS, CPU utilization, temperature/throttling.
10. Pareto plot: trans bobcat recall or F2 vs Pi p95 latency.
11. Full/reduced JPEG decode and reference/fused preprocessing latency/accuracy.
12. At least six representative failure cases with scores and explanations.

All plots include units, sample counts, split, model ID, and commit/run ID.

---

## 18. Risk register and decision rules

| Risk | Decision rule |
|---|---|
| Official train/cis-val sequence leakage biases selection | Preserve official split, use fingerprinted `cis-val-clean` for decisions, report both |
| Empty supplement introduces background/domain bias | Keep it ID/sequence/location-disjoint, deterministic, identical across models, and run the matched no-empty ablation |
| Multi-label annotations make softmax metrics ambiguous | Store all labels, exclude seven multi-label train images from CE, include presence in target metrics, exclude them from single-label confusion/macro F1 |
| Trans bobcat recall is poor | Report honestly; do not train on trans-test. Stretch KD is allowed only after Core completion |
| 224-square letterbox wastes useful pixels | Resolve the pre-registered 224x224-vs-256x192 control before M0 |
| PTQ loses accuracy on depthwise MobileNetV2 | This is pre-registered; use QAT/quantization debugging and keep PTQ as a negative result if necessary |
| QAT export/runtime path is unstable | P0 before training; pin compatible versions; fail early rather than improvise during Pi trial |
| Legacy opset 9 blocks quantization or changes export semantics | Start P0 at opset 17, require one accepted opset across M0-M4, and reject legacy spike artifacts |
| Empty supplement arrives at 2x the resolution of CCT-20 | Per-image CCT downloads are original-resolution; downsize to max 1024 px per side before training, record filter/quality, and run the shortcut probe. A materially above-chance probe blocks training |
| Empty supplement is location-disjoint by necessity | Unavoidable: every `empty` frame at the 10 cis locations is already spent in cis-val/cis-test. Close the removable resolution confound, report the remaining background confound, and measure cis/trans empty false-fire separately |
| Structured pruning does not speed MobileNetV2 | Round surviving widths to multiples of 8 first, so the null result is about the architecture and not about unaligned SIMD lanes; then show real MAC reduction and measured lack of speedup. Final model may be unpruned QAT |
| The chosen QAT library cannot export deployable QDQ ONNX for MobileNetV2 | No library is pre-committed; P0 walks the ranked candidate list in §8.2, stops at the first that ORT executes as integer, and records every rejection as evidence |
| Pruning breaks depthwise groups or residual shapes | Restrict Core roots to coupled expansion-channel groups; assert group/residual shapes and export after every step |
| `gx10` latency misranks Cortex-A76 candidates | Use it only for pathology detection; shortlist by validation/MACs/size and select on Pi validation latency |
| Planned Pi 5 trial is lost | Try another Pi 5 provider, then RPi 4; if no Pi is available, Gate F fails and Core remains incomplete. Preserve a clearly labelled partial submission and never substitute `gx10` timings |
| Confirmation seeds cannot finish inside the trial window | They never gate the freeze or Gate F; seed 42 is the deployed artifact; run seeds 17/73 asynchronously on `gx10`, but require completion before Gate G |
| Pi parity subset disagrees with the frozen `gx10` reference | Stop and diagnose before claiming target equivalence. Report score/decision mismatch rates and treat Pi decisions as authoritative for affected frames. If unexplained, run full test accuracy on Pi; otherwise report the gx10 C++ accuracy only as gx10 evidence and explicitly withhold the Pi-equivalence claim |
| `badger` has one validation positive and `deer`/`fox` have none | Keep all three in the 14-entry catalog with null thresholds and explicit unavailable statuses; the policy loader refuses them as targets |
| C++ preprocessing silently differs | P1 golden tensor fixtures block deployment |
| ORT C++ differs from Python | P3/P4 block deployment |
| Reduced JPEG decode changes decisions | Treat it as an accuracy candidate, not parity; keep only after P4 validation |
| Remote Pi hides sensors/governor | Record unavailable values; use exposed `/proc`/`sysfs`; do not fabricate resource/energy claims |
| Five-day trial is spent debugging | Exact ARM64 dry run and one-command benchmark are prerequisites |
| Full test transfer consumes about 6 GB and rental time | Run frozen full accuracy on gx10; Pi receives the fixed benchmark/parity subset by default |
| ARM64 binary/ABI differs between gx10 and Pi | Prefer one proven official ORT AArch64 artifact in both target-compatible environments; otherwise build from pinned source on Pi |
| `-march=native` targets gx10 instead of Pi | Use target-scoped Release `-O3`; only use explicit Pi CPU flags or `native` when compiling on the same Pi |
| Scope creep | Only Core until Gate G; only crop-teacher KD afterward |
| Public repo cannot store large models | GitHub Release/LFS plus hashes and download script |

---

## 19. Core Definition of Done

Core is complete only when every item is true:

### Data and ML

- [ ] Data manifests, hashes, distributions, and leakage assertions pass.
- [ ] Official cis-val is preserved; fingerprinted `cis-val-clean` drives all
      development decisions.
- [ ] Multi-label train/evaluation rules and counts are tested.
- [ ] Empty supplement is ID-, sequence-, and location-disjoint from CCT-20 and
      reproducible.
- [ ] Empty supplement is downsized to max 1024 px per side to match the CCT-20
      `_sm` archive, and the supplement-versus-CCT-20 shortcut probe scores at or
      near chance.
- [ ] Every M3/M4 surviving channel count is a multiple of 8, with requested and
      realized MAC reduction recorded separately.
- [ ] Empty-supplement and input-shape controls are completed before M0 freeze,
      with the empty ablation matched on optimizer steps rather than epochs.
- [ ] M0, M1, M2, M3, and M4 results exist or a technically justified failed
      candidate is preserved and documented.
- [ ] All deployed M0-M4 ONNX artifacts use the single opset accepted by P0
      (provisionally 17); no opset-9 artifact enters the pipeline.
- [ ] Thresholds use validation only.
- [ ] The threshold catalog has status entries for all 14 animals, numeric
      thresholds for exactly 11 selectable targets, and null thresholds for
      unavailable `badger`, `deer`, and `fox`.
- [ ] Final optimized model is selected on Pi validation evidence before test
      evaluation; `gx10` latency did not rank candidates.
- [ ] Confirmation seeds 17/73 exist for M0 and the selected final transformation.
      Selected-transformation seeds do not gate the Pi freeze or Gate F, but must
      finish before Gate G and final submission.
- [ ] Cis/trans metrics and confidence intervals exist.

### Deployment and C++

- [ ] Baseline and final ONNX models pass preprocessing/model/C++ parity.
- [ ] C++ CLI, dataset runner, benchmark harness, and self-tests pass.
- [ ] Runtime policy/catalog JSON parsing uses the pinned vendored header and
      requires no system YAML/JSON development package.
- [ ] Release builds use recorded target-safe flags, and ELF/glibc compatibility
      checks or an on-Pi source build prove target loadability.
- [ ] The same final model passes bobcat-only and multi-target policy tests without
      another model inference per frame.
- [ ] ARM64 dry run succeeds from a clean environment.
- [ ] Pi baseline and optimized runs use the same application and protocol.
- [ ] Full frozen cis-test/trans-test C++ evaluation runs on gx10; Pi parity subset
      decisions match the frozen reference for both M0-FP32 and the selected winner,
      including the threshold-adjacent stratum, with score deltas recorded.
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

The crop teacher trains on the 15 non-empty categories (14 animals plus `car`),
because empty frames have no box. If the Core student retains 16 outputs, KD is
applied only on non-empty samples and compares the teacher distribution with the
student's non-empty logits after excluding and renormalizing the `empty` dimension.

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
