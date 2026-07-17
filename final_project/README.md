# Wildlife Trigger — Efficient ML Final Project

Status: **Phases A–E complete — Gate E PASSED. The conditional Pi trial (Phase F) is
next.** The full optimization ladder (M0 FP32 / M1 INT8 PTQ / M2 INT8 QAT / M3
structured-pruned FP32 / M4 pruned+QAT) is trained, gated (P3/P4), carded, and compared
(Gate D); the deployable pre-Pi shortlist **M0 · M2 · M4** and the fixed
`benchmark_val_1000.jsonl` are frozen. **Phase E (C++ application + deployment
bundle) is COMPLETE — Gate E PASSED.** The C++ foundation is hardened and exercised
against the real M0 baseline (Gate E1), the preprocessor/session/policy/dataset-runner/
benchmark are each certified against DESIGN §11, the pre-rental QEMU `cortex-a76` ISA
parity is bit-identical, the optimization matrix + correctness consolidation (Gate E6)
are done, and the deployment bundle installs and benchmarks end to end in a clean
ARM64 container (Gate E). The matrix found decode the only latency knob with headroom,
but the reduced-decode drift gate **rejected** it (it loses real bobcat detections), so
shipping stays full decode. The dry-run benchmark (diagnostic, gx10) shows the
quantization payoff: M0 12.36 ms → M2 6.69 ms → M4 5.45 ms. See the phase table below.
**Next: the conditional Phase F Pi trial**, then Phase G (analysis, report, release).

**Trans-domain bobcat recall is poor and is reported as such**, per DESIGN §18's registered
decision rule. M0 catches 86% of bobcat visits at cameras it has seen and 18% at the
unseen one. No threshold reaches DESIGN §6.3's 90% recall floor inside the device's 5%
false-fire budget, so C3 recorded `recall_floor_infeasible` and ships the best operating
point the budget allows. That is a measurement, not a failure — but nothing in this
repository may describe it as the primary rule being met. C5's confirmation seeds
showed it is a property of the frozen recipe, not of seed 42: trans F2 is
0.1142 ± 0.0175 across seeds 17/42/73, and no epoch of any seed crossed 0.135
(issue #18).

| | |
|---|---|
| Gate A (toolchain, target, export/quantization contract) | **passes**, 45 checks |
| Gate B (data acquisition, splits, audit) | **passes**, 43/43 |
| C0/C1/C1a (fixtures, training engine, input decision) | **done** |
| C2 (train M0 seed 42) | **done** — bobcat F2 0.6272 cis / 0.1054 trans |
| C3 (calibrate the operating point) | **done** — `recall_floor_infeasible`: ships threshold 0.5381 inside the fire budget, but the 90% recall floor is out of reach (trans 7.9%); **not a pass** |
| C4 (export and parity) | **done** — ONNX `c3102764…` at opset 17; P1 bit-exact, P2 and ORT py↔cpp passed; policy re-bound to the ONNX after proof |
| C5 (seeds 17/73, model card) | **done** — trans F2 0.1142±0.0175 across three seeds: the gap is the recipe, not the seed (#18); model card `artifacts/model_cards/m0_fp32.md`; comparison table opened with the M0 row |
| Phase D (M1-M4 optimization ladder) | **Gate D PASSES** — see the ladder table below; all five candidates past P3/P4; shortlist and benchmark frozen |
| E1 (C++ project foundation) | **done** — Gate E1 PASSES: foundation hardened (logging convention, `schema_version`) and exercised against real M0; build + 4/4 ctest + self-test/infer(native+QEMU)/benchmark/run-dataset green in the target container (`results/e1/e1_gate.json`) |
| E2 (preprocessing) | **done** — `Preprocessor` fused + reference paths agree ≤1e-6 on six geometries and match the Python golden tensors (P1: python↔fused 0.0, python↔reference 7e-7); BGR-as-RGB rejected; corrupt image raises |
| E3 (model session + policy) | **done** — `ModelSession` (RAII, contract validation, `ORT_ENABLE_ALL`, profiling, optimized-graph) + `Policy` (`mode: any`, model/class-map hash binding, `SHUTTER_TRIGGER` output); full policy/threshold test matrix incl. `empty`-target rejection, green in the target container |
| E4 (dataset runner) | **done** — P4 dataset parity for M0 over cis_val_clean (3214) + trans_val (1725): confusion matrix identical, 0 hard decision disagreements; the FP32 score gap (≤1.1e-2) is the P1 OpenCV 4.6↔4.13 `INTER_LINEAR` drift — diagnosed, reported, not a bug (`results/e4/p4_dataset_parity_m0.json`) |
| E5 (benchmark + system monitor) | **done** — percentile calculation unit-tested (numpy-matching linear interpolation); benchmark emits a `performance_targets` report (200 ms/5 FPS, 100 ms/10 FPS) with `measured_on_pi:false`; system monitor honest on absent sensors (`results/e5/benchmark_m0.json`) |
| E6 (correctness + optimization experiment) | **done — Gate E6 PASSED** (`results/e6/e6_gate.json`). QEMU `cortex-a76` ISA parity bit-identical for M0/M2/M4 (`qemu_parity.json`). Optimization matrix (one factor at a time on M0, diagnostic): decode the only latency knob with headroom (half 1.10×, quarter 1.17×), `threads=4` regresses (0.88×), preprocess/graph/arena within noise (`optimization_matrix.json`). Reduced-decode drift gate **REJECTS** half/quarter — they lose real bobcat detections (M0 17–18, M2 10–12, M4 17–19) + add 1.1–3.2% false fires, so shipping stays full decode (`decode_drift.json`). Native-vs-target: gcc 13/glibc 2.39 and gcc 12/glibc 2.36 both 5/5 ctest + self-test, bit-identical decisions over benchmark_val_1000 (`native_vs_target.json`). P1–P4 consolidated + sha-bound for the shortlist; ALL/EXTENDED graphs retained (differ by sha) |
| E7 (deployment bundle) | **done** — `build_bundle.sh` stages M0/M2/M4 + policies + class map + a 47-frame sample slice + the pinned ORT + `preflight.sh`/`install.sh`/`run_demo.sh`/`run_benchmark.sh` + `BUNDLE.json` (git commit + per-artifact sha) + `MANIFEST.sha256`. OpenCV apt-installed by `install.sh` (Debian's imgcodecs GDAL closure is impractical to carry; Pi OS Bookworm has the matching `.406`). Clean-install test in a fresh `debian:bookworm-slim` passed: max GLIBC 2.34 ≤ 2.36, all libs resolve, self-test + demo run (`results/e7/e7_bundle.json`). **Fail-closed F1 host preflight (issue #77):** refuses non-aarch64 / non-Bookworm / Cortex-A72 (no `asimddp`) before any mutation and writes a machine-readable `environment.json`; success + all refusal paths proven without a physical Pi (`results/e7/preflight.json`) |
| E8 (full ARM64 dry run) | **done — Gate E PASSED** (`results/e8/dry_run.json`). The exact Pi commands (`install.sh` + `run_benchmark.sh` + `run_demo.sh`) run unattended in a clean container (exit 0); the one-command benchmark matrix includes the M0 baseline; outputs machine-readable and `measured_on_pi:false`. Diagnostic latency (gx10, not a Pi result): M0 12.36 ms → M2 6.69 ms → M4 5.45 ms |
| F (Pi trial), G (report/release) | not started — F is conditional, one-shot |

### The optimization ladder (Phase D, validation / deployment ORT)

| model | kind | primary (mean bobcat F2@0.5) | MACs | bytes | on shortlist |
|---|---|---:|---:|---:|:--:|
| M0 | FP32 baseline | 0.3663 | 293.4M | 8,950,645 | ✅ (baseline) |
| M1 | INT8 PTQ (percentile) | 0.3527 | 293.4M | 2,620,130 | — (dominated) |
| M2 | INT8 QAT (lr5e-5) | **0.3832** | 293.4M | 2,536,267 | ✅ |
| M3 | structured-pruned FP32 (c30) | 0.3583 | **205.6M** | 7,035,950 | — (dominated) |
| M4 | pruned + QAT | 0.3730 | **205.6M** | **2,014,806** | ✅ |

Every operating point is `recall_floor_infeasible` (the trans-domain recall floor
is out of reach for the whole ladder, as for M0 — a measured property of the
recipe, never described as a pass). The non-dominated deployment front is **{M2
(accuracy), M4 (MACs + size)}** with M0 as the FP32 baseline; the final model is
chosen by **Pi latency** (Phase F), never gx10 timing. Per-candidate detail:
`artifacts/model_cards/m{1,2,3,4}_*.md`; the machine-readable table is
`results/model_selection/comparison.jsonl`; the shortlist and frozen bundle are
`results/model_selection/pre_pi_{shortlist.md,freeze.json}`.

The Core input is **frozen at 256x192** and the head contract at **16 outputs with a
5,000-frame empty supplement** — decided by C1a on measured data, not assumed:
`results/ablations/data_input_decision.md`. `PLAN.md` is the task tracker and is the
authority on what is done; this table is a summary of it.

Public repository: `REPO_URL`

Final release: `RELEASE_URL`

Final report: `REPORT_URL`

These placeholders must be replaced before submission.

---

## What this project does

Wildlife Trigger is a CPU-only Edge AI application for a Raspberry Pi 5. It
receives a wildlife-camera frame, runs a full-frame classifier in C++, and emits
an emulated shutter signal when any animal selected in the target policy is
present. Bobcat is the primary target for training evaluation and course results.

```text
saved JPEG frame
    -> C++ preprocessing
    -> MobileNetV2 / ONNX Runtime
    -> any configured animal score >= its calibrated threshold?
    -> SHUTTER_TRIGGER=1 or 0
```

Example final behavior:

```text
frame_000412.jpg  predicted=bobcat  score=0.94  SHUTTER_TRIGGER=1
frame_000413.jpg  predicted=coyote  score=0.03  SHUTTER_TRIGGER=0
frame_000414.jpg  predicted=empty   score=0.01  SHUTTER_TRIGGER=0
```

The course submission uses saved images and emulates the shutter signal. A future
physical product could place the same decision between a motion sensor and a
camera/GPIO shutter interface.

The dataset label is **bobcat (`Lynx rufus`)**, not Eurasian lynx (`Lynx lynx`).

---

## Course objective and evidence

The assignment requires a neural network to run natively on a Raspberry Pi, a
baseline-versus-optimized comparison, on-device latency/FPS/resource metrics,
custom model/inference optimizations, and a C++ inference implementation.

This project supplies:

| Rubric area | Project evidence |
|---|---|
| Model/optimization strategy | FP32 MobileNetV2 baseline, INT8 PTQ, INT8 QAT, structured pruning, pruned QAT |
| C++ inference | Correct/fused preprocessing, ONNX Runtime, target policy, signal emulation, dataset runner, tests |
| Benchmarking | Same Pi 5, C++ binary, images, order, and protocol for baseline and optimized models |
| Analysis/presentation | Raw evidence, reproducible notebooks, report, slides, negative results, bottlenecks, next steps |

The mandatory course outputs are the codebase and formal slide deck. This project
also produces a formal report, analysis notebooks, deployable models, a Raspberry
Pi bundle, and raw evidence so the work can be reproduced and defended.

---

## Core scope

Core is intentionally limited to one model and a lightweight configurable target
policy:

- full-frame MobileNetV2, width 1.0; **256x192 input, frozen by C1a** after the matched
  224x224 control it required (the arms tied on the metric, and 256x192 carries +31.1%
  real pixels at -2.0% MACs);
- 16 outputs: 14 CCT-20 animal classes plus `car` and `empty`;
- ImageNet-pretrained transfer learning;
- generic `mode: any` target list with per-class thresholds;
- calibrated bobcat policy as the primary graded configuration;
- one inference per frame;
- ONNX Runtime CPU Execution Provider;
- C++17/OpenCV/ONNX Runtime application;
- rented Raspberry Pi target, with Pi 5 preferred and RPi 4 accepted only as the
  documented contingency; CPU-only final measurements.

Core candidates:

| ID | Model |
|---|---|
| M0 | FP32 baseline |
| M1 | INT8 PTQ |
| M2 | INT8 QAT |
| M3 | Structured-pruned FP32 |
| M4 | Structured-pruned + QAT |

No candidate is assumed to win. Validation accuracy/MACs/size creates a pre-Pi
shortlist; real Pi validation latency selects the final optimized model before
test evaluation.

P0 starts with ONNX opset 17 and accepts one common opset for M0-M4. PTQ uses S8S8
QDQ as the primary representation. Structured pruning is restricted to verified
MobileNetV2 expansion-channel dependency groups; residual/projection output widths
stay fixed. C++ Release builds use target-scoped `-O3`, never `-march=native` when
compiled on `gx10`.

Gate/cascade, object detection, physical GPIO, power measurement, separate
per-species networks/model packs, multi-label simultaneous-species recognition,
illegal-logging detection, custom inference engines, NPU/Hailo, and battery-life
claims are outside Core.

The only optional Stretch is crop-teacher knowledge distillation, and it remains
locked until the complete Core submission passes its Definition of Done.

### Configurable target policy

The model always computes 14 animal scores plus `car` and `empty`. Selecting one
or several catalog-supported animals changes only a JSON policy and adds no model
inference:

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

The checked-in values above are schema examples, not final thresholds. The
pipeline replaces them with validation-calibrated values bound to the final model.
The bundle contains a bobcat-only policy, a per-class threshold catalog, and one
validated multi-target example. Unknown/duplicate classes, `car` or `empty` as a
wildlife target, classes with no calibrated threshold, invalid thresholds,
unsupported modes, and hash mismatches are rejected.

The C++ bundle vendors a pinned `nlohmann/json` single header and license; policy
parsing happens once at startup and requires no system YAML/JSON development
package.

Selecting an existing CCT-20 animal requires no retraining — but only where a
threshold can be calibrated. CCT-20 validation has **zero `deer` and zero `fox`
positives**, while `badger` has only one positive image / one sequence. The catalog
contains status entries for all 14 animals, but numeric thresholds for exactly
**11 selectable targets**; `badger`, `deer`, and `fox` have null thresholds and
are refused by the policy loader. Adding a species outside the 16-class map
requires labelled data and fine-tuning. Detecting two different species
simultaneously in one frame would require a multi-label model or detector and is
outside Core.

---

## Data

Primary benchmark: Caltech Camera Traps-20 (CCT-20).

| Split | Images | Purpose |
|---|---:|---|
| train | 13,553 | Training |
| cis-val | 3,484 | Validation at known locations |
| cis-test | 15,827 | Final test at known locations |
| trans-val | 1,725 | Validation at one unseen location |
| trans-test | 23,275 | Final test at nine unseen locations |

The official train and cis-val share 224 sequences affecting 270 cis-val images,
including 10 bobcat images. The project preserves official cis-val but creates
`cis-val-clean` with 3,214 images / 144 bobcat images for every development
decision. All other relevant sequence intersections are zero.

The metadata also contains a few distinct-class multi-label images. Their complete
label sets are preserved; seven multi-label train images are excluded from
single-label CE, while validation/test target metrics count bobcat as present
whenever it occurs in the label set. Single-label confusion/macro metrics state
their multi-label exclusions.

The current downloadable split files contain 57,864 images in total. Test labels
remain sealed until model, threshold, C++ binary, runtime options, preprocessing,
and thread configuration are frozen.

The official CCT-20 train split currently contains no `empty` training frames.
Core therefore adds a deterministic supplement of 5,000 full-CCT empty images,
selected only from locations disjoint from all 20 CCT-20 locations. Only those
images are downloaded; the full 105 GB CCT archive is not required.

CCT-20 ships downsized to a maximum of 1024 px per side, while per-image CCT
downloads are served at original resolution. The supplement is therefore downsized
to match before training, and a shortcut probe confirms the two pools are not
separable — otherwise resolution alone would predict `empty`. Total data
acquisition is about 8.1 GB: 6 GB for the CCT-20 `_sm` archive, ~2.1 GB for the
supplement, and ~12 MB of metadata.

Before M0 is frozen, matched ablations test the 5k empty supplement and compare
224x224 square letterbox with 256x192 landscape letterbox. These are controlled
data/input decisions, not extra deployment models.

Datasets and large image archives are never committed to the public repository.
Versioned manifests, source URLs, hashes, and audit outputs are committed.

Authoritative data sources:

- https://lila.science/datasets/caltech-camera-traps
- https://beerys.github.io/CaltechCameraTraps/
- https://openaccess.thecvf.com/content_ECCV_2018/papers/Beery_Recognition_in_Terra_ECCV_2018_paper.pdf

---

## Document hierarchy — start here

An implementation agent must read documents in this order:

1. **This README** — orientation, scope, and entry points.
2. [`DESIGN.md`](DESIGN.md) — source of truth for technical decisions,
   acceptance criteria, metrics, deliverables, and Definition of Done.
3. [`PLAN.md`](PLAN.md) — atomic execution tasks, dependencies, outputs, and
   gates.
4. The newest `../Handoff/HANDOFF_*.md` — actual session state, artifacts,
   failures, and next task.
5. [`Final Project TASK.docx`](Final%20Project%20TASK.docx) — original assignment
   and rubric.

If the documents disagree, stop and fix the inconsistency. Do not infer a new
architecture from old files or conversation history.

### Source-of-truth responsibilities

| Document | Responsibility |
|---|---|
| README | Project entry point and reproduction/navigation guide |
| DESIGN | What must be built and why; contracts and acceptance criteria |
| PLAN | In what order to build it; task state, dependencies, and gates |
| Newest handoff | What has actually happened in the latest session |
| TASK.docx | Course requirements and grading rubric |

---

## Execution overview

The implementation follows seven gated Core phases:

1. **A — Repository and toolchain:** pin environments and require an early saved-
   JPEG -> C++ -> ORT -> policy -> benchmark vertical slice before full training.
2. **B — Data:** freeze official/clean splits, multi-label rules, and the location-
   disjoint empty supplement; pass every count/leakage/hash audit.
3. **C — FP32 baseline:** resolve empty/input controls, train M0, calibrate bobcat,
   define generic policy, export ONNX, and pass parity gates P1-P4.
4. **D — Optimization:** build M1-M4 independently and freeze a deployable
   validation/MACs/size shortlist, not a gx10-latency winner.
5. **E — C++ and deployment:** implement/test the application, benchmark harness,
   system monitor, deployment bundle, and clean ARM64 dry run.
6. **F — Raspberry Pi:** select the optimized winner on Pi validation latency,
   freeze, run full C++ test accuracy on gx10, and benchmark/parity-check on Pi.
7. **G — Submission:** generate analysis, report, slides, public release, model
   artifacts, deployment bundle, and submission manifest.

No Pi rental begins before the complete ARM64 dry-run gate. No Stretch begins
before Phase G passes.

---

## Target command interface

The following commands describe the required final interface. They are not
expected to work until their corresponding PLAN tasks are implemented.

### Python pipeline

```bash
python -m wildlife_trigger.data.prepare --config configs/data/cct20.yaml
python -m wildlife_trigger.data.audit --config configs/data/cct20.yaml
python -m wildlife_trigger.train --config configs/train/m0_fp32.yaml
python -m wildlife_trigger.optimize.ptq --config configs/optimize/m1_ptq.yaml
python -m wildlife_trigger.optimize.qat --config configs/optimize/m2_qat.yaml
python -m wildlife_trigger.optimize.prune --config configs/optimize/m3_prune.yaml
python -m wildlife_trigger.export --run results/training/c2/RUN_ID --policy artifacts/policies/bobcat_v1.json
python -m wildlife_trigger.validate.parity --run-id RUN_ID
python -m wildlife_trigger.evaluate --run-id RUN_ID --split val
python -m wildlife_trigger.calibrate --run results/training/c2/RUN_ID --target bobcat
```

### C++ application

```bash
./wildlife_trigger infer \
  --model artifacts/models/final.onnx \
  --policy artifacts/policies/bobcat_v1.json \
  --image demo/bobcat.jpg

./wildlife_trigger run-dataset \
  --model artifacts/models/final.onnx \
  --policy artifacts/policies/bobcat_v1.json \
  --manifest data/manifests/benchmark_val_1000.jsonl \
  --output results/predictions.jsonl

./wildlife_trigger benchmark \
  --model artifacts/models/final.onnx \
  --manifest data/manifests/benchmark_val_1000.jsonl \
  --threads 4 --warmup 100 --iterations 1000

./wildlife_trigger self-test --fixtures tests/fixtures/
```

### Intended one-command workflows

```bash
bash scripts/run_core_pipeline.sh
bash scripts/build_cpp.sh
bash scripts/package_pi.sh
bash scripts/run_pi_benchmarks.sh
bash scripts/generate_submission.sh
```

Every stage must also remain runnable separately for debugging and evidence.

---

## Compute environments

### Dedicated `gx10` — primary project environment

`gx10` is fully allocated to this final project until completion. It is the single
development and compute host for Phases A-E and G: repository work, data download
and preparation, GPU training, PTQ/QAT/pruning, export, Python evaluation, C++
development, CPU-only ONNX Runtime inference, shutter emulation, all pre-Pi tests,
profiling, notebooks, report/slides, and deployment packaging.

The known platform is NVIDIA GB10, ARM64, CUDA 13.0; the pipeline must capture
and publish the exact observed environment rather than rely on this README. Long
runs must be checkpointed, logged, and safely resumable. GPU acceleration is for
training; the final C++ inference path is still ONNX Runtime CPU EP.

Pi compatibility is tested on `gx10` in a clean target-compatible ARM64 container.
ARM64 alone does not guarantee matching Raspberry Pi OS/glibc dependencies. The
pipeline pins a matching container base by digest and audits `ldd`/required
`GLIBC_*` symbols. If a portable binary cannot be proven, the bundle includes
pinned source/build scripts and compiles the C++ executable while provisioning the
Pi.

Do not hard-code private hostnames, usernames, SSH keys, tokens, or paths into the
public repository. Private deployment values belong in ignored local config or
environment variables.

### Raspberry Pi target (Pi 5 preferred)

The Pi is the only valid source of target latency/FPS/resource evidence. Its trial
is measurement time, not normal
development time. It receives a frozen deployment archive only after the clean
`gx10` ARM64 dry run; then target smoke/parity tests are repeated before final
benchmarks. `gx10` performance numbers must never be presented as Pi results. The
`gx10` host continues to orchestrate remote commands and store copied raw evidence.
The five-day schedule and fair-comparison rules are defined in DESIGN §12 and
PLAN §8.

If the planned Pi 5 becomes unavailable, the fallback order is another Pi 5
provider and then an RPi 4 provider, which the assignment also permits. Without
any Raspberry Pi, Gate F fails and the work is only a partial submission: `gx10`
results must never be relabelled as target-hardware evidence.

---

## Correctness gates

No model reaches the Pi until all applicable gates pass:

1. P0: FP32/PTQ/QAT toolchain and ARM64 C++ execution.
2. P1: Python vs C++ preprocessing golden tensors.
3. P2: PyTorch vs ORT FP32 logits and decisions.
4. P3: quantized graph/operator/runtime validation.
5. P4: Python vs C++ validation dataset scores, decisions, and metrics.

This is especially important because OpenCV decodes BGR, while the pretrained
model expects RGB/ImageNet normalization.

---

## Final evaluation

Full frozen C++/ORT accuracy runs on `gx10` after the Pi validation freeze and is
reported separately for cis-test and trans-test:

- bobcat recall — primary;
- bobcat precision and F2;
- false-fire rate and fire rate;
- frame and sequence-balanced target recall;
- event capture rate and recall by positive-sequence length strata;
- per-class support and support-aware macro F1;
- confusion matrix;
- sequence-bootstrap metric and threshold intervals;
- per-location bobcat recall on trans-test.

The Pi repeats parity on a fixed validation subset and provides the authoritative
performance measurements; transferring the full approximately 6 GB test image set
to the rented Pi is optional.

System results compare the same C++ application and protocol:

- model load time;
- decode/preprocess/inference/policy/end-to-end p50/p95/p99;
- inference FPS and end-to-end FPS;
- model/file size and MACs;
- peak RSS and CPU utilization;
- exposed frequency, temperature, and throttling status;
- three process-level repetitions.

Primary performance target: Pi p95 end-to-end latency <= 200 ms (>=5 FPS).
Aspirational target: p95 <= 100 ms (about 10 FPS). Full/reduced JPEG decode,
reference/fused preprocessing, ORT graph settings, threads, and memory settings are
measured as bounded inference-level candidates.

Latency is not presented as measured energy. There is no physical power meter.

---

## Required submission package

The final public release contains:

1. Public GitHub repository, clean tagged commit, license, citation file.
2. Python training/conversion/evaluation and C++ inference source.
3. Exact configs, lockfiles, tests, automation, and reproduction instructions.
4. FP32 baseline and final optimized ONNX models, policies, model cards, hashes.
5. Raspberry Pi deployment archive with installer, executable/runtime, models,
   bobcat/multi-target policies, threshold catalog, class map, sample data, demo
   command, and checksums.
6. Raw training/evaluation/parity/Pi evidence and generated figures/tables.
7. `notebooks/01_data_audit.ipynb` and `02_results_analysis.ipynb`, cleanly
   executable from frozen artifacts.
8. `report/final_report.md` and visually verified `final_report.pdf`.
9. `slides/final_presentation.pptx` and visually verified PDF.
10. `SUBMISSION.md` containing canonical repository/release/report links, final
    commit, model hashes, headline metrics, and reproduction commands.

The public repository URL must appear in README, report, first slide, final slide,
and `SUBMISSION.md`. Large images/checkpoints are published through documented
external data sources, GitHub Releases, or Git LFS rather than ordinary Git.

Notebooks support inspection and analysis; they are not the only executable form
of the project. Training, export, evaluation, C++ build, and benchmarks must be
scriptable from a clean environment.

---

## Expected repository layout

The final structure is specified in DESIGN §14. The major directories are:

```text
configs/       resolved data/train/optimization/runtime contracts
src/           Python package
cpp/           C++17 ONNX Runtime application and tests
scripts/       setup, pipeline, build, package, Pi, submission automation
tests/         Python tests and golden fixtures
notebooks/     data audit and results analysis
data/          small committed manifests/docs, never the image archives
artifacts/     policies, model cards, hashes, release links
deploy/        Raspberry Pi bundle sources
results/       raw evidence, comparisons, Pi logs, generated figures
report/        final report source/PDF
slides/        final presentation PPTX/PDF
demo/          sample command/output and optional short recording
```

---

## Reuse and legacy assets

| Existing asset | Core use |
|---|---|
| `Docker_VSCode/` | Legacy ARM64/C++/ONNX toolchain reference only; its smoke code is not the final application |
| `hw1/src/structured.py` | Starting point for dependency-aware structured pruning; must be adapted and tested |
| `hw2`/`hw3` QAT code | Training-loop ideas only; Core requires a new deployable affine INT8 path proven in P0 |
| `hw3/src/distill.py` | Post-Core crop-teacher KD Stretch only |
| `hw4/` NAS/supernet | Search discipline, MBConv/width-space insight, and proxy-rank caveats only; no Core supernet because gx10 latency cannot rank Pi and NAS would add a second selection problem |

Do not copy legacy preprocessing, input sizes, class counts, quantizers, or metrics
without adapting them to DESIGN and proving parity. In particular, no opset-9
artifact or host-native compiler flag may enter Core.

---

## Current result placeholders

No measurements exist yet. Do not replace `TBD` with estimates.

| Result | FP32 baseline | Final optimized |
|---|---:|---:|
| cis-test bobcat recall | TBD | TBD |
| trans-test bobcat recall | TBD | TBD |
| p95 inference latency on Pi 5 | TBD | TBD |
| end-to-end FPS on Pi 5 | TBD | TBD |
| peak RSS | TBD | TBD |
| model size | TBD | TBD |

Every final value must link back to a machine-readable result file and frozen
run/commit ID.

---

## Core completion and Stretch

Core completion is defined by the checklist in DESIGN §19 and Gate G in PLAN.

Only after it passes may the project run the optional crop-teacher KD experiment:

- crop teacher on the 15 non-empty classes (14 animals + `car`);
- crop-augmentation control;
- cross-view KD under the same student budget;
- KD counts as successful only if it beats the crop-augmentation control.

The frozen Core result remains the primary submission even if Stretch is added.

---

## References

- Original assignment: [`Final Project TASK.docx`](Final%20Project%20TASK.docx)
- Design/source of truth: [`DESIGN.md`](DESIGN.md)
- Execution plan: [`PLAN.md`](PLAN.md)
- MobileNetV2: https://arxiv.org/abs/1801.04381
- CCT-20 paper: https://openaccess.thecvf.com/content_ECCV_2018/papers/Beery_Recognition_in_Terra_ECCV_2018_paper.pdf
- CCT/LILA: https://lila.science/datasets/caltech-camera-traps
- ONNX Runtime quantization: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
