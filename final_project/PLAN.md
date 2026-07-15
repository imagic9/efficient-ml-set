# Final Project — Autonomous Core Execution Plan

Status: **Core approved; implementation not started.**

This file converts [`DESIGN.md`](DESIGN.md) into executable work. It is the task
tracker for an implementation agent; `DESIGN.md` remains authoritative for every
technical decision, metric, acceptance threshold, and deliverable contract.

If PLAN and DESIGN ever disagree, stop and resolve the documents. Do not choose
one silently.

---

## 1. Document hierarchy and agent protocol

Read in this order:

1. [`README.md`](README.md) — project orientation, scope, and entry points.
2. [`DESIGN.md`](DESIGN.md) — complete technical and submission specification.
3. This file — task order, dependencies, outputs, and gates.
4. The newest `Handoff/HANDOFF_*.md` — current session state and deviations.

This file is the only task tracker. The next task is the first `[ ]` in phase
order; each completed task carries a dated note with what it established. A
second status board was tried and removed: it duplicated these checkboxes and
would have drifted from them.

The executing agent must follow these rules:

- use the dedicated `gx10` host for Phases A-E and G, including all data work,
  training, C++ work, shutter emulation, tests, dry runs, and deliverable builds;
- use the rented Raspberry Pi target (Pi 5 preferred, RPi 4 contingency) only for
  Phase F target-hardware verification and final measurements; never present
  `gx10` timings as Pi results;
- keep bobcat as the primary graded target while implementing generic target-set
  configuration over the 14 animal outputs; `car` and `empty` are not targets;
- work only on Core until Gate G passes;
- the only permitted post-Core Stretch is crop-teacher KD;
- mark a task complete only when its listed artifact exists and its checks pass;
- preserve commands, configs, logs, raw predictions, and raw timings;
- never reconstruct slide/report numbers manually;
- never evaluate models or make decisions from cis-test/trans-test labels before
  the freeze task; mechanical manifest/schema audits are allowed;
- use validation data for model/runtime/thread decisions;
- stop at a failed gate and fix the cause before continuing;
- keep all paths and commands non-interactive and rerunnable;
- update this plan and the newest handoff at the end of each work session.

### Completion states

- `[ ]` not started;
- `[~]` in progress;
- `[x]` completed and verified;
- `[!]` blocked, with the reason and evidence recorded in the newest handoff.

Do not use `[x]` for a partial implementation.

---

## 2. Fixed scope and critical path

Core is one full-frame MobileNetV2 with 16 outputs and a generic configurable
target policy, exported to ONNX and executed by a C++ ONNX Runtime application on
a Raspberry Pi 5. Bobcat remains the primary calibrated and graded target; target
selection never loads or runs another neural network.

Optimization candidates:

- M0 — FP32 baseline;
- M1 — INT8 PTQ;
- M2 — INT8 QAT;
- M3 — structured-pruned FP32;
- M4 — structured-pruned + QAT.

Dependency overview:

```mermaid
flowchart TD
  A0 --> A1 --> A2 --> A3
  A3 --> A4 --> GA[Gate A]
  A4 --> E1 --> E2 --> E3 --> E4 --> E5
  GA --> B0 --> B1 --> B2 --> B3 --> B4
  B4 --> C0
  B4 --> C1
  C0 --> C1A[C1a controls]
  C1 --> C1A --> C2 --> C3 --> C4 --> C5
  C0 -.-> E2
  C4 -.-> E3
  C5 --> D1
  C5 --> D2
  C5 --> D3 --> D4
  D2 --> D5
  D4 --> D5
  D1 --> D6
  D2 --> D6
  D4 --> D6
  D5 --> D6
  D6 --> E6
  E5 --> E6 --> E7 --> E8
  E8 --> F1 --> F2 --> F3 --> F4 --> F5
  F5 --> G1 --> G2 --> G3 --> G4 --> G5 --> GG[Gate G]
```

A4 implements the minimal interfaces later hardened by E1-E5 against trained M0
and the optimized shortlist. No Pi bundle is frozen before D6 produces the
complete deployable shortlist.

The dotted edges are the ones easiest to miss when reading the solid chain as the
critical path: E1 may indeed start right after A4, but E2 cannot be finished before
C0 freezes the golden fixtures and E3 cannot be finished before C4 exports a real
model. The E chain therefore overlaps phases B-D rather than preceding them.

---

## 3. Phase A — repository and toolchain

### A0 — Record starting state

- [x] Confirm access to the dedicated `gx10` working copy and capture
      `git status`, current branch/commit, disk space, CPU/GPU, ARM64 architecture,
      OS, CUDA, compiler, and available persistent-job mechanism.
- [x] Preserve unrelated user changes.
- [x] Create a dated run/session log.

**Output:** `results/provenance/project_start.json` and newest handoff update.

**Done 2026-07-15.** Captured by `scripts/capture_provenance.py`; run log at
`results/provenance/RUNS.md`. Key facts that shape later tasks:

- gx10 is Ubuntu 24.04 / **glibc 2.39**, but Raspberry Pi OS Bookworm ships glibc
  2.36. A natively-built binary would request `GLIBC_2.38/2.39` symbols the target
  cannot resolve, so A2's container must be `debian:bookworm-slim`. A binary built
  against 2.36 still loads on a newer Pi OS; the reverse is false.
- gx10 CPU is Cortex-X925 + A725 with `i8mm`, `sve`, `sve2`. Pi 5 Cortex-A76 has
  none of these. This is the measured basis for the DESIGN §12.2 parity strata.
- 20 cores, GB10 / CUDA 13.0 / torch 2.11.0+cu130, 502.8 GiB free, 117.8 GiB RAM
  available after the boreal stack was stopped (see DESIGN §4 operational note).
- `torch-pruning 1.6.1` is already present. `onnx`, `onnxruntime` and `opencv` are
  not — A2 installs them.
- gx10 commits and pushes directly over SSH using a dedicated repo-scoped deploy
  key (`~/.ssh/id_ed25519_efficientml`, host alias `github-efficientml`,
  `IdentitiesOnly yes`). Write access verified. The `~/.netrc` HTTPS token is
  read-only and is no longer used for this repository. Commit as
  `Vadym <imagic9@gmail.com>` from either machine.

### A1 — Create repository skeleton

Depends on: A0.

- [x] Create the package, configs, C++ directories, scripts, tests, notebooks,
      data-manifest directories, artifact directories, results, report, slides,
      demo, and deployment directories specified by DESIGN §14.
- [x] Add `.gitignore` rules for datasets, caches, credentials, build outputs,
      large checkpoints, and temporary benchmark files.
- [x] Add placeholder `SUBMISSION.md`, `CITATION.cff`, license, and artifact/data
      READMEs.
- [x] Establish Python and C++ test commands.

**Done when:** a clean checkout has an understandable structure and empty test
suites execute successfully.

**Done 2026-07-15.** Test commands, both verified on `gx10`:

```bash
python -m pytest tests/python                 # 28 passed
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
ctest --test-dir build --output-on-failure    # 1/1 passed
```

Notes for later phases:

- The issue #3 `.gitignore` fix is now proved on a live tree, not on hypothetical
  paths: `src/wildlife_trigger/data/`, `configs/data/` and `data/manifests/` all
  stage correctly.
- `cpp/` ships a real `cpu_features` library, not a stub, and it is the first
  thing wired through the emulation harness. Verified: the **same binary** reports
  `asimd,asimddp` under `qemu-aarch64 -cpu cortex-a76` and
  `asimd,asimddp,sve,sve2,i8mm,bf16` natively. That divergence is exactly the
  mechanism by which ORT will select different kernels, so the E6 rehearsal rests
  on a harness that is already known to work.
- CMake **rejects** `-mcpu=native` unless `WILDLIFE_ALLOW_NATIVE=ON`; verified it
  fires. Building with `-DWILDLIFE_CPU_TARGET=cortex-a76` works.
- No C++ test framework and no vendored `nlohmann/json` yet. Both are deferred to
  E1, when there is behaviour to assert and a caller that reads JSON;
  `cpp/third_party/README.md` records the no-network vendoring contract.
- `LICENSE` is MIT with Vadym as copyright holder.

### A2 — Reproducible environments

Depends on: A1.

- [x] Define the isolated `gx10` Python/GPU training environment.
- [x] Define the `gx10` CPU-only C++/ONNX Runtime development environment.
- [x] Define a clean target-compatible ARM64 container on `gx10` for Pi build,
      bundle-install, and full dry-run checks.
- [x] Record the target distro/glibc/compiler contract, pin the matching container
      base by digest, and add `ldd` plus required-`GLIBC_*` symbol checks. If exact
      compatibility cannot be proved, make the pinned on-Pi source build the
      deployment path. A0 measured gx10 at glibc **2.39** against Pi OS Bookworm's
      **2.36**, so the base is `debian:bookworm-slim`: a binary linked against 2.36
      loads on a newer Pi OS, and the reverse does not.
- [x] Install and pin `qemu-user` in that container for ISA-level checks. A0
      measured `-cpu cortex-a76` advertising exactly the Pi 5 feature set
      (`asimd`+`asimddp`, no `sve`/`sve2`/`i8mm`/`bf16`) and `-cpu cortex-a72` the
      Pi 4 one (no `asimddp`). ORT dispatches kernels from these bits at runtime, so
      emulation reproduces the Pi's kernel choice and numerics. It reproduces
      **nothing** about latency — no emulated timing enters a results table.
- [x] Pin compiler, CMake, OpenCV, ONNX, ONNX Runtime, and Python dependencies.
- [x] Add environment-capture tooling and resolved run-config serialization.
- [x] Add checkpoint/resume and persistent logging for every long-running job.
- [x] Verify no secret, SSH key, token, or dataset credential is committed.

**Outputs:** lockfile(s), environment setup scripts, and environment JSON schema.

**Done 2026-07-15.** All pins live in `configs/env/pins.env`, the
single source of truth both the Dockerfile and the setup scripts read.

*ORT is not a compatibility risk.* The official aarch64 tarball needs at most
`GLIBC_2.27` / `GLIBCXX_3.4.21` and links only standard system libraries — far
below bookworm's 2.36. One identical ORT binary therefore serves the gx10
container and the Pi, which is the cheapest de-risking available. Re-verify on any
bump: it is a property of their build, not a promise.

*Version alignment is a parity concern, not hygiene.* The first resolve produced
Python ORT 1.27.0 against C++ 1.27.1, and Python OpenCV **5.0.0** against C++
4.6.0. P3 compares ORT Python with ORT C++, and P1 compares the two preprocessing
implementations — under a version split both gates would still pass or fail, but
about the distance between two upstream releases rather than between our own two
call sites. ORT is now 1.27.0 on both sides (PyPI publishes no 1.27.1 wheel;
matching beats newer). OpenCV Python is pinned to the newest 4.x.

*One gap is left open on purpose.* C++ OpenCV stays at bookworm's 4.6.0 because
that is what the Pi can have, against Python's 4.13. Only the `INTER_LINEAR`
resize could differ — decode, BGR→RGB, pad, scale and normalise are trivially
defined. **P1 must quantify it**; if drift exceeds tolerance, the answer is to
build a matching OpenCV in the container and bundle those `.so` to the Pi. Measure
before deciding.

*The training venv is isolated on purpose.* Not `~/efficientml/venv`, which sets
`include-system-site-packages=true` and inherits ~5 GB from `~/.local`; a lockfile
from there would describe an environment we neither control nor can reproduce.
`setup_gx10.sh` asserts the isolation before installing and fails if CUDA is
missing rather than silently training on CPU.

**Verified state.**

| | Python (`~/venvs/wildlife_trigger`) | C++ (`wildlife-trigger-target:bookworm`) |
|---|---|---|
| ONNX Runtime | 1.27.0 | 1.27.0 |
| OpenCV | 4.13.0 | 4.6.0 (bookworm apt) |
| torch | 2.11.0+cu130, CUDA on GB10 cap 12.1 | — |
| glibc | 2.39 (gx10, never shipped) | **2.36**, matching Pi OS Bookworm |
| compiler | — | gcc 12.2, cmake 3.25.1 |

Commands:

```bash
scripts/setup_gx10.sh              # isolated venv + requirements.lock (49 pins)
scripts/build_target_container.sh  # bookworm image, ORT verified at build time
scripts/verify_target_env.sh       # the A2 gate, unattended
python -m pytest tests/python      # 40 passed
```

`verify_target_env.sh` proves, end to end: our binary needs GLIBC ≤ 2.34 and
libonnxruntime ≤ 2.27 against a 2.36 target; ORT links, reports 1.27.0 and
constructs a session **under `qemu -cpu cortex-a76`**. So the ISA rehearsal works
with real ORT, not only with the probe.

Secrets audit: no key material or credential-shaped string anywhere in history,
not merely at HEAD. A dotfile `.env` is now ignored — deliberately `.env` and not
`*.env`, since `configs/env/pins.env` must be committed — and the token-pattern
check is a test rather than a one-off.

**Left for E7:** OpenCV soname. Bookworm apt gives 4.6.0; a Trixie Pi would give
another. Either bundle the `.so` or link statically — do not assume the Pi's apt
matches.

### A3 — P0 toolchain spike

Depends on: A2.

- [x] Export ImageNet-pretrained MobileNetV2 FP32 to ONNX at provisional opset 17;
      explicitly reject the legacy opset-9 spike artifact.
- [x] Create a small static PTQ ONNX model.
- [x] Choose the QAT library here rather than assuming one. Try, in DESIGN §8.2
      order, direct QDQ fake-quant + `torch.onnx.export`, then NVIDIA
      `pytorch-quantization`, then `torchao`; stop at the first that yields a QDQ
      graph ORT executes as integer. Record every rejected candidate and its
      failure — that is P0 evidence and belongs in the report.
- [x] Run one epoch/minimal step of the chosen QAT path and export deployable
      INT8 ONNX.
- [x] On `gx10`, load all three models with the exact planned C++ ORT build
      inside the target-compatible ARM64 environment.
- [x] With the C++ API, start from `ORT_ENABLE_ALL`, call
      `SessionOptions::EnableProfiling(prefix)`, save the session-optimized graph,
      and run a fixture. Save profiles plus operator/data-type coverage; do not use
      one fused-node name as the sole proof of INT8 execution.
- [x] Verify FP32/PTQ/QAT use the same P0-accepted opset. Compare
      `ORT_ENABLE_EXTENDED` later only as an explicitly named E6 candidate.
- [x] Re-run all three models under `qemu-aarch64 -cpu cortex-a76` in the same
      container. Confirm integer execution survives without `i8mm`/`sve2` and record
      the operator/data-type coverage ORT picks instead. A QAT path that only works
      because gx10 has `i8mm` is a **P0 failure**; this is the cheapest place to
      find that out.
- [x] Pin versions only after FP32/PTQ/QAT all work end to end, natively and under
      `-cpu cortex-a76`.

**Output:** P0 evidence that all three model forms execute in ARM64 C++ and the
QAT artifact is genuinely quantized.

**Done 2026-07-15.** `scripts/run_p0_spike.sh` runs the whole gate unattended; all
16 checks pass (`results/p0/p0_gate.json`). One command reproduces every claim
below.

*The QAT library question is answered, and the answer is that it was the wrong
question.* Candidate 1 (DESIGN §8.2's first) works, so `pytorch-quantization` and
`torchao` were never installed. But the first attempt at candidate 1 **failed** in
exactly the way §8.2 warns about — a float graph carrying rounded weights: 45
FusedConv + 5 Conv still running float, only 2 of 52 convolutions quantized. The
cause was not the library. Fake-quant on the convolution *input* is what every
TensorRT-oriented library emits, and ORT's float-level ConvActivationFusion reaches
`Conv + Clip -> FusedConv` before the QDQ rule can match `DQ -> Conv -> Q`.
Candidates 2 and 3 place QDQ the same way and would have reproduced the failure
with more dependencies. **The axis that mattered was QDQ placement against ORT's
fusion rules, not the library.** This belongs in the report's "what did not work".

What P0 established, all measured, none assumed:

| Fact | Evidence |
|---|---|
| M1 PTQ and M2 QAT both execute as **integer** in C++ ORT, natively and under `-cpu cortex-a76` | `p0_gate.json`, 16/16 |
| The QAT and PTQ **optimized graphs are identical** — `QLinearConv:52, QLinearAdd:10, QGemm:1, QLinearGlobalAveragePool:1` | `*.coverage.json` |
| Integer execution **survives the loss of `i8mm`/`sve2`**; the emulated CPU reports `asimd,asimddp` and `looks_like_pi5=true` | `*.cpp-qemu.probe.json` |
| M0 FP32 does **not** report integer execution (the negative control) | `fp32_stays_float` |
| C++ and Python ORT are both **1.27.0** and agree on argmax over a shared fixture blob | `*.probe.json` |
| All three forms carry **opset 17** *after* PTQ/QAT rewrote the graph | `opset_parity.json` |

Three findings that cost real time and would have cost more later:

- **ORT requires rank-0 QDQ scales.** torch exports per-tensor scale/zero-point as
  shape `[1]`. ORT refuses the graph outright — and `onnx.checker.check_model` with
  `full_check=True` passes it. `optimize/qdq_scalar.py` repairs the rank and only
  the rank. A structural check would have shipped this artifact.
- **ReLU6 must be absorbed into the activation quantizer**, not left between Conv
  and Q. This is exact rather than convenient (a quantizer over `[0, m<=6]` already
  clamps as ReLU6 does), and `verify_relu6_removal_is_exact` measures it at 0.0
  difference rather than arguing it. ORT's own PTQ quantizer does the same removal.
- **The classifier's Gemm needs its flattened input quantized.** `torch.flatten`
  sits between the pooled vector's quantizer and the Gemm, so nothing else can
  reach that tensor; without it every convolution ran integer around a float
  classifier.

Two bugs found in this task's *own* checks, worth recording because both passed
while proving nothing:

- the Python/C++ agreement check compared the C++ blob against the C++ probe's own
  report — C++ against itself — while Python was separately generating a different
  input. Both call sites now read one shared fixture blob, and the gate refuses to
  compare unless they did.
- `ort_coverage` exited 1 both when a model was not integer and when ORT could not
  load it, leaving a **stale report** on disk that read as a plausible result. Exit
  codes are now 0/2/1 (integer / not-integer / could-not-run) and the report is
  written even on failure.

**Reproducibility:** `nn.Dropout` in the classifier draws from torch's *global* RNG,
so the QAT export differed on every run (argmax 21, then 908) despite seeded data
generators. Seeded in `optimize/qat.py`; two consecutive exports now produce
byte-identical SHA-256.

**Left for later:** ORT warns that a session-optimized graph serialized above
`ORT_ENABLE_EXTENDED` "should only be used in the same environment the model was
optimized in" — so the optimized graphs here are evidence of what ORT *chose*, and
**must never be shipped to the Pi as artifacts**. E6/E7 ship the ordinary model and
let the Pi optimize it. PTQ/QAT accuracy is meaningless here by construction: both
used synthetic data, because no CCT download is permitted before Gate A.

### A4 — Mandatory early C++ vertical slice

Depends on: A3. This task implements the minimal smoke subset of E1-E5.

- [x] Export a deterministic 16-output smoke model and class map.
- [x] Run saved JPEG -> C++ decode/preprocess -> ORT -> generic policy ->
      `SHUTTER_TRIGGER` JSON.
- [x] Produce schema-valid per-stage benchmark JSON and system-monitor output.
- [x] Build and install a provisional ARM64 deployment bundle in the clean
      target-compatible environment.
- [x] Preserve the exact command, output, ORT profile, and bundle checksum.

**Done 2026-07-15.** `scripts/run_a4_slice.sh` runs the whole slice unattended; all
29 checks pass (`results/a4/a4_gate.json`).

*The slice earned its keep on its first run.* It failed immediately, with the model
contract check refusing to infer: the smoke model had been exported at ImageNet's
224x224 while the application defaults to DESIGN §5.5's provisional Core input of
256x192. That mismatch would otherwise have surfaced during C1a — after training —
as a model that silently could not be deployed. Both now read
`INPUT_SHAPE_PROVISIONAL_CORE` from one place. This is the entire argument for
building a vertical slice before the data exists, and it paid on day one.

What A4 established:

| Fact | Evidence |
|---|---|
| Full path works: JPEG -> C++ decode/preprocess -> ORT -> policy -> `SHUTTER_TRIGGER` JSON | `evidence/infer.native.json` |
| The path runs under **`-cpu cortex-a76`** and agrees with the native run's decision | `evidence/infer.qemu.json` |
| The policy loader **refuses all 12 invalid policies through the real CLI**, not only in ctest | `evidence/policy_rejections.json` |
| A multi-target policy works on the same model with **no reload** (DESIGN §4) | `evidence/infer.multi_target.json` |
| Benchmark JSON is schema-valid; p50<=p95<=p99 holds for every stage | `evidence/benchmark.native.json` |
| Absent sensors report **`"unavailable"`**, never 0 | same |
| Bundle: 7 files, checksums verify, **max GLIBC 2.34 <= the Pi's 2.36** | `evidence/bundle_audit.json` |
| The staged bundle runs its own self-test from its own launcher | `evidence/bundle_self_test.json` |

Judgement calls:

- **`nlohmann/json` 3.12.0 vendored now**, not in E1. `third_party/README.md` said to
  wait for a caller that reads JSON; A4's policy loader is that caller. GitHub
  publishes no digest for the asset, so the hash was cross-checked by fetching the
  same header from the tagged repo tree and confirming byte-identity — weaker than a
  signed digest, and the strongest thing upstream offers. `scripts/verify_vendored.sh`
  re-checks it as a gate.
- **SHA-256 is implemented in-tree** (`cpp/src/hashing.cpp`) rather than linking
  OpenSSL or vendoring a second dependency for ~60 lines. The only need is binding a
  policy to a model by hash — no secrets, no adversary. Tested against the NIST
  vectors including the multi-block padding case.
- **The session-optimized graph is deliberately not bundled**, per P0's finding that
  ORT considers it valid only in the environment that produced it.
- **OpenCV is not bundled** — a known E7 gap, since bookworm gives 4.6.0 and a Trixie
  Pi would give another soname.
- The fixture JPEG is **synthetic** and says so; a bright shape sits near the frame
  edge so a centre-cropping preprocessor would visibly lose it.

**Gate A:** P0 passes and the thin C++ inference/benchmark/deployment path works
end to end before data preparation or long training.

**PASSED 2026-07-15**, 45 checks across both gates (`results/gate_a.json`):

```bash
python -m wildlife_trigger.validate.gate_a \
    --p0 results/p0/p0_gate.json --a4 results/a4/a4_gate.json
# PASS P0 (16 checks) · PASS A4 (29 checks) -> GATE A PASSED
```

Phase B (CCT-20 download) and Phase C (long training) are now permitted. Neither was
started before this passed.

---

## 4. Phase B — data

### B0 — Acquire and fingerprint sources

Depends on: Gate A.

- [x] Download `eccv_18_annotations.tar.gz` (3 MB) for the official CCT-20 splits.
- [x] Download `eccv_18_all_images_sm.tar.gz` (6 GB), capped at 1024 px per side.
- [x] Download `caltech_camera_traps.json.zip` (9 MB) for empty-supplement
      selection. Do not download `cct_images.tar.gz` (105 GB) or the bounding boxes
      (35 MB, Stretch KD only).
- [x] Record URLs, timestamps, file sizes, and SHA-256 hashes.
- [x] **Record the observed image-dimension distribution of every split** and
      confirm the dominant frame against DESIGN §5.5. The input-shape argument and
      the reduced-decode alignment both rest on these numbers; neither may be
      inherited from the paper or from DESIGN.
- [x] Verify licensing/citation text for README/report/model card.

Budget roughly 8.1 GB of downloads and about 40 GB of working disk on `gx10`
(archive plus extraction plus artifacts).

**Outputs:** `data/README.md`, source manifest, checksums, and the split dimension
report.

**Done 2026-07-15.** Downloaded 6.50 GB total from LILA, now served from Google
Cloud Storage (`storage.googleapis.com/public-datasets-lila/...`) — the URL is
recorded per run because LILA has rehosted before. Hashes in
`data/raw/source_manifest.json`; the 105 GB archive and the bboxes were not fetched.
Licence: Community Data License Agreement (permissive variant).

**DESIGN §5.5 is now measured rather than inherited.** The annotation JSON records
the *original* geometry (2048x1494) and could never have answered this — the `_sm`
archive is capped at 1024 px, so the frames we actually decode are different files.
`data/dimensions.py` reads all 57,864 JPEG headers:

| Claim | DESIGN said | Measured |
|---|---|---|
| dominant decoded frame | 1024x747 | **1024x747** |
| its share of the corpus | ~91% | **91.4%** |
| `_sm` long-side cap | 1024 | **1024** (max observed) |
| unreadable frames | — | **0 of 57,864** |

So the 256x192 input choice and the `1024 / 4 = 256` reduced-decode alignment both
rest on the real corpus. Per-split dominance varies (trans-val is 100% 1024x747,
cis-test 89.7%), which is why the aspect-ratio argument is made against the corpus
rather than one split.

### B1 — Build official split manifests

Depends on: B0.

- [x] Parse train, cis-val, cis-test, trans-val, and trans-test JSON.
- [x] Freeze the exact 16-class order in `configs/data/classes.yaml`.
- [x] Assert the class set is 14 animals plus `car` and `empty`; mark only the 14
      animal classes as selectable policy targets.
- [x] Emit deterministic JSONL manifests with complete `labels`, optional
      `primary_label`, location, sequence, dimensions, and source metadata.
- [x] Reconcile counts to 13,553 / 3,484 / 15,827 / 1,725 / 23,275.
- [x] Fingerprint the official train/cis-val overlap as exactly 224 sequences,
      270 cis-val images, and 10 bobcat images.
- [x] Generate immutable `cis_val_clean.jsonl` with 3,214 images / 144 bobcat
      images by removing every train-overlapping `seq_id`.

**Outputs:** five versioned manifests plus category/location summaries.

**Done 2026-07-15.** Every number DESIGN pinned reconciles **exactly** against the
downloaded JSONs — no adjustment, no rounding:

| | Expected | Observed |
|---|---|---|
| split counts | 13,553 / 3,484 / 15,827 / 1,725 / 23,275 | all match (57,864 total) |
| train↔cis-val overlap | 224 seqs · 270 imgs · 10 bobcat | **exact** |
| `cis_val_clean` | 3,214 imgs · 144 bobcat | **exact** |
| trans-val bobcat (§4 table) | 793 | 793 |
| multi-class train images | "the seven" (B3) | 7 |

That the leakage fingerprint lands on all three numbers is the real result: it means
LILA has not republished the metadata, and DESIGN's §5.3 analysis was done against
this exact data.

**The class order is frozen: ascending CCT category ID**, which puts `bobcat` at
index 3. Derived from the dataset rather than chosen, so it is traceable and
deterministic; `car` (12) and `empty` (11) are marked non-selectable per DESIGN §4.
The IDs are sparse (1, 3, 5, ... 99), so "the 16 classes" was not an order until
this froze one — and every calibrated threshold binds to an index, so changing it
later would silently rebind thresholds to different animals.

The first generated `classes.yaml` was **malformed** (a flow mapping missing a comma
after the padded name), which parsed as nonsense rather than failing. The writer now
parses back what it wrote and asserts a round-trip. A generated config that does not
load is a bug that surfaces far from its cause.

**Note for C:** the A4 smoke `class_map.json` used a placeholder alphabetical order
and is now superseded — the real class map must be generated from `classes.yaml`.
A4's artifacts were explicitly marked provisional for exactly this reason.

### B2 — Build `cct_empty_train_v1`

Depends on: B1.

- [x] Extract all 20 CCT-20 location IDs.
- [x] Select exactly 5,000 full-CCT `empty` images from locations disjoint from
      all 20, stratified across locations/sequences with seed 42.
- [x] Download selected images only (~2.1 GB, served at original resolution).
- [x] **Downsize every image to max 1024 px per side** per DESIGN §5.2 step 7,
      matching the `_sm` archive. Record the resampling filter and JPEG quality in
      the data config. Without this the supplement arrives at ~2048 px while every
      CCT-20 split is 1024 px, making resolution a shortcut feature perfectly
      correlated with `empty` — and one that fails silently, because val/test
      contain only `_sm` frames.
- [x] Compute original and downsized checksums; emit the supplement manifest with
      both dimension sets.
- [x] **Run the supplement-versus-CCT-20 shortcut probe.** Train a small binary
      classifier to separate the two pools. Near chance means the confound is
      closed; a materially higher score blocks training and is recorded.
- [x] Confirm no selected image ID, sequence, or location leaks into CCT-20.

**Output:** frozen `cct_empty_train_v1.jsonl`, checksums, and the shortcut-probe
result.

**Done 2026-07-15.** 5,000 empty frames from 106 locations / 3,044 sequences, seed 42,
max single-camera share **4.4%**. All fetched, 0 failures, 4,955 downsized (45 were
already within the cap and are re-encoded anyway — uniform treatment, so no frame gets
one JPEG generation while its neighbour gets two). 1.16 GB on disk.

**Shortcut probe: 0.5775 held-out accuracy against a chance of 0.50** — below the 0.60
attention threshold, so the resolution/encoding confound is closed as far as the probe
can tell. The residual ~8% is consistent with the *unavoidable* location-disjoint
background difference (rule 3), which the probe cannot distinguish from encoding by
construction. The report says so rather than claiming the supplement is pristine.

**A silent bug that would have voided rule 3 entirely.** The two metadata files disagree
about a type: full CCT stores `location` as a string (`"26"`), CCT-20 as an integer
(`38`). So `image["location"] in cct20_locations` is `"26" in {26, 38}` — **False for
every image**. Rule 3 would have been disabled and the supplement drawn from the very
cameras it must avoid, with nothing crashing. Measured: the raw comparison finds 0
overlapping images, the normalised one finds 32,255. Fixed by `normalise_location`, and
the selector now **refuses to proceed if rule 3 rejects zero candidates** — because zero
is not a clean dataset, it is a broken comparison. It now rejects 3,917.

**Azure is 18x slower than GCP for per-image fetches**, and the retry loop was hiding it.
Measured from gx10 over 48 concurrent images: Azure 40 img/min with 4 of 48 failing, GCP
740 img/min with none, GCP at 48 workers 2,144 img/min. The full 5,000 would have taken
**over two hours** on Azure and took ~4 minutes on GCP. LILA mirrors the same bytes on
GCP, AWS and Azure; all three are recorded in `LILA_IMAGE_MIRRORS`. Re-measure rather
than inherit this.

### B3 — Implement data and preprocessing code

Depends on: B1, B2.

- [ ] Implement dataset readers and manifest validation.
- [ ] Exclude the seven distinct-class multi-label train images from CE while
      retaining full label sets for target-presence evaluation.
- [ ] Implement canonical aspect-preserving resize/pad/RGB/NCHW/ImageNet
      normalization from DESIGN §5.5.
- [ ] Implement training-only photometric augmentation without animal-removing
      crops.
- [ ] Make validation/test preprocessing deterministic.
- [ ] Build the offline preprocessing cache (DESIGN §5.5): steps 1-4 computed once
      into per-shape uint8 letterbox arrays, so training does not re-decode 57,864
      JPEGs every epoch of every run. Sound only because the augmentation list has
      no random crop/resize; the cache builder must call the *same* preprocessing
      code path the C++ application uses, must not re-encode to JPEG, and must be
      keyed by a hash of the preprocessing config plus source manifest and verify it
      on load. A cache that outlives its config trains on stale pixels and nothing
      downstream can detect it.
- [ ] Add unit tests for manifests, labels, missing/corrupt files, and transforms.
- [ ] Test that the cache builder and the on-the-fly path produce **identical**
      tensors, and that a changed preprocessing config invalidates the cache rather
      than being ignored.

**Output:** tested Python data package, resolved data config, and a
config-fingerprinted preprocessing cache.

### B4 — Data audit gate

Depends on: B3.

- [x] Implement every assertion in DESIGN §5.3.
- [x] Produce class, location, sequence, split, and supplement statistics.
- [x] Verify multi-label counts 7 / 0 / 1 / 61 / 9 across the five official
      splits and test target-presence semantics.
- [x] Emit the per-class validation support table (images and sequences on
      cis-val-clean and trans-val) and assert that the animal classes with zero
      validation positives are exactly `deer` and `fox`, while `badger` has
      exactly one positive image / one sequence. Record all three as unavailable
      targets with null thresholds.
- [ ] Render/inspect representative RGB, IR-like, empty, bobcat, small, portrait,
      and landscape samples.
- [ ] Complete and execute `notebooks/01_data_audit.ipynb` from a clean kernel.
- [x] Store machine-readable audit output and figures. *(audit output done;
      figures belong with the notebook item above)*

**Gate B:** every DESIGN §5.3 count, known-overlap fingerprint, clean-split,
category, multi-label, ID/sequence/location, path, and hash assertion passes. No
model training begins before Gate B.

**PASSED 2026-07-15**, 43/43 (`results/data_audit/gate_b.json`):

```bash
python -m wildlife_trigger.data.audit --manifests-dir data/manifests ...
# GATE B PASSED — 43 checks, 0 failed
```

Every §5.3 assertion holds on the real download: split counts, unique IDs, no image in
two splits, the known 224/270/10 overlap, `cis_val_clean` at 3,214/144, all four
must-be-empty sequence intersections, train locations disjoint from both trans splits,
the supplement disjoint on ID/sequence/location and capped at 1024, multi-label counts
7/0/1/61/9, and the shortcut probe near chance.

The validation support table reproduces DESIGN §4 **exactly** — bobcat 937 img / 315 seq
(= 144 cis-val-clean + 793 trans-val), badger exactly 1/1, deer and fox exactly 0. So
the catalog really does contain 11 selectable targets and three that no threshold can
be defended for. That was a design claim; it is now a measurement.

The gate prints the recorded source hashes on any failure, because DESIGN §5.3's rule is
that these counts fingerprint a specific upstream download: if one breaks, the first
question is whether LILA republished — never edit an expected number to make it pass.

**Outstanding (does not gate training):** the two notebook/figure items above are
presentation deliverables. Gate B's condition is the assertion suite, which passes, so
Phase C is permitted. The notebook is scheduled with the other reporting work in G.

---

## 5. Phase C — FP32 baseline M0

### C0 — Golden preprocessing fixtures

Depends on: Gate B.

- [x] Select at least 20 validation fixtures covering edge cases.
- [x] Freeze raw image hashes and preprocessing metadata; tensor shapes remain
      provisional until C1a selects the input contract.

**Output:** frozen raw fixture set.

**Done 2026-07-15.** 20 fixtures in `tests/fixtures/golden_raw.json`, chosen
adversarially to the letterbox rather than sampled: every observed source geometry, the
aspect-ratio extremes, odd dimensions where the integer pad is asymmetric, a bobcat
frame and an empty frame. Drawn from validation only — DESIGN §5.4 seals the test
splits, and a fixture is read every time the C++ preprocessing is checked.

Tensor shapes are deliberately **not** frozen here. Freezing them before C1a picks the
input contract would either pin the wrong shape or quietly bless whichever one ran
first; the raw hashes are the part that is stable across that decision.

### C1 — Model and training engine

Depends on: Gate B.

- [x] Implement ImageNet-pretrained MobileNetV2 width 1.0 with configurable fixed
      input shape and 16 outputs (14 animals + `car` + `empty`).
- [x] Implement effective-number weighted cross-entropy and persist its numeric
      class-weight vector.
- [x] Implement two-phase head/full fine-tuning, checkpointing, early stopping,
      history logging, and run provenance.
- [x] Implement cis-val-clean/trans-val frame and sequence-balanced target metrics,
      support-aware macro F1, multi-label presence semantics, and selection score.
- [x] Add unit and smoke tests.

**Output:** tested training/evaluation engine and M0 config.

**Done 2026-07-15.** `train.py` (engine), `data/dataset.py` (dataset, augmentation,
weights), `metrics.py` (metrics and threshold selection), config `configs/train/m0_fp32.yaml`.
The engine is proven by C1a: three arms ran end to end under it. Width/height are config
keys rather than constants, which is what let C1a change the input geometry without
touching code.

The class-weight vector is persisted numerically into each run's history
(`train.py:400`), not just derived at runtime — DESIGN §6.2's weighting is a claim about
the run, so the run has to carry the actual numbers. `class_weights` counts
`primary_label` rather than the full label set (weighting by co-occurring labels would
inflate classes that appear alongside others) and floors an absent class at one sample:
before the supplement, CCT-20's train split has no `empty` at all, and `1/0` would
poison the entire vector.

Two bugs found here were the silent kind, and both are now regression-tested: a loop
rebound `index` in `__getitem__` so the returned dataset index became a *class* index —
every sequence metric would have been computed against the wrong sequences — and a
repeated `--override` silently replaced the earlier one (argparse `nargs="*"`), which
launched a C1a arm with the wrong step budget and the wrong dataset.

### C1a — Data and input controls

Depends on: C0, C1.

- [x] Run the matched no-empty 15-output versus 5k-empty 16-output ablation from
      DESIGN §5.2 and record cis/trans empty false-fire effects. Match the arms on
      **optimizer steps, not epochs** (13,546 vs 18,546 images = +36.9% steps per
      epoch); record steps, effective epochs, total images-seen, and non-empty
      images-seen for both.
- [x] Select the data/head contract from those two provisional 256x192 runs, reuse
      the winner as the landscape reference, and train exactly one additional
      matched 224x224 run. Do not run an unnecessary full 2x2 control matrix.
- [x] Select/freeze the Core input using cis-val-clean/trans-val target metrics,
      real-pixel utilization, and MACs; prefer 256x192 when statistically tied.
- [x] Permit 320x240 only if both planned inputs fail the bobcat-recall rule.
      **Not triggered** — both candidates meet it; see the caveat below.
- [ ] Generate canonical Python golden tensors for the selected input shape and
      freeze their hashes.

**Output:** `results/ablations/data_input_decision.md`, frozen preprocessing config,
and completed golden fixture set.

**Decided 2026-07-16.** **Core input 256x192; data/head contract 5k-empty, 16 outputs.**
Three arms, all on the same 6,000/1,055 optimizer-step budgets:

| arm | steps used | score | cis F2 | trans F2 | cis false-fire | trans false-fire |
|---|---:|---:|---:|---:|---:|---:|
| `c1a_empty5k_16out_256x192` | 4,335 | **0.4280** | 0.5875 | 0.2684 | 0.0423 | 0.0386 |
| `c1a_noempty_15out_256x192` | 4,220 | 0.3929 | 0.6028 | 0.1829 | 0.0547 | 0.0998 |
| `c1a_empty5k_16out_224x224` | 6,000 | 0.3926 | 0.6499 | 0.1353 | 0.0544 | 0.0300 |

**Neither decision was made on the selection score, because neither could be.** Both
gaps are ~0.035, and a single arm's score moves up to 0.099 between *consecutive*
epochs — the score is a max over that curve. A 10,000-replicate paired bootstrap over
sequence clusters (`validate.tie_test`; sequences, not frames, because CCT frames come
in bursts) calls both pairs **tied**:

- input: CI [-0.0824, +0.0143], P(224x224 better) 7.5%
- head: CI [-0.0095, +0.0792], P(supplement better) 93.9%

So each decision rests on something the data does support:

- **Input** — PLAN's pre-registered tie-break, which is not arbitrary here: at **-2.0%
  MACs** 256x192 carries **+31.1% real pixels** (97.47% utilisation against 72.83%;
  `validate.input_cost`). Both are ~49-50k-pixel tensors, but CCT's dominant frame is
  1024x747 and a square spends a quarter of itself on grey bars. DESIGN §5.5's Pi
  argument is the tiebreak's tiebreak: libjpeg scales 1024 by 1/4 during decode onto
  exactly 256, so the network input needs no resize step.
- **Head** — the false-fire effect, which is what the supplement was added to produce
  and is far larger than the F2 gap: trans 0.0386 against 0.0998, cis 0.0423 against
  0.0547. It reaches this having seen **25% fewer** animal frames, which is the opposite
  of what "it just saw more animals" would predict.

The 224x224 arm consumed its **whole 6,000-step budget** against the winner's 4,335 and
still lost, so the one step imbalance in the set runs against the selected shape rather
than for it.

**The caveat on the 320x240 bullet, which matters for C3.** Both candidates "meet" the
DESIGN §6.3 bobcat-recall rule, so the contingency is not triggered — but they meet it
only at thresholds of **0.0011** and **0.000049**, which fire on ~78% of trans frames at
a **67.6% false-fire rate**. The `non_trivial` guard rejects only a threshold that fires
on *literally every* frame (`fire_rate >= 1.0`), and 0.779 clears that bar while being
useless as a shutter trigger. DESIGN §6.3's primary rule has no false-fire ceiling: it
reports the rate (§6.4) but never constrains it, so as written it will hand C3 this
operating point and report the primary rule satisfied. Measured on the C1a arms, which
are shorter than M0 will be — but C3 must not read a satisfied rule as a working
trigger. Flagged for Vadym; not changed here, because it is DESIGN's rule to change.

### C2 — Train primary baseline

Depends on: C1a.

- [ ] Train seed 42 on gx10.
- [ ] Save best/last checkpoints and optimizer/scheduler state.
- [ ] Save full history, resolved config, environment, dataset/model hashes, and
      validation logits/predictions.
- [ ] Verify the selected checkpoint follows the configured rule.

**Output:** complete M0 seed-42 run directory.

### C3 — Calibrate and evaluate validation operating point

Depends on: C2.

- [ ] Search thresholds using cis-val-clean and trans-val only.
- [ ] Apply the two-domain 90% sequence-balanced recall rule from DESIGN §6.3.
- [ ] Bootstrap `seq_id` clusters and save the threshold distribution/95% interval.
- [ ] Without excluding or down-weighting short sequences, report the positive
      sequence-length distribution, `1-2`/`3-5`/`>5` recall where supported, and
      event capture rate alongside frame/sequence-balanced recall.
- [ ] Save `artifacts/policies/bobcat_v1.json` bound to class map/model hash.
- [ ] Implement the versioned generic policy schema with `mode: any`, non-empty
      unique animal targets, per-class thresholds, and model/class-map hashes;
      reject `car` and `empty` as wildlife targets. Runtime policy and catalog
      artifacts are JSON; training configuration may remain YAML.
- [ ] Produce validation precision/recall/F2/false-fire/fire-rate results and score
      distributions.

**Output:** versioned M0 bobcat policy, generic policy schema, and validation
report.

### C4 — Export and parity

Depends on: C3, C0.

- [ ] Export FP32 ONNX with fixed input/output contract, metadata, and the
      P0-accepted opset (provisionally 17).
- [ ] Pass P1 preprocessing parity against the reference C++ preprocessor.
- [ ] Pass P2 PyTorch-vs-ORT FP32 parity.
- [ ] Pass initial ORT Python-vs-C++ fixture parity.
- [ ] Save parity tolerances, raw comparisons, hashes, and failures if any.

**Output:** deployable M0 ONNX and parity report.

### C5 — Reproducibility confirmation and model card

Depends on: C4.

- [ ] Train confirmation seeds 17 and 73.
- [ ] Report validation mean/std for baseline training variability.
- [ ] Complete M0 model card: data, intended use, limitations, preprocessing,
      metrics, policy, license, and hashes.
- [ ] Add the M0 row to the machine-readable comparison table.

**Gate C:** M0 is reproducible, exported, parity-checked, calibrated, documented,
and ready for the same Pi application as optimized candidates.

---

## 6. Phase D — optimization ladder

### D1 — M1 INT8 PTQ

Depends on: Gate C.

- [ ] Build the fixed 1,024-image calibration manifest from training data only.
- [ ] Generate MinMax, Entropy, and Percentile static INT8 candidates.
- [ ] Use S8S8 QDQ as the primary representation; test QOperator only as an
      explicitly named ARM candidate. Save quantized/optimized graphs, ORT profile,
      operator/data-type coverage, and remaining FP32 nodes.
- [ ] Record the pre-registered MobileNetV2 PTQ risk before viewing results.
- [ ] Run quantization debugging for material accuracy drops.
- [ ] Calibrate candidate-specific bobcat policies on validation.
- [ ] Pass P3/P4 quantized ORT/C++ validation for the selected M1 candidate.

**Output:** selected M1 model, policy, profile, metrics, and comparison row.

### D2 — M2 INT8 QAT

Depends on: Gate C, A3.

- [ ] Initialize from M0 FP32, never from M1 PTQ.
- [ ] Run the validated affine INT8 fake-quant/QAT recipe.
- [ ] Search only the documented low learning-rate range on validation.
- [ ] Export a genuinely quantized ONNX graph.
- [ ] Inspect integer execution using exported/optimized graphs,
      operator/data-type coverage, ORT profile, and latency together rather than a
      single version-specific fused-node name.
- [ ] Calibrate policy and pass P3/P4.

**Output:** selected M2 model, policy, profile, metrics, and comparison row.

### D3 — Pruning sensitivity

Depends on: Gate C.

- [ ] Adapt `hw1/src/structured.py` to the frozen MobileNetV2 input and restrict
      Core pruning roots to expansion channels. Each dependency group must couple
      expansion output/BN, depthwise input/output/groups/BN, and projection input;
      keep projection/residual widths, stem, final conv, and classifier fixed.
- [ ] Set `round_to=8` on the pruner (`hw1/src/structured.py:33`) so surviving
      widths stay SIMD-aligned. Unaligned widths make MACs fall while latency does
      not, which would turn the pruning verdict into an artifact of the solver
      rather than a fact about MobileNetV2.
- [ ] Profile M0 parameters/MACs.
- [ ] Produce sensitivity evidence for dependency groups.
- [ ] After each pruning step, test depthwise group/channel equality, residual-add
      shapes, forward/backward execution, classifier width, and ONNX export.

**Output:** sensitivity report and reproducible pruning config.

### D4 — M3 structured-pruned FP32

Depends on: D3.

- [ ] Create approximately 15%, 30%, and 45% MAC-reduction candidates with
      `round_to=8`; record requested versus realized MAC reduction separately.
- [ ] Physically remove channels and verify changed shapes/MACs.
- [ ] Assert every surviving channel count is a multiple of 8 before fine-tuning.
- [ ] Fine-tune each under the fixed data/loss contract.
- [ ] Export deployable candidates with the P0-accepted opset and parity-check
      them; verify changed physical shapes/MACs in ONNX.
- [ ] Calibrate policies and add all validation rows.
- [ ] Select one M3 point for M4 using the validation Pareto frontier.

**Output:** M3 candidate set, selected checkpoint, models, policies, and evidence.

### D5 — M4 structured-pruned + QAT

Depends on: D4, D2.

- [ ] Apply the validated QAT recipe to the selected M3 FP32 checkpoint.
- [ ] Export, profile, calibrate, and pass P3/P4.
- [ ] Add M4 to the comparison table without assuming it is the winner.

**Output:** M4 model, policy, profile, metrics, and comparison row.

### D6 — Freeze deployable pre-Pi shortlist

Depends on: D1, D2, D4, D5.

- [ ] Reject any candidate failing correctness/export/parity gates.
- [ ] Apply DESIGN §8.5 validation selection rules.
- [ ] Use `gx10` latency only to detect float fallback/pathologies, never to rank
      Cortex-A76 candidates.
- [ ] Remove candidates dominated on validation bobcat F2, MACs, and model size.
- [ ] Write `results/model_selection/pre_pi_shortlist.md`, including every
      rejection and all non-dominated deployable candidates.
- [ ] **Build and freeze `benchmark_val_1000.jsonl`** per DESIGN §12.2. No earlier
      task owned this file, yet E7 packages it and F4 runs the mandatory parity on
      it. Stratify by bobcat, empty, rare, multi-label, and preprocessing edge
      cases, and add the dedicated **threshold-adjacent stratum**
      (`|score - threshold| < eps`, over-sampled) using the M0 operating point.
      Only those frames can flip a decision between gx10 and the Pi, so a subset
      without them can pass while proving nothing. The manifest is fixed and
      identical for every model, including M0-FP32.
- [ ] Freeze models, candidate-specific bobcat policies, preprocessing, class map,
      and hashes for Pi validation; keep test labels sealed.

**Gate D:** M0 and the complete deployable optimized shortlist are frozen for Pi
validation. No final optimized winner has been selected using `gx10` latency.

---

## 7. Phase E — C++ application and deployment bundle

### E1 — C++ project foundation

Depends on: A4; harden the smoke implementation using M0.

- [ ] Replace the 145-line course smoke test with a C++17 application/library
      structure, tests, configuration, and CLI.
- [ ] Implement RAII/error/logging conventions and deterministic JSON schemas.
- [ ] Vendor a pinned `nlohmann/json` single header plus license/version/hash;
      require no system JSON/YAML development package for the runtime bundle.
- [ ] Pin the ORT CPU EP build and compiler flags. Use target-scoped Release `-O3`;
      forbid `-march=native` on `gx10`. Permit explicit Pi CPU tuning or `native`
      only for a build performed on the same Pi.
- [ ] Prefer one proven official ONNX Runtime Linux AArch64 artifact for the clean
      `gx10` target environment and Pi; fall back to pinned source build only if
      P0 proves the artifact incompatible.

### E2 — Preprocessing

Depends on: E1, C0; smoke path is provisional until C1a freezes the input.

- [ ] Implement correct reference preprocessing.
- [ ] Implement fused/preallocated preprocessing.
- [ ] Pass golden tensor fixtures for both paths.
- [ ] Reject the old BGR-as-RGB behavior.

### E3 — Model session and policy

Depends on: E1, C4.

- [ ] Implement model contract validation and ORT session/thread configuration.
- [ ] Default to `ORT_ENABLE_ALL`, support the registered E6 graph-level
      comparison, enable C++ profiling with an explicit file prefix, and support
      persistence of the session-optimized graph.
- [ ] Implement class-map/model-hash-bound loading of one or more target classes,
      each with its own threshold from JSON policy/catalog artifacts; Core
      combination semantics are `mode: any`.
- [ ] Implement `SHUTTER_TRIGGER=0/1` output with selected scores and passing
      targets in the structured inference result.
- [ ] Add single-target, multi-target, exact-boundary, `car`/`empty`/duplicate/
      unknown target, unsupported-mode, wrong-model, class-map, and threshold tests.

### E4 — Dataset runner

Depends on: E2, E3.

- [ ] Consume manifests deterministically.
- [ ] Emit ordered JSONL scores, classes, decisions, errors, and stage timings.
- [ ] Preserve complete label sets and match multi-label target-presence metrics.
- [ ] Define corrupt/missing-image behavior.
- [ ] Match Python validation outputs and confusion matrix.

### E5 — Benchmark and system monitor

Depends on: E4.

- [ ] Implement warm-up, repetitions, p50/p95/p99, inference/end-to-end FPS, and
      per-stage timings.
- [ ] Implement peak RSS and CPU-utilization capture.
- [ ] Capture available frequency/temperature/throttling signals and explicit
      `unavailable` values.
- [ ] Validate output schemas and percentile calculations.
- [ ] Report whether Pi p95 end-to-end meets the primary 200 ms / 5 FPS target
      and the aspirational 100 ms / 10 FPS target; do not treat them as measured
      until Phase F.

### E6 — Correctness and C++ optimization experiment

Depends on: E5, Gate D.

- [ ] Pass P1-P4 for M0 and every shortlisted optimized model.
- [ ] Measure reference-vs-fused preprocessing with model/config held constant.
- [ ] Compare full JPEG decode against reduced 1/2 and 1/4 decode; test 1/8 only
      with explicit validation accuracy/decision-drift evidence.
- [ ] Measure supported ORT graph levels, threads 1/2/4, memory arena on/off, and
      stable CPU affinity if exposed; change one factor at a time.
- [ ] Treat `ORT_ENABLE_ALL` as the default and `ORT_ENABLE_EXTENDED` as a measured
      alternative; retain optimized graphs and profiles for both rather than
      inferring execution type from node names alone.
- [ ] Keep reduced decode only if validation bobcat metrics meet the predeclared
      tolerance; it is not preprocessing parity.
- [ ] Run Python-vs-C++ validation dataset parity.
- [ ] On `gx10`, run all unit/integration/self-tests under both a clean native
      CPU-only build and the target-compatible ARM64 build.
- [ ] **Run P1-P4 for M0 and every shortlisted model under
      `qemu-aarch64 -cpu cortex-a76`** and record score deltas against native gx10,
      not just decision agreement. This is the pre-rental rehearsal of the §12.2
      parity claim: emulation withholds `i8mm`/`sve2`, so ORT dispatches the Pi's
      kernels and any divergence surfaces here, in minutes, instead of on Day 4 with
      the rental clock running. Expect the FP32 arm to move and INT8 not to.
- [ ] If the RPi 4 contingency is live, repeat under `-cpu cortex-a72` for
      dispatch evidence; Cortex-A72 has no `asimddp` and INT8 will differ.
- [ ] Record emulated **correctness** only. Emulated latency is not evidence and
      must not reach a results table.

**Gate E6:** the C++ application is correct before performance claims are made.

### E7 — Raspberry Pi deployment bundle

Depends on: E6.

- [ ] Package C++ executable, required runtime/install instructions, M0 and every
      shortlisted ONNX model/policy, class map, validation benchmark/parity subset,
      sample images, `install.sh`, `run_demo.sh`, and checksums.
- [ ] Generate bundle manifest with commit/model/policy hashes.
- [ ] Test installation in a clean target-compatible container on `gx10` without
      access to the host training environment or unbundled artifacts.
- [ ] If binary compatibility cannot be proven, include pinned source/build
      automation and make Pi-side compilation part of `install.sh`.

**Output:** versioned ARM64 deployment archive.

### E8 — Full ARM64 dry run

Depends on: E7.

- [ ] On `gx10`, run the exact future Pi provision/install/demo/benchmark
      commands inside the target-compatible ARM64 environment.
- [ ] Verify unattended execution and machine-readable outputs.
- [ ] Verify baseline is included in the measurement matrix.
- [ ] Copy and parse results using the reporting code.
- [ ] Record a known-good dry-run log for later diffing.

**Gate E:** the deployment bundle and one-command benchmark work end to end. Do
not rent the Pi before this gate.

`gx10` dry-run latency is diagnostic only. Phase F remains mandatory because only
measurements produced on a Raspberry Pi count as target-hardware evidence.

---

## 8. Phase F — Raspberry Pi target trial (Pi 5 preferred)

If the planned Pi 5 cannot be provisioned, use another Pi 5 provider and then an
RPi 4 provider if necessary. The assignment permits both. If no Pi is available,
Gate F fails and the result is a partial submission rather than completed Core;
never replace Pi measurements with `gx10` timings.

### F1 — Day 1: provision and smoke test

- [ ] Use `gx10` as the control/evidence host and verify the remote Pi connection.
- [ ] Record hardware/OS/kernel/governor/compiler/ORT/OpenCV/environment.
- [ ] Record exposed frequency, temperature, and throttling interfaces.
- [ ] Install the frozen bundle.
- [ ] Run self-test and a short validation smoke test.
- [ ] Do not tune models or inspect test labels.

### F2 — Day 2: validation performance profiling

Depends on: F1.

- [ ] Run the same fixed validation benchmark for M0 and every shortlisted model.
- [ ] Measure the bounded decode/preprocessing/ORT/thread/runtime matrix from E6.
- [ ] Run ORT/C++ profiles and identify bottlenecks; do not open test labels.
- [ ] Make only safe runtime fixes validated against parity tests.

### F3 — Day 3: repeat validation and freeze

Depends on: F2.

- [ ] Repeat validation after any safe fix.
- [ ] Select the final optimized model using validation accuracy, real Pi latency,
      model size, and simplicity; write `final_decision.md`.
- [ ] Build 14 threshold-catalog status entries. Emit numeric thresholds for the
      **11 selectable targets** (nine two-domain and two single-domain fallbacks)
      and null thresholds for unavailable `badger`, `deer`, and `fox`. Generate
      `threshold_catalog.json`, the bobcat policy, and validated
      `bobcat_coyote_v1.json`; record combined validation false-fire metrics.
- [ ] Freeze git commit, binary, selected model, policies, preprocessing/decode
      mode, ORT options, and thread count.
- [ ] Archive freeze manifest before test evaluation.
- [ ] **Only after the freeze:** launch confirmation seeds 17/73 for the selected
      transformation on `gx10` in the background with frozen hyperparameters. They
      measure variability, never replace the seed-42 deployment artifact, and must
      not gate this freeze, any later trial day, or Gate F. Retraining a
      pruned+QAT candidate twice can outlast a trial day; they may finish after the
      trial expires but must finish before Gate G and final submission.

**Freeze gate:** no artifact or configuration changes after this point.

### F4 — Day 4: frozen full test and Pi benchmark

Depends on: F3.

- [ ] On `gx10`, run full frozen cis-test and trans-test through the exact C++/ORT
      model/policy/runtime artifacts; save frame and sequence-aware metrics.
- [ ] Run full M0-vs-optimized Pi benchmark, at least 1,000 frames, three separate
      processes/repetitions as specified.
- [ ] Run the fixed Pi parity subset for **both M0-FP32 and the selected winner**
      and match decisions to the frozen gx10 reference; full test transfer to Pi is
      optional, not required. Record score deltas, not only decision agreement, so
      drift that has not yet crossed a threshold stays visible. Expect any
      divergence in the FP32 arm rather than the INT8 one: QDQ convolutions
      accumulate exactly in int32, while float accumulation order differs between
      GB10 SVE2 and Pi NEON.
- [ ] **If the parity subset disagrees, stop before claiming target equivalence.**
      Report score and decision mismatch rates and treat Pi decisions as
      authoritative for affected frames. If the difference cannot be explained,
      either run full test accuracy on Pi or report the gx10 C++ accuracy only as
      gx10 evidence with the Pi-equivalence claim explicitly withheld.
- [ ] Capture raw per-frame predictions/timings/system logs.
- [ ] Copy artifacts off the Pi immediately.

### F5 — Day 5: unchanged reproducibility repeat

Depends on: F4.

- [ ] Repeat the frozen Pi benchmark/parity run without changes.
- [ ] Compare runs and investigate only measurement anomalies, without tuning.
- [ ] Back up every result, log, environment file, binary, and model.
- [ ] Copy and checksum all raw Pi evidence back to `gx10`.
- [ ] Verify result schemas and hashes before the trial expires.

**Gate F:** baseline and optimized Pi evidence contains latency, FPS, RSS, CPU
utilization, model size, available thermal/frequency data, parity, and raw
repetitions under a frozen protocol; full frozen test accuracy exists from gx10.

---

## 9. Phase G — analysis, public release, and submission

### G1 — Freeze analysis dataset

Depends on: Gate F.

- [ ] Validate and index all raw training/evaluation/parity/Pi result files.
- [ ] Confirm seeds 17/73 have finished for M0 and the selected final
      transformation; archive their variability metrics before continuing.
- [ ] Create a machine-readable canonical results table.
- [ ] Record missing/unavailable fields explicitly.

### G2 — Results notebook and figures

Depends on: G1.

- [ ] Complete and clean-run `notebooks/02_results_analysis.ipynb`.
- [ ] Generate every table/figure required by DESIGN §17.
- [ ] Compute sequence-bootstrap metric/threshold intervals, support-aware macro
      F1, official-vs-clean cis-val effects, per-location trans recall,
      sequence-length-stratified recall, and event capture rate.
- [ ] Report multi-label exclusions for confusion/macro metrics and retain those
      images in target-presence metrics.
- [ ] Select representative failure cases without hiding negative results.

### G3 — Final report

Depends on: G2.

- [ ] Write all sections specified by DESIGN §15.
- [ ] Include public repository/release placeholders and later replace them.
- [ ] Separate measured, published, and estimated values.
- [ ] Include what worked, what failed, bottlenecks, limitations, and next steps.
- [ ] Export and visually inspect `final_report.pdf`.

### G4 — Slide deck

Depends on: G2, G3.

- [ ] Create the 10-12 slide narrative from DESIGN §16.
- [ ] Include repository URL and QR on first/final slides.
- [ ] Include result units, sample counts, run/commit identity, and limitations.
- [ ] Export PDF and visually inspect every slide.
- [ ] Rehearse explanations for C++, pruning, quantization, thresholds, and every
      headline number.

### G5 — Public repository and submission manifest

Depends on: G3, G4.

- [ ] Clean README quick start and reproduction commands.
- [ ] Publish code without secrets/data archives.
- [ ] Publish models/deployment bundle through GitHub Release or LFS.
- [ ] Verify checksums/download links from a clean environment.
- [ ] Tag `v1.0-final` and record the commit hash.
- [ ] Replace every `REPO_URL`/`RELEASE_URL` placeholder.
- [ ] Complete `SUBMISSION.md` with canonical links and commands.
- [ ] Run every Definition of Done item in DESIGN §19.

**Gate G:** the complete Core submission is public, reproducible, visually
checked, and every headline number traces to raw machine-readable evidence.

---

## 10. Optional Phase S — crop-teacher KD

This phase is locked until Gate G. It must not alter the frozen Core release.

### S1 — crop data and teacher

- [ ] Build GT-crop manifests using train boxes only.
- [ ] Define and test multi-box/padding behavior.
- [ ] Train a 15-non-empty-class crop teacher (14 animals + `car`).

### S2 — fair control and KD

- [ ] S0: reproduce Core FP32 student budget.
- [ ] S1: crop augmentation without teacher.
- [ ] S2: cross-view KD with identical student initialization/budget.
- [ ] Apply KD only to non-empty samples and align all 15 non-empty logits.

### S3 — decision

- [ ] Report KD as successful only if S2 beats S1.
- [ ] Preserve a null result if it does not.
- [ ] Publish Stretch as a separate config/result/release addition.

---

## 11. Session handoff template

At the end of each implementation session, create a new dated handoff containing:

- tasks completed and gates passed;
- exact runs/models/artifacts created;
- commands and environment used;
- failures and diagnostic evidence;
- current git status and whether changes are committed/pushed;
- next unblocked task ID from this plan;
- decisions that changed DESIGN/PLAN and why;
- external state required next, such as gx10 or Pi credentials.

The next session starts from the newest handoff and verifies the filesystem before
trusting prose.
