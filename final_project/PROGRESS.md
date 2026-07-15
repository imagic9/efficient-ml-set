# Progress board

**Where we are:** Phase A, 3 of 5 tasks done. **Next: A3 — the P0 toolchain spike.**

A one-line-per-task status board. [`PLAN.md`](PLAN.md) says *what each task must
do*; this file says *whether it is done and what comes next*. When the two
disagree, **PLAN.md is right** — fix this file, not the plan.

Updated after every task. `[x]` only when the artifact exists and its checks pass.

Legend: `[ ]` not started · `[~]` in progress · `[x]` done and verified · `[!]` blocked

---

## Right now

| | |
|---|---|
| **Branch** | `phase-a/repo-and-toolchain` (22 commits, not yet merged) |
| **Last done** | A2 — reproducible environments |
| **Next** | **A3 — P0 toolchain spike**, the riskiest task in the project |
| **Blocked on** | nothing |
| **Gate ahead** | Gate A — no data download or long training before it passes |

### Why A3 is the risky one

It has to prove one end-to-end **QAT → ONNX → ORT C++** path exists at all, and
pick the library that provides it — the design deliberately does not pre-commit
one. If none of the three candidates emits a QDQ graph ORT runs as integer, M2
and M4 lose their basis and the ladder needs rethinking. Better to learn that now
than after the data and the baseline are built.

It must also hold under `qemu -cpu cortex-a76`: a quantized path that only works
because gx10 has `i8mm` is a **P0 failure**, not a pass.

---

## Phase A — repository and toolchain

| | Task | Evidence |
|---|---|---|
| `[x]` | **A0** Record starting state | `results/provenance/project_start.json` |
| `[x]` | **A1** Repository skeleton | pytest 40 · ctest 1/1 |
| `[x]` | **A2** Reproducible environments | `requirements.lock` · `scripts/verify_target_env.sh` |
| `[ ]` | **A3** P0 toolchain spike ← **NEXT** | opset-17 FP32/PTQ/QAT in ARM64 C++ ORT |
| `[ ]` | **A4** Early C++ vertical slice | JPEG → C++ → ORT → policy → shutter JSON |
| | **Gate A** — all three model forms load in ARM64 C++ ORT and the thin path runs end to end | |

**What A0–A2 established that later phases depend on:**

- gx10 is glibc **2.39**, Pi OS Bookworm is **2.36** → build in `debian:bookworm-slim`.
- gx10 has `i8mm`/`sve2`, Pi 5 Cortex-A76 has neither → `qemu -cpu cortex-a76` rehearses the Pi's kernels.
- **Pi 4's Cortex-A72 has no `asimddp`** → RPi 4 is a degraded target, not an equal one.
- ORT's aarch64 tarball needs only GLIBC_2.27 → one identical ORT binary for gx10 and Pi.
- Python and C++ ORT are both **1.27.0** on purpose; a split would make P1/P3 measure the wrong thing.
- OpenCV 4.13 (Python) vs 4.6 (C++) — a **known gap left for P1 to quantify**.

## Phase B — data

| | Task | Notes |
|---|---|---|
| `[ ]` | **B0** Acquire and fingerprint sources | ~8.1 GB; record real image dimensions, do not inherit them |
| `[ ]` | **B1** Official split manifests | 13,553 / 3,484 / 15,827 / 1,725 / 23,275; build `cis_val_clean` |
| `[ ]` | **B2** `cct_empty_train_v1` | 5,000 empties — **must downsize to 1024 px** + shortcut probe |
| `[ ]` | **B3** Data and preprocessing code | |
| `[ ]` | **B4** Data audit gate | |
| | **Gate B** — every §5.3 assertion passes. No training before it | |

## Phase C — FP32 baseline M0

| | Task | Notes |
|---|---|---|
| `[ ]` | **C0** Golden preprocessing fixtures | |
| `[ ]` | **C1** Model and training engine | |
| `[ ]` | **C1a** Data and input controls | 3 runs, not a 2×2: empty ablation ×2, then one 224×224 |
| `[ ]` | **C2** Train primary baseline | seed 42 |
| `[ ]` | **C3** Calibrate operating point | validation only |
| `[ ]` | **C4** Export and parity | P1–P4 |
| `[ ]` | **C5** Confirmation seeds and model card | seeds 17, 73 |
| | **Gate C** — M0 reproducible, exported, parity-checked, calibrated | |

## Phase D — optimization ladder

| | Task | Notes |
|---|---|---|
| `[ ]` | **D1** M1 INT8 PTQ | |
| `[ ]` | **D2** M2 INT8 QAT | uses the library A3 selects |
| `[ ]` | **D3** Pruning sensitivity | `round_to=8` — unaligned widths make the verdict an artifact |
| `[ ]` | **D4** M3 structured-pruned FP32 | |
| `[ ]` | **D5** M4 pruned + QAT | |
| `[ ]` | **D6** Freeze pre-Pi shortlist | also **builds `benchmark_val_1000.jsonl`** |
| | **Gate D** — shortlist frozen; gx10 latency never ranked candidates | |

## Phase E — C++ application and bundle

| | Task | Notes |
|---|---|---|
| `[ ]` | **E1** C++ project foundation | vendor `nlohmann/json` here |
| `[ ]` | **E2** Preprocessing | needs C0 |
| `[ ]` | **E3** Model session and policy | needs C4 |
| `[ ]` | **E4** Dataset runner | |
| `[ ]` | **E5** Benchmark and system monitor | |
| `[ ]` | **E6** Correctness and optimization experiment | **P1–P4 under `-cpu cortex-a76`** |
| `[ ]` | **E7** Pi deployment bundle | resolve the OpenCV soname: bundle or link statically |
| `[ ]` | **E8** Full ARM64 dry run | |
| | **Gate E** — one-command benchmark works. **Do not rent the Pi before this** | |

## Phase F — Raspberry Pi trial

Vadym rents the Pi once Gate E passes. No calendar; the trial is one-shot and
must not start early.

| | Task | Notes |
|---|---|---|
| `[ ]` | **F1** Day 1 — provision and smoke test | |
| `[ ]` | **F2** Day 2 — validation profiling | |
| `[ ]` | **F3** Day 3 — select final model, freeze | launch seeds 17/73 after the freeze |
| `[ ]` | **F4** Day 4 — frozen test on gx10, Pi benchmark | parity on M0-FP32 **and** the winner |
| `[ ]` | **F5** Day 5 — unchanged repeat, back up | |
| | **Gate F** — baseline-vs-optimized Pi evidence with 3 repetitions | |

## Phase G — analysis and submission

| | Task |
|---|---|
| `[ ]` | **G1** Freeze analysis dataset |
| `[ ]` | **G2** Results notebook and figures |
| `[ ]` | **G3** Final report |
| `[ ]` | **G4** Slide deck |
| `[ ]` | **G5** Public repo and submission manifest |
| | **Gate G** — Core Definition of Done passes |

## Phase S — optional Stretch

Locked until Gate G. Crop-teacher KD only.

| | Task |
|---|---|
| `[ ]` | **S1** Crop data and teacher |
| `[ ]` | **S2** Fair control and KD |
| `[ ]` | **S3** Decision — S2 must beat S1, or report the null result |

---

## Design decisions already closed

Do not re-litigate these without a documented reason; each cost a review round.

| Decision | Where |
|---|---|
| NAS excluded from Core | DESIGN §7.1 |
| Full test accuracy runs on gx10, licensed by the Pi parity subset | DESIGN §12.2 |
| Empty supplement must be location-disjoint — no alternative exists | DESIGN §5.2 |
| 256×192 provisional input; `1024/4 = 256` aligns the JPEG decoder | DESIGN §5.5 |
| The empty ablation cannot change the 16-output head | DESIGN §5.2 |
| Emulated timings are never results | DESIGN §4 |

## Closed issues

`#3` gitignore blocker · `#4` data resolution · `#5` pruning/QAT · `#6` parity subset ·
`#7` doc consistency · `#9` QEMU ISA parity
