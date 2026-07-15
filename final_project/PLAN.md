# Final Project — Work Plan

Execution plan for the design in [DESIGN.md](DESIGN.md). Read that first: this
document says **what we build, in what order, and what has to be true before the
next thing starts**. It contains no new decisions — where it looks like a decision,
DESIGN.md is the source of truth.

Status: **not started.** Nothing below is done.

---

## 1. The one constraint that shapes everything

The Pi is rented on a **5-day free trial** (Hostpro). The trial is not development
time — it is **measurement time**. Anything authored, debugged, or discovered
during the trial is a day burned from a 5-day budget that cannot be extended.

> **The rule: nothing runs on the Pi for the first time during the trial.**
> Every artifact and every script arrives already exercised end-to-end on the
> ARM64 devcontainer. The trial re-runs a known-good procedure on real silicon.

This is what makes the devcontainer (§8 of DESIGN.md) load-bearing rather than
convenient, and it dictates the phase order below: **the trial is opened last, and
only once the dry run has passed.**

Second constraint, from the rubric: *"measure and document on-device latency, FPS
and resource utilization for your **baseline** model"*. The FP32 baseline is not
just a training artifact — it is a **measured deliverable on the Pi**. It ships to
the trial alongside the optimized models. Forgetting this costs 10 points and
cannot be fixed after the trial closes.

## 2. Dependency graph

**Three models exist in this project** (DESIGN.md §7 roster): **Gate** and
**Species** ship to the Pi; the **crop-teacher** never leaves gx10. MegaDetector is
cited, never run. The ladder M5–M8 compresses **Species only**.

```
  D1 data ─┬─ D2 splits ─┬─ D3 crops ──┬─ M1 crop-teacher ─┐
           │             │             │                   ├─ M3 student+KD ─┐
           │             │             └─ M2b crop-aug ────┤   (judged vs M2b)│
           │             ├─ M2 student FP32 (baseline) ─────┘                │
           │             │                                                   │
           │             └─ D4 gate labels ── M4 gate model ─────────────────┤
           │                                                                 │
           └─ D5 eval subsets ───────────────────────────────────────────┐   │
                                                                         │   │
   M3 ─── M5 PTQ ── M6 QAT ── M7 prune+QAT ── M8 full stack ─────────────┼───┤
                                                                         │   │
                                          M9 ONNX export + parity gate ──┴───┘
                                                       │
                                     C1 cascade app ───┤
                                     C2 fused preproc ─┤
                                     C3 bench harness ─┼── DRY RUN (devcontainer)
                                     C4 dataset runner ┘         │
                                                                 ▼
                                                        ===  TRIAL OPENS  ===
                                                          T1 measure · T2 repeat
                                                                 │
                                                                 ▼
                                                            W write-up

   Off the critical path, any time after D3:
     X3 unstructured pruning (the negative-result row)
```

**Critical path:** `D1 → D2 → D3 → M1 → M3 → M6 → M8 → M9 → C1 → dry run → trial`.
Everything else has slack. If something must be cut, cut from X1–X3 and from the
stretch elephant module — never from the path.

## 3. Phase D — Data (gx10)

The whole project's credibility rests here. A split bug is not a bug — it is a
retraction.

| # | Task | Done when |
|---|---|---|
| **D1** | Download CCT-20 (~6 GB images + metadata) to gx10; verify checksums; record dataset version | Image count reconciles to **57,868**; classes reconcile to **15 + empty** |
| **D2** | Build the official cis/trans splits from the metadata — *not* our own splits | Counts match DESIGN.md §6 **exactly**: train 13,553 · cis-val 3,484 · cis-test 15,827 · trans-val 1,725 · trans-test 23,275 |
| **D3** | Extract GT bounding-box crops for the teacher | Crop count reconciles to the box annotations; crops spot-checked visually |
| **D4** | Derive gate labels (empty / non-empty) from the class labels | Empty share ≈ the ~70% the design assumes — **if it is not ~70%, the cascade's energy maths in §3 changes and must be recomputed before it is quoted** |
| **D5** | Build eval subsets: day vs night-IR, small-animal | Subsets are stable, versioned, and reproducible from a seed |

**D2 is a gate, not a step.** Three assertions run in CI-style before any training:

1. **No sequence spans train and val** (the paper's split rule).
2. **Location disjointness:** `locations(train) ∩ locations(trans-test) = ∅`, and
   likewise for trans-val. This assertion is what keeps "trans" meaning
   *unseen location*, and it is the same assertion that guards the contingency in
   DESIGN.md §11 if we ever extend the training data.
3. **Split counts equal the published numbers.** If they do not, we have
   misunderstood the metadata, and every number we produce afterwards is
   incomparable to the published baseline.

**Test-set discipline (course rule, memory):** `cis-test` and `trans-test` are
**touched once**, at the end, for the final table. All intermediate decisions —
thresholds, early stopping, model selection, the compression ladder's internal
comparisons — use **cis-val / trans-val only**. There is no "quick check on test".

## 4. Phase M — Models (gx10)

Each row lands the §9 metric set on **both** cis and trans, at the operating point,
recorded to one results file as it is produced. We do not reconstruct a results
table from memory at slide-writing time.

| # | Task | Depends | Done when |
|---|---|---|---|
| **M1** | FP32 crop-teacher (ResNet50 / EffNet-B0 on GT crops) | D3 | Beats the student on cis; is the accuracy ceiling (ladder row 0) |
| **M2** | FP32 student — MNv2 ImageNet-pretrained, full frame, **no KD** | D2 | **The baseline** (row 1). Sanity anchor: cis error in the neighbourhood of the published ~20.8%; trans roughly double it |
| **M2b** | Student, **crops as extra training data**, no KD | D3, M2 | Row 2b — **the control**. Cheap: one dataloader change, no teacher |
| **M3** | Student **+ crop-teacher KD** | M1, M2, **M2b** | Row 2. Judged against **M2b, not M2** — beating the naive baseline proves only that crops carry information. **The §5 ablation resolves either way**; a null result is a reportable finding, not a failure |
| **M4** | Gate model (MNv2 `width_mult=0.35` @128²) | D4 | High recall on non-empty; measured share of frames exiting at the gate |
| **M5** | INT8 PTQ | M3 | Row 3 |
| **M6** | INT8 QAT (`torch.ao`) | M5 | Row 4; QAT ≥ PTQ or we explain why |
| **M7** | Structured channel pruning + QAT | M6 | Row 5; **real MAC reduction**, not just a smaller file |
| **M8** | KD + pruned + QAT | M7 | Row 6 — the full stack |
| **M9** | ONNX QDQ export + **parity gate** | M8 | See below |
| **M10** | Threshold calibration → module YAML (`bobcat-v1`) | M8 | Calibrated on **val**; the YAML in DESIGN.md §3 gets real numbers |

**Reuse (DESIGN.md §7):** `hw3/src/distill.py` drives M3 directly; `hw1/src/structured.py`
drives M7 (example input 32² → 224²); the `hw2`/`hw3` QAT loop drives M6 with the
quantizer swapped to affine INT8. `hw2/src/kmeans_quant.py` does not transfer.

**M9 is the second gate.** ONNX export is where silent corruption enters — a QDQ
graph that loads fine and is quietly wrong. Before any C++ work consumes a model:

> **Parity check:** ORT and PyTorch must agree on the **same fixed image set** —
> logits within tolerance, and an **identical confusion matrix and identical
> fired/not-fired decisions**. A model that fails parity does not reach the Pi.

Without this, a C++ bug and an export bug are indistinguishable on the Pi, during
a 5-day window. This check costs an hour now and can save the trial.

**Off-path (start once D3 lands, drop without regret if time runs short):**

- ~~**X1** — stage-2 externally-pretrained teacher~~ — **dropped 2026-07-15**
  (DESIGN.md open question 8). MegaDetector does not classify species, so it was
  never a possible teacher. Our crop-teacher (M1) is the teacher, and always was.
- ~~**X2** — run MegaDetector for the option-B ceiling~~ — **dropped 2026-07-15**.
  **M1 already is that ceiling**: the crop-teacher classifies GT-box crops, i.e.
  option B handed a free perfect detector, so it upper-bounds B (DESIGN.md §4). The
  bound is the stronger argument *and* free. **MegaDetector is not run at all** —
  §4 cites its published MACs, which needs no weights.
- **X3** — unstructured pruning: the honest negative-result row.

## 5. Phase C — The C++ application (ARM64 devcontainer)

Depends on **M9 passing**, not on the ladder being finished: C1–C4 can be built
against the M2 baseline export and later re-pointed at the final models. This is
deliberate — it takes the C++ off the critical path's tail, where schedule risk
concentrates.

| # | Task | Earns |
|---|---|---|
| **C1** | Cascade + decision logic: gate → threshold → species → threshold → shutter → sleep | The design, in code (15 pts, item 1) |
| **C2** | **Fused preprocessing** — replace the container's 4 passes over memory (`convertTo`→`subtract`→`divide`→`split`) with one, copies removed | The measurable low-level win |
| **C3** | Benchmark harness — p50/p95/p99, thread sweep, thermal logging, stage timing | 10 pts; **none of this exists in the container** |
| **C4** | Dataset runner — iterate the test set, emit decisions + latencies + confusion matrix | Makes the trial a batch job, not an interactive session |

**Correctness gate for C1–C4:** the dataset runner's confusion matrix must equal
the Python one from M9, on the same images. Same rationale as the parity gate — we
establish that the C++ is *right* in the devcontainer, so the trial only has to
establish how *fast* it is.

**In reserve (DESIGN.md §8):** hand-write the gate's engine only, if the C++ points
need reinforcing and the schedule allows.

## 6. Phase DRY — Dry run (devcontainer). The gate to the trial.

The full benchmark protocol (DESIGN.md §10), executed end-to-end on ARM64, exactly
as it will be run on the Pi — same scripts, same flags, same outputs.

**Exit criteria — all must hold before the trial is provisioned:**

- [ ] Every model in the ladder is exported, parity-checked, and **on disk**
- [ ] The **FP32 baseline is in the measurement set** (the rubric's baseline-vs-optimized comparison)
- [ ] One command runs the whole protocol unattended and writes machine-readable results
- [ ] The results file is the direct input to the slides — no manual transcription
- [ ] A dated run log exists, so a trial anomaly can be diffed against a known-good run

Anything not on this list at trial time does not go to the Pi.

## 7. Phase T — Trial (5 days, rented Pi 5). Measurement only.

| Day | Work |
|---|---|
| **1** | Provision, SSH, deps. **Answer open question §11.5 immediately:** does the instance expose `vcgencmd` and governor control? Report what is available either way. Smoke-test the app. |
| **2** | Full protocol: baseline + ladder, cis/trans, day/night, thread sweep. Thermals logged throughout. |
| **3** | Read results. Profile bottlenecks (DESIGN.md §8 item 4). Fix only what is cheap and safe. |
| **4** | **Re-run.** This day exists so that day 2 is allowed to go wrong. |
| **5** | Buffer. Pull every artifact off the instance **before** the trial expires. |

**Replace every estimate with a measurement** (open question §11.7). The `~5 ms` /
`~25 ms` / `~12.5 ms` figures throughout DESIGN.md are estimates and are marked as
such; each one is either measured or removed from the report. None of them enters
a slide as a claim.

## 8. Phase W — Write-up

The rubric's Results Analysis (10 pts) asks four questions explicitly, so the deck
answers them **by name**: what worked · what didn't · what the hardware/software
bottlenecks were · concrete next steps.

We are unusually well positioned here, because the design already commits to
reporting things that did not work: the unstructured-pruning row (X3), the
possibility that KD's attention transfer is null (M3), the possibility that
structured pruning barely helps MobileNetV2 (§11.3), and trans recall if it
collapses (§11.2). **Planned negative results are worth more than a clean sweep** —
they are what "critical evaluation" means, and they cannot be manufactured in the
last week if the ladder was not designed to expose them.

Slides carry the limitations honestly: energy is **proxied by latency**, not
measured (no power meter on a datacenter Pi); no camera, PIR or GPIO; the elephant
module, if it appears, carries the Serengeti sequence-label caveat on its own slide.

## 9. Rubric coverage

| Criterion | Pts | Where earned |
|---|---:|---|
| Model training & optimization strategy | 15 | M1–M8 ladder + DESIGN.md §4–§7 (the *justification* is the doc; the ladder is the evidence) |
| C++ inference implementation | 15 | C1–C4; gate-engine in reserve |
| Benchmarking & metrics | 10 | C3 + Phase T, protocol §10, baseline **and** optimized on-device |
| Results analysis & presentation | 10 | Phase W; the planned negative results are the asset |

## 10. What would sink this, in order of likelihood

1. **A split bug** — silently inflates every number. Guarded by the D2 assertions.
2. **Trial time spent debugging** — guarded by the dry-run exit criteria.
3. **An export/parity bug found on the Pi** — guarded by the M9 gate.
4. **The baseline not measured on-device** — costs 10 points; it is on the dry-run
   checklist for exactly this reason.
5. **Scope creep from the stretch items** — X1–X3 and the elephant module are
   droppable by construction, and dropping them is the plan, not a failure.
