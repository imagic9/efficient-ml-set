# Final Project — Design

Living design note for the Edge AI final project.
Status: **design agreed, implementation not started.**

Numbers in this document are marked either as **measured/published** (with a
source) or as **estimate** (to be replaced by our own measurements). Nothing here
is a claim until it is measured on the Pi.

---

## 1. The product

**One line:**

> A Raspberry Pi device that fires a wildlife photographer's camera only when the
> animal they actually want walks into frame.

**Who it is for.** Wildlife photographers running unattended camera traps in
remote locations for weeks or months. They come back to thousands of frames of
wind-blown grass, and the wrong animals. The shots they wanted are buried, the
card is full, and the battery died three weeks in.

**How it works:**

```
PIR/motion trigger
   → wakes our device
   → device captures a preview frame and analyses it
   → target species present?  → fire the main camera (shutter signal)
   → back to sleep
```

**Product form.** A small Raspberry Pi-based box that sits between a motion sensor
and the photographer's camera and controls the shutter. The photographer
downloads a **module** for the animal they want to shoot (`bobcat-v1`,
`elephant-v1`) from our site.

**Why this must run on the device.** The decision has to happen in the field, in
milliseconds, offline, on battery. There is no link to a cloud, and there would be
no time to use one even if there were. This is Edge AI in its purest form: the
latency *is* the product.

**Honest scenario boundary — "lingering animals".** The device adds inference to
the critical path between motion and shutter. A Pi cannot decide in the ~50–200 ms
that professional PIR triggers (Camtraptions, Cognisys) achieve, and it certainly
cannot boot from sleep in that time. So we scope to subjects that **stay in
frame**: waterholes, feeding sites, carcasses, salt licks, dens, trail
bottlenecks. A leopard at full sprint is out of scope, and we say so. This is not
a small niche — it is a large share of how real camera-trap photography is set up.

## 2. What the course project demonstrates

The project answers one falsifiable engineering question:

> **Can a Raspberry Pi 5 decide fast enough, and cheaply enough, to be the core of
> this device?**

Concretely: `is there anything in frame?` → `what is it?` → `fire the shutter if
it is the target`. Every millisecond we save is a better photograph, and every
millijoule is another night in the field. Unlike a generic "make it faster"
exercise, here latency and energy have a physical meaning, which is what makes
this a good fit for the course brief ("improve FPS, ideally to real-time").

**In scope:** the decision core — model, compression, C++ application, on-device
benchmarks, on a saved dataset (the brief explicitly permits this).

**Out of scope, and why:**

| Excluded | Reason |
|---|---|
| Camera, PIR, GPIO shutter signal | The rented Pi is a datacenter instance — no physical access. The shutter decision is printed, not wired. |
| Power measurement | Same. Energy is **proxied by latency**; stated as a limitation, not hidden. |
| Our own from-scratch inference engine | Decided against: ONNX Runtime is a proven, optimized runtime. Our C++ is the application (§8). |
| A detector on-device | Quantified trade-off, §4. |
| Hailo / AI HAT+ | The rubric wants our C++ on the CPU, not a vendor compiler. Phase 1. |
| Audio (chainsaw/gunshot) | Second modality. Phase 1. |

## 3. On-device pipeline

A **true cascade of two separate models** — cheap one first.

```
frame (PIR already fired)
   │
   ├─ [GATE]  tiny CNN, binary: empty / non-empty        ~5 ms   (estimate)
   │     │
   │     ├─ empty  (~70% of triggers) → sleep.  Species model never ran.
   │     │
   │     └─ non-empty (~30%)
   │           │
   │           └─ [SPECIES]  MobileNetV2, full frame     ~25 ms  (estimate)
   │                 │
   │                 └─ target class ≥ threshold → SHUTTER FIRE
```

**Why two separate models, not two heads on a shared backbone.** A shared backbone
would be paid on every frame — the cascade saving would be nil. The saving comes
from the gate being a *genuinely different, much smaller network* that lets ~70%
of frames exit early. (An earlier revision of this document conflated the two;
that was wrong.)

**Gate:** MobileNetV2 `width_mult=0.35` @128² (or a small custom CNN) — roughly
25× cheaper than the species model. Binary, calibrated for **high recall on
non-empty**: the gate must never be the reason we miss a shot.

**Species:** MobileNetV2 @224², multi-class over the CCT-20 classes. One forward
pass yields probabilities for **all** classes.

### Modules

A **module** is not a model. It is a **policy**: a target class, a calibrated
threshold, and the honest expectation that goes with them.

```yaml
module_id: bobcat-v1
target_class: bobcat
threshold: 0.62          # calibrated on val for 94% recall
expected:
  recall_cis:  0.94      # at a location like the ones it was trained on
  recall_trans: 0.81     # at a brand-new location
  false_fire_rate: 0.04
```

The separation matters: **the model is the capability, the module is the intent.**
What is in frame is a fact the model computes; whether it is worth a photograph is
a human choice that cannot be inferred from data.

**Modules are what make the species model worth its energy.** Fire on *all* 15
classes and the behaviour becomes identical to firing whenever the gate says
`non-empty` — the species model would contribute nothing while costing ~25 ms.
That is a motion trigger with extra steps: exactly the thing the photographer
already owns and dislikes. The species model only earns its energy when the device
fires on a **subset**. Selectivity is the product.

So the number of enabled modules is a dial:

| Enabled | Effect |
|---|---|
| 0 | Never fires. Useless. |
| 1 | Maximum selectivity, best battery. The sweet spot. |
| ~all 15 | Species model is dead weight; the gate alone is equivalent and 25 ms cheaper. |

There is therefore a **measurable break-even**: beyond how many enabled classes
does the species model stop paying for itself? Computable offline from the
confusion matrix — see §9.

**Multiple modules are free in compute.** Because the model is multi-class, three
enabled modules mean checking three outputs of the *same* forward pass: one
inference, same latency, same energy.

**This is the argument that settles multi-class vs per-species binary models.**
Per-species binary models would be more accurate individually, but three enabled
modules would then mean **three inferences** — 3× latency and energy. Multi-module
support and per-species binary models are incompatible. We keep multi-class.

**What multiple modules do cost: the false-fire budget adds up.** With softmax and
thresholds above 0.5, at most one class can exceed its threshold, so the per-class
false-fire rates are roughly **additive**: three modules at ~5% each ≈ ~15%
combined. More modules → more shots → shorter battery and a fuller card. This is a
real trade-off the photographer must see, not discover in the field (§9, §13).

**Known limitation — softmax couples the classes.** If a coyote and a bobcat are in
frame together, softmax forces a single winner; multi-label sigmoid would handle
both. CCT-20 labels one class per image, so the dataset does not even represent
this case — there is nothing to train or measure it on. Softmax is correct for the
MVP; sigmoid is a Phase-1 upgrade once real multi-animal data exists.

**Falls out for free:** a *negative* module ("shoot everything except raccoons") is
the same mechanism inverted.

**Why the abstraction survives Phase 1.** Within one region a module is config on a
shared model. **Across regions it must be a different model** — our CCT-20 model
knows 15 North American species and has never seen an elephant; a photographer in
Tanzania needs different weights entirely. One model cannot cover the world's
species. The module is the primitive that hides that difference from the user, and
that is the real reason it exists (§13).

**The cascade's honest trade-off.** It optimizes **energy**, not shutter lag. On a
real detection we pay gate + species (~30 ms) — 5 ms *worse* than species alone.
What we buy is that 70% of wake-ups cost 5 ms instead of 25 ms:

| | Average per trigger (70% empty) |
|---|---|
| Single model, no gate | ~25 ms |
| Cascade | 0.7 × 5 + 0.3 × 30 = **~12.5 ms** |

≈ **2× less energy per trigger, for +5 ms of shutter lag on real detections.** This
is our own optimization, and it is measurable — it becomes an ablation, not a
claim.

## 4. Why not a detector on-device — the trade-off, quantified

The literature is clear that cropping to the animal helps a lot: on CCT-20,
classifying **bounding-box crops instead of full frames roughly halves the error**
(19.06% → 8.14% on cis, 41.04% → 19.56% on trans; Beery et al. 2018, Table 1,
**published**). So an obvious design is: MegaDetector-compact → crop → classify.

We rejected it, on numbers.

**The trap: parameters lie, MACs do not.** MegaDetector V6-compact (~2.3M params)
looks *smaller* than MobileNetV2 (~3.4M params). But cost is architecture ×
**input resolution**, and detectors need high resolution precisely to see the small
animals they are hired to find:

| Model | Params | Input | ~Compute |
|---|---|---|---|
| MobileNetV2 | 3.4M | 224² | **~0.3 GMAC** |
| MegaDetector V6-compact (YOLOv10-class) | 2.3M | 640² | **~3.4 GMAC** |

The "smaller" detector costs roughly **10× more** than the "bigger" classifier.

**A vs B** (estimates, 70% empty):

| | **A — gate → MNv2 full frame** | **B — MD-compact → crop → MNv2** |
|---|---|---|
| Gate step | ~5 ms | ~200–400 ms @640² (~60–100 @320²) |
| Species step | ~25 ms | ~25 ms (on crop) |
| **Average per trigger** | **~12.5 ms** | **~250+ ms** |
| **Shutter lag on a detection** | **~30 ms** | **~275 ms** |
| **Energy** | **1×** | **~20× worse** |
| Accuracy | baseline | ~2× lower error |

B pays its most expensive step on **100% of frames — including the 70% where it
finds nothing**. It burns the costliest computation exactly where it is useless,
and it breaks both of the product's promises (months of autonomy, fast shutter) to
buy accuracy.

**And the accuracy gain is overstated for us:** the ~2× figure is for the
**15-class** problem. Our operating question is much closer to binary
("bobcat / not bobcat"), which is easier — so the real crop benefit will be
**smaller than 2×**. We would be paying ~20× the energy for a fraction of a
doubling.

**B is not discarded — it becomes a measured ceiling.** We evaluate the crop
pipeline **offline** on the same test set, and report:

> "A detector cascade would cut our error by X pp. We declined it: it costs ~20×
> the energy and ~9× the shutter lag, which kills both core promises. Here is the
> price, and here is what we bought with it."

That is the critical evaluation the rubric asks for, with numbers instead of
opinions. We do not need to run B on the Pi to say it — accuracy offline, cost
from MACs plus one confirming measurement.

## 5. Crop-teacher + distillation — the crop, without paying for it

The key realisation: **crops are needed for *training*, not for *inference*.**

CCT-20 ships **ground-truth bounding boxes** (~66k over CCT; MTurk, 3–10
annotators, PascalVOC IoU ≥ 0.5). So:

```
OFFLINE (gx10, compute is free):
    teacher trains on GT-box CROPS  →  high accuracy
          │  knowledge distillation
          ▼
ON-DEVICE (Pi, every millijoule counts):
    student runs on the FULL FRAME, but inherits the teacher's knowledge
```

The student learns to attend where the teacher looked, **without ever running a
detector**. This is exactly what KD is for, and the camera-trap literature names
it directly: *crop-based distillation — teacher box → student learns attention*.

This makes distillation **load-bearing** rather than cosmetic, and it yields a
clean ablation:

> full-frame student **without** KD **vs** the same student **with** KD from a
> crop-trained teacher.

If the second wins, we have shown that distillation transfers *attention*, not
just soft-target smoothing — a stronger result than HW3's modest +0.1–0.2 pp.

**Teacher, staged** (unchanged decision):

1. **Our own FP32 crop-teacher** (ResNet50 / EfficientNet-B0 on GT crops) —
   control, fully our own work, reuses the HW3 pipeline.
2. **MegaDetector V6-compact** — upgrade. Trained on millions of camera-trap
   images across hundreds of locations, so it generalizes to unseen locations,
   which is exactly our weak spot. Also a data cleaner / pseudo-labeler.
3. **Compare them.** "Does an externally-pretrained teacher beat a self-teacher
   under domain shift?" is itself a result.

**Action item:** verify the license of the exact MegaDetector weights used — V6
variants ship under mixed MIT / Apache-2.0 / **AGPL** terms depending on the YOLO
backbone. Fine for coursework; must be checked and stated.

## 6. Data — CCT-20

**Caltech Camera Traps-20** (Beery et al., ECCV 2018), the official benchmark
subset of CCT. **57,868 images, 20 camera locations, 15 classes + empty.**
CDLA-permissive. ~6 GB (downsampled to max 1024 px/side).

**Class frequency, most → least** (Fig. 4 legend): raccoon, rabbit, coyote,
**bobcat**, cat, empty, squirrel, dog, **car**, bird, skunk, rodent, deer, badger,
fox.

**Target species: bobcat** — the 4th most common class, so there is enough data to
chase high recall, and a photographer staking out a bobcat is a real user. (Note:
CCT-20 does contain a `car` class, though rare — the illegal-logging branch is not
as far away as it looked.)

**Splits — cis and trans, and this is a gift.** From the 20 locations: 9 random →
trans-test, 1 → trans-val; the remaining 10 are cis. Within cis: odd days →
cis-test; 5% of even days → cis-val; the rest → train, with train and val never
sharing a sequence.

| Split | Images | Meaning |
|---|---:|---|
| train | **13,553** | 10 locations |
| cis-val | 3,484 | same locations as train |
| cis-test | 15,827 | **same locations** as train |
| trans-val | 1,725 | 1 unseen location |
| trans-test | 23,275 | **9 unseen locations** |

This gives us two test sets that answer two different product questions:

- **cis** = "does it work at the spot I trained it on?"
- **trans** = **"will it still work when I move my trap somewhere new?"** — the
  question every photographer actually has. We can answer it with a number.

**Never a random split.** Camera traps have fixed backgrounds; a random split lets
train and test share the background, so the model learns the camera, not the
animal. This is the domain-specific form of our no-leakage rule, and CCT-20 has it
built in.

**Published baselines to compare against** (Inception-v3 @299², full image,
**published**): **20.83% top-1 error on cis vs 41.08% on trans** — the error
roughly **doubles** on unseen locations (+97%). Our domain-shift concern is not a
hypothesis; it is a measured property of this data.

**Data character to design for:** sequences are 1–5 frames at ~1 fps; false
triggers come from **wind and heat rising off the ground**; empty frames also
occur when the animal leaves the frame mid-sequence. Night frames are IR
grayscale. Animals are often small, blurred, occluded, or partially out of frame.

**Label quality — why CCT and not Snapshot Serengeti.** Serengeti is tempting
(7.1M images, 61 species including elephant and lion, 225 camera sites). But LILA
states its annotations *"are only reliable at the sequence level"*: volunteers
labelled a 3-frame burst as "elephant" when the elephant may be in only one frame.
Our device classifies **one frame**, so both our training labels and — fatally —
our **test** labels would be noisy, making our headline metric unmeasurable. CCT's
labels are **per-frame** (biologists tracked motion across the sequence to label
each frame). Rigor wins. Also: elephant and lion are Serengeti's long tail
(wildebeest/zebra/gazelle dominate), and seasons are 25–636 GB.

**Stretch — the elephant module.** Train `elephant-v1` on a Serengeti sample and
run it through the same application to **demonstrate the module mechanism** (same
device, swap the module, now it shoots elephants). Headline metrics stay on
CCT-20, with the sequence-label caveat stated on the slide. We get the
demonstration without paying for it in credibility.

## 7. Model & compression

**Student: MobileNetV2, pretrained on ImageNet, transfer-learned.**

This reverses the earlier TinyCNN decision, and the reason matters: TinyCNN existed
*only* because depthwise-separable convolution is hard to make fast in a
hand-written engine. With that engine gone, the constraint is gone — and
ORT/XNNPACK has excellent depthwise kernels, so MobileNetV2 is precisely what the
runtime is fast at. Decisive on top of that: **the training set is only 13,553
images.** Training from scratch on that is hopeless; ImageNet-pretrained transfer
learning is not optional here. Augmentation matters (brightness/contrast,
grayscale/IR simulation, blur, random crop).

**Rejected:** MobileNetV3 — hard-swish breaks post-training quantization (use only
with QAT, if at all).

**Compression ladder** — this is the core ablation; every row gets the §9 metric
set on **both** cis and trans:

| # | Model | Purpose |
|---|---|---|
| 0 | FP32 crop-teacher | Accuracy ceiling |
| 1 | FP32 student, full frame, no KD | Uncompressed baseline |
| 2 | FP32 student **+ crop-teacher KD** | Does KD transfer attention? (§5) |
| 3 | INT8 PTQ | Cheap quantization |
| 4 | INT8 QAT | QAT > PTQ |
| 5 | Structured-pruned + QAT | Real MAC reduction |
| 6 | **KD + pruned + QAT** | The full stack — expected best |
| — | Cascade **off** (single model) | Cascade ablation (§3) |
| — | Detector cascade (B), offline | The declined ceiling (§4) |
| — | Unstructured-pruned | **Honest negative result** |

Notes:

- **Structured channel pruning**, not unstructured — unstructured shrinks the file
  but does not speed up ORT's dense kernels. We keep it in the table to *show*
  that; a measured negative result beats silence. Caveat: pruning
  depthwise-separable nets with residuals is fiddlier than pruning VGG, and the
  gain on an already-efficient MobileNetV2 may be modest. We report what we find.
- **QAT over PTQ**, with a calibration set that covers **night/IR and small
  animals** — a PTQ scale fitted on daylight frames will not cover IR.
- Compression is judged on **recall at the operating point**, never on top-1.

**What transfers from HW1–HW4:**

| Asset | Status |
|---|---|
| `hw3/src/distill.py` — `DistillLoss`, `kd_train` | **Direct reuse**, and now load-bearing (§5) |
| `hw1/src/structured.py` — sensitivity-guided channel pruning (torch-pruning) | **Reuse**; example input 32×32 → 224² |
| `hw2`/`hw3` `src/qat.py` — QAT loop, already accepts `teacher`/`distill` | **Loop reused**; quantizer replaced |
| `hw1/src/prune.py` — `FineGrainedPruner` (unstructured) | The negative-result row |
| `hw2/src/kmeans_quant.py` — K-Means centroid quantization | **Does not transfer.** It is weight-sharing: compresses *storage*, not arithmetic. ORT needs affine INT8. |
| HW4 NAS supernet | Not used. |

**Genuinely new work:** affine INT8 (`torch.ao` QAT → ONNX QDQ export → ORT), the
crop-teacher/KD setup, and the C++ application.

## 8. The C++ application — our deliverable

We are **not** writing an inference engine. ONNX Runtime is a proven, heavily
optimized runtime (MLAS/XNNPACK, hand-tuned NEON), and a student engine would lose
to it on speed while costing weeks. The brief explicitly lists ONNX as a
legitimate inference-level optimization.

What we write is **the device's control application** — and this is where the C++
points are earned. The course container (`Docker_VSCode/`) is a **145-line
smoke-test**: one inference, one image, one thread hard-coded, **no timing code at
all**. It is a toolchain proof, not a starting template.

Our C++ application:

1. **The cascade and decision logic** — gate → threshold → species → threshold →
   shutter decision → sleep. This is our design, in code.
2. **Fused preprocessing.** The container does `convertTo` → `cv::subtract` →
   `cv::divide` → `cv::split` — **four separate passes over memory**, plus copies.
   We fuse to a single pass and remove copies. The literature notes preprocessing
   is often a large share of end-to-end latency; this is our low-level code with a
   measurable win.
3. **Benchmark harness** — p50/p95/p99, thread sweep, thermal logging. None of
   this exists in the container.
4. **Profiler-driven bottleneck removal** — ORT profiler → slowest ops → fix
   (fusion, layout, threading, selective quantization).
5. **Dataset runner** — iterate the saved test set, emit decisions + latencies +
   the confusion matrix.

Honest position: this is a weaker claim on the 15 C++ points than a from-scratch
engine would be, and we compensate with the four items above. Optional upside kept
in the back pocket: hand-write the *gate* engine only (it is tiny, and runs on
100% of frames) while ORT serves the species model.

**Demo:**

```
$ ./trigger --module bobcat-v1 --data cct20/trans-test/
frame_00412.jpg  gate=non-empty(0.98)  species=bobcat(0.94)   → SHUTTER FIRE  e2e=31ms
frame_00413.jpg  gate=empty(0.02)      → sleep                                e2e=6ms
...
summary: fired 412/438 target frames (recall 94.1%), 17 false fires
         p50 6ms / p95 33ms   |  71% of frames exited at the gate
```

## 9. Metrics

Accuracy is a trap: ~70% of frames are empty, so "always say empty" scores 70%.

**The error costs are asymmetric, and they set the metric:**

- **Missing a bobcat = a missed shot = the product failed at its one job.** This
  is what the photographer paid to prevent. Expensive.
- **A false fire = one wasted frame + a little battery.** Cheap.

So this is a **recall-first** product with a **precision budget** (fire on every
raccoon all night and the battery dies and the card fills — which destroys the
value a different way).

| Metric | Why |
|---|---|
| **Recall on the target class (primary)** | The product's one job |
| Precision on the target class | The battery/card budget |
| Recall @ the chosen operating point, **cis vs trans** | "Does it survive a new location?" |
| Gate recall on non-empty | The gate must never cause a miss |
| % of frames exiting at the gate | The cascade's energy win |
| Per-class recall, macro F1 | Long-tailed classes |
| Day vs night-IR recall, separately | Different failure mode |
| **p50 / p95 / p99 shutter lag** | p95 is the product spec, not the mean |
| Average latency per trigger | Energy proxy → battery life |
| Model size (FP32 vs INT8) | Compression |
| **Fire rate & battery life vs number of enabled modules** | The multi-module trade-off (§3) |
| **Species-model break-even point** | Beyond how many enabled classes is the gate alone equivalent? (§3) |

**Two results that cost us nothing.** Both fall out of the same confusion matrix,
offline, with no extra training and no extra measurement:

1. **The multi-module curve** — false-fire rate, shots/night and estimated battery
   life as a function of how many modules are enabled. Example shape:

   > 1 module: 94% recall, 4% false fires, ~200 shots/night.
   > 3 modules: ~92% mean recall, ~13% false fires, ~600 shots/night, ≈½ the
   > autonomy.

2. **The break-even** — the number of enabled classes at which firing on the
   species model's subset stops beating firing on the gate alone. Past that point
   the species model is 25 ms of dead weight, and we can say so with a number.

**The headline slide** is not "we hit N FPS". It is:

> "At 94% bobcat recall on **unseen locations**, the device fires in 31 ms (p95
> 33 ms) and averages 12.5 ms per PIR trigger — ≈2× less energy per wake-up than
> a single-model design, at +5 ms of shutter lag."

## 10. Benchmark protocol

On the rented Raspberry Pi 5 (Hostpro), CPU-only (BCM2712, 4× Cortex-A76 @
2.4 GHz).

1. Report the CPU governor; use `performance` if permitted.
2. Warm-up 50–100 inferences; measure ≥1000; batch size 1.
3. Report **p50/p95/p99**, not the mean.
4. Time **separately**: JPEG decode → resize → normalize / gate / species /
   end-to-end. Preprocessing is easy to hide and often dominates.
5. Thread sweep: 1 / 2 / 4.
6. Thermals throughout: `vcgencmd measure_temp`, `vcgencmd get_throttled`. The Pi
   5 throttles from ~80 °C — the first 30 s are not representative.
7. Identical images for every configuration.
8. Separate runs: cis-test / trans-test / day / night-IR.

**Stated limitation:** the Pi is a datacenter instance. No physical access →
**no inline power meter**, no camera, no GPIO. Energy per inference is **proxied
by latency**, not measured. This goes on the limitations slide.

## 11. Risks & open questions

| # | Item | Mitigation / decision rule |
|---|---|---|
| 1 | **13,553 training images is small** | ImageNet-pretrained transfer learning + aggressive augmentation. If trans-location recall collapses, consider training on the larger CCT split (106,428 / 65 locations) and keeping CCT-20 only for eval — at the cost of benchmark comparability. |
| 2 | Trans-location recall may be poor (published error doubles) | This is the honest finding if it happens — report it. The crop-teacher KD (§5) is our main lever against it. |
| 3 | Structured pruning on MobileNetV2 may give little | Report what we find; the ladder is designed to show it either way |
| 4 | MegaDetector weight license (AGPL variants) | Check before use; state explicitly |
| 5 | Does the Hostpro Pi allow `vcgencmd` / governor control? | Verify on day 1 of the trial; if not, report what is available |
| 6 | 15 C++ points are weaker without an own engine | §8 items 1–5; gate-engine option in reserve |
| 7 | Latency estimates in this doc are estimates | Every one is replaced by a measurement before it enters the report |

## 12. Schedule — driven by the 5-day free trial

Hostpro's Pi 5 offers a 5-day free trial. Everything that can happen off-device
**must** happen before the window opens.

| Phase | Where | Work |
|---|---|---|
| 1. Data | gx10 | CCT-20 (6 GB), cis/trans splits, day/night + small-animal subsets, crop extraction from GT boxes |
| 2. Models | gx10 | Crop-teacher, MNv2 student, KD, gate model, structured pruning, INT8 QAT, ONNX QDQ export |
| 3. Application | local ARM64 devcontainer | C++ cascade app, fused preprocessing, benchmark harness |
| 4. Dry run | local ARM64 devcontainer | Full protocol end-to-end, so nothing is authored during the trial |
| **5. Trial** | **rented Pi 5** | **Final measurements only**, with slack for a repeat |
| 6. Write-up | — | Slides, analysis, limitations |

The container's ARM64 devcontainer is what makes this schedule safe: the C++ is
built and debugged without any Pi.

## 13. Roadmap beyond the course

Phase 0 proves the decision core. The device grows outward from there, and the
same cascade is the reusable brick for every mission.

```
Trigger (PIR / radar / audio)
  → [OUR GATE: is anything there?]
  → [OUR CLASSIFIER: what is it?]
  → decision: shutter / alert / discard
```

| Phase | Content |
|---|---|
| **Phase 0 — this course** | Pi 5 CPU-only decision core, ORT, cascade, compression stack, measured on CCT-20 |
| **Phase 1 — MVP device** | Pi 5 + AI HAT+ (Hailo-8L) + NoIR camera + PIR + real shutter GPIO + LTE; **region model packs** (Africa/Europe/N. America — one model cannot cover the world's species); multi-label sigmoid for multi-animal frames; **fire budget** (photographer sets "max N shots/night", the device solves thresholds jointly across enabled modules); audio (chainsaw/gunshot); collect real data |
| **Phase 2 — field prototype** | CM5 + Hailo-8L + STM32 supervisor (sleep/wake, power gating), eMMC, IP67, solar; OTA modules |
| **Phase 3 — low-power product** | STM32N6 / Himax WE2 + Syntiant always-on audio, LTE-M / LoRa, months of autonomy, low BOM |

**Where the market goes.** The photographer's device and the conservation sensor
are the *same box with different modules*:

| Module | Mission | Buyer |
|---|---|---|
| `bobcat-v1`, `elephant-v1` | Target-species photography | Wildlife photographers |
| `wildlife-filter-v1` | Empty-frame filtering, monitoring | Ecologists, researchers |
| `antipoaching-v1` (`person`) | Intrusion alert | Reserves, NGOs, rangers |
| `forestguard-v1` (`car` + chainsaw audio) | Illegal logging, timber trucks | Forest services |

CCT-20 already contains a `car` class, so the logging branch is closer than it
looks. The commercial reference point is Nightjar/TrailGuard (Movidius Myriad X,
PIR, IR camera, IP67, alerts <30 s, 18-month battery, from $799).

**Local relevance:** illegal logging in the Carpathians and human-wildlife
conflict (bear/wolf/boar near farms) are real, documented Ukrainian problems — a
concrete addressee rather than an abstract Serengeti.

## 14. References

**Method & data**

- Beery, Van Horn, Perona, *Recognition in Terra Incognita*, ECCV 2018 —
  https://arxiv.org/abs/1807.04975 — CCT-20, cis/trans splits, the cis→trans error
  doubling, crop-vs-full-frame Table 1
- Caltech Camera Traps (LILA) — https://lila.science/datasets/caltech-camera-traps/
- CCT dataset page / splits — https://beerys.github.io/CaltechCameraTraps/
- Snapshot Serengeti (LILA) — https://lila.science/datasets/snapshot-serengeti/ —
  incl. the sequence-level label caveat
- Cunha et al., *Filtering Empty Camera Trap Images in Embedded Systems*, CVPRW
  2021 — https://arxiv.org/abs/2104.08859
- Vélez et al., *Choosing an Appropriate Platform and Workflow for Processing
  Camera Trap Data using AI* — https://arxiv.org/abs/2202.02283

**Models**

- MegaDetector — https://github.com/microsoft/megadetector ·
  Model Zoo — https://github.com/microsoft/MegaDetector/blob/main/docs/model_zoo.md
- PyTorch-Wildlife — https://microsoft.github.io/Pytorch-Wildlife/model_zoo/
- Sandler et al., *MobileNetV2*, CVPR 2018 — https://arxiv.org/abs/1801.04381

**Hardware & benchmarking**

- Raspberry Pi 5 — https://www.raspberrypi.com/products/raspberry-pi-5/
- Pi 5 heating and cooling — https://www.raspberrypi.com/news/heating-and-cooling-raspberry-pi-5/
- *Impact of Thermal Throttling on Long-Term Visual Inference* —
  https://arxiv.org/abs/2010.06291

**Product reference**

- Nightjar — https://www.nightjar.tech/
- Dertien et al., BioScience 2023 (TrailGuard deployment) —
  https://academic.oup.com/bioscience/article/73/10/748/7261057
</content>
