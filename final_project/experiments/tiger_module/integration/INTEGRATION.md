# Integration — the tiger head embedded in the deployed M2 graph

> How the few-shot tiger module is actually wired into the shipping model, with the
> parity and on-Pi latency evidence. Additive and branch-only: the released `M2.onnx`,
> the bundle, and `v1.0-final` are untouched. Nothing here is merged to `main` until reviewed.

## Design

The deployed classifier is a `Gemm(1280 → 16)` on a pooled embedding. Adding "tiger" is one
extra row on that same embedding, run **in parallel** to the INT8 16-class path so the latter stays
bit-identical. We bake it directly into the ONNX graph:

```
input → … MobileNetV2 INT8 backbone … → GlobalAveragePool → (pool_quant) → /Flatten_output_0  ─┬─→ [INT8 classifier] → logits[16]   (UNCHANGED)
                                                                     [1×1280 embedding]         └─→ ReduceL2 → Div → Gemm(w,b) → tiger_score[1]   (NEW, FP32)
```

`tiger_score = w · L2norm(embedding) + b`. Decision: `tiger_score > threshold` → shutter for tiger.
The weights `w[1280]`, bias `b`, and `threshold` are the A2 linear head, recomputed in **M2's own
embedding space** (see `tiger_head_M2.json`). A new animal is another such row — no backbone retrain.

**For the "configurable to any animal" product, see [`registry/REGISTRY.md`](registry/REGISTRY.md)** —
the general form: one exposed embedding + a JSON registry of any number of targets, scored in C++,
verified multi-target on the real Pi. The tiger bake below is the single-target proof of the path.

Two graph variants are produced:
- **`M2_plus_lean.onnx`** — outputs `logits` + `tiger_score` (embedding stays internal). **This is the
  one to deploy**: ORT still fuses the classifier tail, so latency is indistinguishable from base M2.
- **`M2_plus.onnx`** — also exposes `/Flatten_output_0` (the 1280-d embedding) as an output, for the
  flexible "add animals via a host-side registry" design. Costs ~0.35 ms more (forcing the embedding
  as an output blocks tail fusion).

## Evidence

### Head quality in M2 space (T1)
Rebuilt on M2's INT8 embedding (tapped at `/Flatten_output_0`), ATRW tiger vs CCT background,
held-out test: **ROC-AUC 0.9999, F2 0.992, recall 1.000 at the 5 % false-fire budget**
(prec 0.865). Matches/beats the M0-space experiment — the deployment space is fine.

### Graph correctness (T2), verified in Python ORT
- **`logits` bit-identical** to the released `M2.onnx` (`max|Δ| = 0.000e+00`) → every Core parity /
  frozen-test / Pi-benchmark result still holds unchanged.
- In-graph `tiger_score` matches the sklearn head's `decision_function` to **3.96e-6**.

### Python ↔ C++ ↔ Pi parity (T3)
Identical golden input tensors fed to Python ORT (gx10), C++ `tiger_probe` (gx10), and C++
`tiger_probe` on the **real Pi CM5** — all agree to the printed precision:

| Sample | argmax (16-cls) | tiger_score | trigger (thr −1.408) |
|---|---:|---:|---|
| tiger (ATRW) | 11 (`empty`) | **+1.9013987** | **yes** |
| CCT empty | 7 | −5.4706335 | no |
| CCT bobcat | 7 | −5.1206145 | no |

The 16-class head is blind to tiger (it can only pick a CCT class — here `empty`); the **side-head
catches it**, and correctly stays quiet on a real CCT cat (bobcat). C++ (Pi) == C++ (gx10) ==
Python, bit-for-bit.

### On-Pi latency (real CM5, governor `performance`, threads=1, ORT infer only)

Per-inference time over several reps (800–2000 iters each). Run-to-run noise on the 4-core Pi is
**±~2 ms**, which is larger than any difference between the variants:

| Model | Outputs | Pi infer (ms, observed range) |
|---|---|---|
| `M2` (base) | `logits` | 14.0 – 15.7 |
| `M2_plus_lean` | `logits` + `tiger_score` | 14.1 – 16.1 |
| `M2_plus` | `logits` + `tiger_score` + embedding | 14.3 – 15.9 |

- The three are **statistically indistinguishable** at this precision — the tiger head adds **no
  measurable end-to-end latency**. This agrees with the standalone microbenchmark (`bench_head.c`:
  the head math is ~1–2 µs, i.e. ~0.01 % of a 14 ms inference).
- `logits` are **bit-identical** for both variants vs the released `M2` (`max|Δ| = 0.0`), so the
  16-class decisions — and every Core result — are unchanged.
- Prefer `M2_plus_lean` for deployment (embedding internal → ORT keeps the tail fused).

## Reproduce

```sh
# on gx10 (venv: torch, onnx, onnxruntime, sklearn, opencv-python-headless)
python build_tiger_head_m2.py       # taps M2 embedding, fits A2 head → tiger_head_M2.json
python build_m2_plus_onnx.py        # M2_plus.onnx (embedding + tiger_score) + Py verify (C1/C2)
python build_m2_plus_lean_onnx.py   # M2_plus_lean.onnx (tiger_score only)
python make_golden_and_reference.py # golden_*.bin + golden_ref.json (Python ORT reference)

# C++ probe (ORT 1.27.0 headers+lib in ort1270/)
g++ -O2 -std=c++17 -mcpu=cortex-a76 -I ort1270/include tiger_probe.cpp \
    -L ort1270/lib -lonnxruntime -o tiger_probe
# gx10 or Pi (LD_LIBRARY_PATH → the 1.27.0 ORT):
./tiger_probe --model M2_plus_lean.onnx --input-bin golden_tiger.bin
./tiger_probe --model M2_plus_lean.onnx --input-bin golden_tiger.bin --time --iterations 2000
```

Model files (`*.onnx`, `*.bin`) are gitignored and live on gx10 under `~/efficientml/tiger_embed/`
(+ the Pi at `/tmp`). Committed here: the scripts, `tiger_probe.cpp`, the deployable head
(`tiger_head_M2.json`), and the golden reference (`golden_ref.json`).

## Honest notes

- **Same easy-target caveat as the experiment**: tiger separates cleanly; a subtle animal would score
  lower. The mechanism (bit-identical core + a cheap side-head) is what generalises.
- The side-head is **FP32** (reads the pooled embedding); this is exact and the latency cost is nil.
  A fully-INT8 baked head is possible but unnecessary here.
- Full production would also add: a target **registry/policy** entry (name + threshold) beside the
  Core's threshold catalogue, and the C++ **app** (`cpp/src/session.cpp`) reading `tiger_score` to
  emit the shutter — this branch proves the graph + parity + latency; wiring the app is the next step.
