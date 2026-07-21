# The general, registry-driven shutter engine

> This is the **configurable-to-any-animal** form. The tiger bake (`../INTEGRATION.md`) proved the
> path end-to-end for one target; this generalises it: one frozen backbone, one exposed embedding,
> and a **JSON registry** of any number of targets. Adding an animal is a new registry row — **no
> retraining, no recompile, no graph change.**

## How it works

```
frame → MobileNetV2 INT8 backbone → 1280-d embedding (exposed by M2_plus.onnx)
                                          │  (L2-normalise once)
                                          ├─ target[tiger]:   w·e + b > thr ?  → fire
                                          ├─ target[bobcat]:  w·e + b > thr ?  → fire
                                          ├─ target[coyote]:  …
                                          └─ target[…]:       any number of targets
```

The registry (`target_registry.json`) is a list of `{name, weight[1280], bias, threshold}`. The C++
engine (`shutter_registry.cpp`) runs the backbone once per frame, L2-normalises the embedding, and
scores **every** registered target, firing the shutter for any that clear their threshold. Several
targets can be armed at once, or just one — the caller chooses.

**Adding a new animal** = run `build_target_registry.py` with a few example images (it fits one
linear head in the frozen M2 embedding space and appends the row). Nothing else changes — the same
`M2_plus.onnx`, the same binary.

## Demonstration — four animals, one mechanism

The registry was built with **tiger** (exotic, absent from the 16 CCT classes) plus **bobcat /
coyote / raccoon** (native CCT classes, added via the *same* registry path, not the 16-class head):

| Target | Source | held-out ROC-AUC | recall @5% false-fire |
|---|---|---:|---:|
| tiger | ATRW | 0.9999 | 1.000 |
| bobcat | CCT | 0.969 | 0.833 |
| coyote | CCT | 0.978 | 0.891 |
| raccoon | CCT | 0.972 | 0.904 |

### On-device decisions (real Pi CM5) — one image per category

| Image | fired targets |
|---|---|
| tiger | **tiger** |
| bobcat | **bobcat** |
| coyote | bobcat, **coyote** |
| raccoon | bobcat, **raccoon** |
| empty | *(none)* |

The mechanism is general and multi-target: tiger fires cleanly, empty fires nothing, and each target
is independent. **Parity: Python (gx10) == C++ (gx10) == C++ (real Pi) to ~1e-7** on every target
score.

### Honest note on the native heads

The tiger head is near-perfect (a visually distinct animal). The **bobcat** head is the weakest
(AUC 0.969, and it false-fires on the coyote and raccoon demo frames): bobcat is a small, camouflaged
felid — exactly the difficulty the whole project reports. These registry heads are quick **frame-level
linear probes**; the Core's *production* bobcat path is a better-tuned pipeline (calibrated threshold
+ sequence-level event capture). The registry demonstrates the **mechanism** — add any animal
uniformly — not that every animal reaches tiger-level accuracy. Head quality is per-animal and
improvable (more/better examples, or the A3 distillation recipe).

## Files

| File | What |
|---|---|
| `build_target_registry.py` | fit a head per target in M2 space → `target_registry.json` |
| `target_registry.json` | the registry: 4 targets × {weight[1280], bias, threshold, metrics} |
| `shutter_registry.cpp` | C++ engine: embedding → score every target → fire (nlohmann json + ORT) |
| `make_registry_golden.py` | golden input tensors + Python reference decisions |
| `reg_ref.json` | Python reference fire decisions/scores (parity oracle) |

## Reproduce

```sh
# gx10 (reuses M2-space embeddings from ../build_tiger_head_m2.py -> m2_emb.npz)
python build_target_registry.py           # -> target_registry.json
python make_registry_golden.py            # -> reg_*.bin + reg_ref.json

# C++ (ORT 1.27.0 headers/lib + repo's nlohmann/json.hpp)
g++ -O2 -std=c++17 -mcpu=cortex-a76 -I ort1270/include -I <json_dir> \
    shutter_registry.cpp -L ort1270/lib -lonnxruntime -o shutter_registry

# run (gx10 or real Pi, LD_LIBRARY_PATH → 1.27.0 ORT)
./shutter_registry --model M2_plus.onnx --registry target_registry.json --input-bin reg_tiger.bin
```

Model/tensor artifacts (`*.onnx`, `*.bin`) are gitignored; they live on gx10
`~/efficientml/tiger_embed/` and the Pi `/tmp`. Committed here: scripts, the registry, the C++
engine, and the reference.

## Next step for full product integration

Fold this into the shipped `cpp/src/session.cpp`: read the embedding output, load the registry
beside the Core's policy/threshold catalogue, and emit the shutter signal per target. The Core's
16-class path (and bobcat) stays exactly as released; the registry rides alongside on the same
embedding.
