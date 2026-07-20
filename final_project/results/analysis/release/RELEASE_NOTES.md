# Wildlife Trigger — v1.0-final

On-device bobcat shutter trigger: MobileNetV2 quantized to INT8, deployed in C++ (ONNX Runtime
CPU EP) on a Raspberry Pi CM5. Efficient ML final project, SET University.

## Headline (real Raspberry Pi CM5, Cortex-A76 @ 2.4 GHz, Ubuntu 24.04)

**Final model M2 (INT8 QAT) vs the FP32 baseline M0, frozen config threads=1:**

- **2.27× faster** end-to-end — 49.06 → 21.61 ms p50 (20.4 → 46.3 FPS)
- **3.5× smaller** — 8.95 → 2.54 MB
- **Accuracy-equivalent** — better in-distribution (cis-test bobcat event-capture 0.858 vs 0.767 at equal 5.5 % false-fire)
- **Pi ↔ gx10 decisions bit-identical**; F4/F5 reproducible within ±3.5 %

Trans-domain (new locations) recall stays low (~35–40 % capture) — the honest, reported
limitation; both models are `recall_floor_infeasible` (the 90 % recall floor is unreachable inside
the 5 % false-fire budget).

## Assets

| File | What |
|---|---|
| `wildlife-trigger-deployment-bundle-v1.0-final.tar.gz` | Self-contained Pi bundle — C++ binary (`-mcpu=cortex-a76`), pinned ONNX Runtime, M0/M2/M4 ONNX + policies + class map, `install.sh`/`run_demo.sh`/`run_benchmark.sh`, `BUNDLE.json`, `MANIFEST.sha256` |
| `SHA256SUMS` | Checksums for the tarball + the three standalone ONNX models |

## Install (Raspberry Pi, Ubuntu 24.04)

```sh
tar xzf wildlife-trigger-deployment-bundle-v1.0-final.tar.gz
./install.sh        # fail-closed preflight + apt OpenCV 4.6.0 + environment.json
./run_demo.sh M2    # self-test + infer + benchmark + run-dataset
```

## Reproduce

See `README.md` and `SUBMISSION.md`. Every headline number traces to
`results/analysis/canonical_results.json`.

## Source

<https://github.com/imagic9/efficient-ml-set> · report `final_project/report/final_report.pdf` ·
slides `final_project/slides/final_presentation.pptx`.
