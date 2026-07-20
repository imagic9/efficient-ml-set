# Submission manifest

> The single index a reviewer opens first. Every number below traces to a machine-readable
> result file (`results/analysis/canonical_results.json` and the raw JSON it cites) — nothing is
> reconstructed by hand (DESIGN §9.2). **Final model = M2 (INT8 QAT).**

## Canonical locations

| Item | Location |
|---|---|
| Public repository | <https://github.com/imagic9/efficient-ml-set> |
| Release tag | `v1.0-final` — **pending final review** (see "Release" below) |
| Final commit | tag `v1.0-final` will pin the reviewed commit on `main` |
| Report | `final_project/report/final_report.pdf` (13 pp) |
| Slides | `final_project/slides/final_presentation.pptx` / `.pdf` (12 slides) |
| Results notebook | `final_project/notebooks/02_results_analysis.ipynb` (clean-run) |
| Data-audit notebook | `final_project/notebooks/01_data_audit.ipynb` (clean-run) |
| Canonical results | `final_project/results/analysis/canonical_results.json` |
| Pi deployment bundle | GitHub Release asset — **pending** (staged: `results/e7/bundle/`) |

## Models (frozen, opset 17)

| ID | Description | File | Size | SHA-256 |
|---|---|---|---:|---|
| M0 | FP32 baseline | `models/M0.onnx` | 8.95 MB | `c3102764…25ce5154` |
| **M2** | **selected final** — INT8 QAT | `models/M2.onnx` | 2.54 MB | `499bc3ec…4706d45ecc` |
| M4 | INT8 pruned+QAT (measured, not selected) | `models/M4.onnx` | 2.01 MB | `2c9d53b4…d9bc770da` |

Full digests: `results/model_selection/pre_pi_freeze.json` / `results/f3/freeze_manifest.json`.

## Headline results — real Raspberry Pi CM5 (Cortex-A76 @ 2.4 GHz, Ubuntu 24.04)

| Metric | M0 (FP32) | Final M2 (INT8 QAT) | Source |
|---|---:|---:|---|
| End-to-end p50 latency | 49.06 ms | **21.61 ms** (2.27×) | `results/f4/f4_summary.json` |
| End-to-end p95 latency | 48.70 ms | 21.57 ms | `results/f4/bench_*_rep1.json` |
| End-to-end FPS | 20.4 | **46.3** | `results/f4/f4_summary.json` |
| Peak RSS | 96.5 MB | 89.1 MB | `results/f4/bench_*_rep1.json` |
| Model size | 8.95 MB | **2.54 MB** (3.5×) | `results/analysis/canonical_results.json` |
| cis-test bobcat frame recall | 0.6466 | **0.7202** | `results/f4/frozen_test_*.json` |
| cis-test bobcat event capture | 0.7668 | **0.8575** | `results/f4/frozen_test_*.json` |
| trans-test bobcat frame recall | 0.2314 | 0.1938 | `results/f4/frozen_test_*.json` |
| trans-test bobcat event capture | 0.3952 | 0.3468 | `results/f4/frozen_test_*.json` |
| Pi ↔ gx10 parity | — | **bit-identical** (0 disagreements) | `results/f4/parity_comparison.json` |

Measured (not estimated); Pi latency is a real on-device result (DESIGN §12.4). Both models remain
`recall_floor_infeasible` — the 90 % sequence-balanced recall floor is unreachable inside the 5 %
false-fire budget (a pre-registered outcome, not a satisfied rule).

## Reproduction

```bash
# 1. Python training/export env (gx10, isolated venv + requirements.lock)
scripts/setup_gx10.sh

# 2. Data (CCT-20 + empty supplement) and Gate B audit
python -m wildlife_trigger.data.prepare  --config configs/data/cct20.yaml
python -m wildlife_trigger.data.audit    --manifests-dir data/manifests   # Gate B 43/43

# 3. Baseline, optimization ladder, freeze (M0 -> M1/M2/M3/M4 -> shortlist)
python -m wildlife_trigger.optimize.qat_train --config configs/optimize/m2_qat.yaml --lr 5e-5

# 4. Target-compatible ARM64 container + C++ build + parity + bundle
scripts/build_target_container.sh
scripts/build_bundle.sh                        # stages results/e7/bundle/

# 5. Raspberry Pi trial (from gx10): install, benchmark, frozen test, parity
scp -r results/e7/bundle/ cm5-pi:/opt/bundle && ssh cm5-pi /opt/bundle/install.sh
ssh cm5-pi /opt/bundle/run_f4_pi_benchmark.sh 50 1000 3     # Pi latency (F4)
scripts/run_f4_frozen_test.sh                               # frozen cis/trans test on gx10

# 6. Analysis: canonical table + results notebook + report/slides
python scripts/build_canonical_results.py
jupyter nbconvert --to notebook --execute --inplace notebooks/02_results_analysis.ipynb
```

Full per-phase commands and gate outputs are in `README.md` and `PLAN.md`.

## Assignment mapping (50 pts)

| Rubric area | Points | Evidence |
|---|---:|---|
| Model training & optimization strategy | 15 | `report/final_report.md` §4–5; ladder `results/model_selection/`; §8.4 rule `results/model_selection/final_decision.md` |
| C++ inference implementation | 15 | `cpp/`; parity `results/{parity,e4,e6}/`; bundle `deploy/pi/` + `results/e7/` |
| Benchmarking & metrics | 10 | real Pi `results/f4/f4_summary.json`, reproducibility `results/f5/`, parity `results/f4/parity_comparison.json` |
| Results analysis & presentation | 10 | `slides/final_presentation.pptx`, `notebooks/02_results_analysis.ipynb`, `report/final_report.pdf` |

## Definition of Done (DESIGN §19)

Status recorded in `results/analysis/definition_of_done.md` — Data/ML and Deployment/C++ items all
pass (Gates A–F); Submission items pass except the two that require the public Release
(**model release links/checksums** and **tagged repo**), which are staged and gated on final review.

## Release

The public GitHub Release `v1.0-final` (tag + the M0/M2/M4 ONNX + policies + the C++ deployment
bundle + `SHA256SUMS`) is **prepared but not yet created** — it is deliberately left for
`Vadym` to create after reviewing the final results, so the "final" tag marks the reviewed state.
Once created, this manifest's Release-tag / final-commit / bundle-URL rows and the README
`RELEASE_URL` are filled with the live URLs. Assets + checksums are staged at
`results/analysis/release/`.
