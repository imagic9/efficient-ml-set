# Core Definition of Done — DESIGN §19 status (2026-07-20)

Every item verified against committed evidence. **All Data/ML and Deployment/C++ items pass
(Gates A–F). Submission items pass except the two that require the public GitHub Release**
(model release links/checksums, tagged repo), which are staged and gated on Vadym's final review.

## Data and ML

- [x] Data manifests, hashes, distributions, leakage assertions — **Gate B 43/43** (`results/data_audit/gate_b.json`).
- [x] Official cis-val preserved; `cis_val_clean` (3,214 / 144 bobcat) drives all decisions.
- [x] Multi-label train/eval rules + counts tested (7/0/1/61/9) — `tests/python/`, `data/audit`.
- [x] Empty supplement ID/sequence/location-disjoint + reproducible (seed 42) — `cct_empty_train_v1`.
- [x] Supplement downsized ≤1024 px; shortcut probe **0.5775** (≈0.50).
- [x] Every M3/M4 surviving channel a multiple of 8; requested vs realized MAC reduction recorded (30 %).
- [x] Empty-supplement + input-shape controls done before M0 freeze; empty ablation matched on **steps**.
- [x] M0–M4 results exist; M1 (PTQ) preserved as the documented negative result.
- [x] All M0–M4 ONNX at **opset 17**; no opset-9 artifact.
- [x] Thresholds use validation only (C3 / `calibrate`).
- [x] Threshold catalog: **14 animals, 11 numeric thresholds, 3 null** (badger/deer/fox) — `results/f3/threshold_catalog.json`.
- [x] Final model selected on **Pi** validation evidence before test; gx10 latency did not rank — `final_decision.md`.
- [x] Confirmation seeds 17/73 finished for M0 (C5) and final M2 (F3); non-gating — `canonical_results.json`.
- [x] Cis/trans metrics + confidence intervals exist — `calibration.json`, notebook §7.

## Deployment and C++

- [x] Baseline + final ONNX pass preprocessing/model/C++ parity — P1/P2/P3/P4.
- [x] C++ CLI, dataset runner, benchmark harness, self-tests pass — Gates E1–E5.
- [x] Policy/catalog JSON via vendored `nlohmann/json`; no system YAML/JSON dev package.
- [x] Release builds use target-safe flags; ELF/glibc loadability proven — `bundle_audit.json`, E8/F1.
- [x] Final model passes bobcat-only **and** multi-target policy tests, one inference/frame — `bobcat_coyote_v1.json`.
- [x] ARM64 dry run from a clean environment — **Gate E PASSED** (`results/e8/dry_run.json`).
- [x] P0 + E6 pass under `qemu -cpu cortex-a76`; no emulated timing in any table — `qemu_parity.json`.
- [x] Pi baseline + optimized use the same application/protocol — `run_f4_pi_benchmark.sh`.
- [x] Full frozen cis/trans-test on gx10; Pi parity subset **bit-identical** for M0 and winner, score deltas recorded — `results/f4/{frozen_test_*,parity_comparison}.json`.
- [x] Latency, FPS, RSS, CPU, size, thermal recorded with raw evidence — `results/f{1,4,5}/`.

## Submission

- [x] Public repository clean; **tag pending** (`v1.0-final`, gated on review).
- [~] Model release links + checksums — **staged** (`results/analysis/release/SHA256SUMS`); live on Release creation.
- [x] Pi bundle installs + runs from its documented entry point — **F1** on the real CM5.
- [x] README reproduces setup/train/export/build/demo/benchmark — `README.md`, `SUBMISSION.md`.
- [x] Data-audit + results notebooks execute cleanly — `notebooks/0{1,2}_*.ipynb`.
- [x] Final report MD/PDF includes the repo URL — `report/final_report.pdf` §15.
- [x] Final slides PPTX/PDF include the repo URL/QR — `slides/final_presentation.pptx` (S1/S12).
- [x] `SUBMISSION.md` points to every canonical artifact.
- [x] `REPO_URL` placeholders replaced (repo URL live; release URL pending the tag).
- [x] All headline numbers trace to machine-readable raw results — `canonical_results.json`.
- [x] Slides answer what worked / failed / bottlenecks / next steps — S11/S12.

## Gate G

**Passes on all but the public Release** (tag `v1.0-final` + upload assets), deliberately left for
Vadym to create after reviewing the final results. Everything the Release needs — models,
policies, bundle, `SHA256SUMS`, release notes — is staged under `results/analysis/release/`.
