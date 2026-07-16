# D2 pre-registration — M2 INT8 QAT (written before any QAT result was viewed)

Date: 2026-07-16. Commit order is the proof, as in D1: this file lands before
any QAT arm is trained, evaluated or calibrated. PLAN D2, DESIGN §8.2.

## 1. Initialization — registered before anything runs

- M2 initializes from **M0's selected checkpoint** (`best.pt`, sha
  `42079c362013…`, epoch 11 of `c2_m0_fp32_seed42_20260716T061203Z`) — never
  from M1 PTQ (DESIGN §8.2, verbatim). The trainer refuses a checkpoint whose
  bytes do not hash to this value.
- Quantization structure is the P0-pinned contract (pins.env): direct QDQ
  fake-quant, **output-side placement** at every tensor boundary, ReLU6
  absorbed exactly (verified numerically at export, as in the A3 spike);
  weights per-channel symmetric S8, activations per-tensor affine S8.

## 2. The recipe (frozen here)

- **BatchNorm**: folded into convolutions **at initialization**, from the M0
  checkpoint's statistics. This is how §8.2's "freeze BN statistics after the
  initial stabilization epoch" bullet is satisfied — the statistics are frozen
  from step 0 by construction, matching ORT's PTQ graph semantics (the
  deployed INT8 graph has no BN either). Recorded as a registered
  interpretation, not silently assumed.
- **Observers** (activation ranges): initialized by a calibration pass over
  the **frozen D1 manifest** (`calibration_1024.jsonl`, sha `9fab904c…`,
  training data only — reused deliberately so M1 and M2 share their
  calibration corpus); observers keep updating through **epoch 1**
  (stabilization), frozen from epoch 2 on. Fake-quant active from step 0.
- **Data, loss, augmentation**: the frozen §7.2 contract, unchanged — same
  manifests (train + empty supplement), same weighted CE with ignore_index=-1,
  same augmentation, batch 64, workers 8, seed 42.
- **AMP off.** M0 trained under AMP; QAT does not: fake-quant arithmetic in
  fp16 would train scales against arithmetic the deployed INT8 graph does not
  perform. Registered as a deliberate deviation from the M0 recipe.
- **Optimizer**: AdamW, weight_decay 1e-4 (M0's family), cosine annealing over
  the full budget, full model trainable (no head-only phase — the head was
  trained in C2; QAT's job is adapting all weights to quantization noise).
- **Budget**: **6 epochs fixed** (inside DESIGN §8.2's 5-10), no early
  stopping — arms must be step-matched to be comparable.

## 3. The LR search (the only searched axis)

Arms: **1e-5, 3e-5, 5e-5** — three points spanning DESIGN §8.2's documented
range, nothing outside it, chosen now.

- **Within an arm**: best epoch by the frozen §7.2 rule
  (`metrics.is_better_checkpoint`) on per-epoch torch-side validation with
  fake-quant active — the same rule every training run in this project uses.
- **Across arms**: the D1 selection rule, unchanged — coverage gate first
  (`integer_execution == true` on the exported artifact), then the frozen §7.2
  key on each arm's **ORT deployment-regime evaluation**
  (`optimize.evaluate_onnx`), then smaller ONNX bytes. Applied mechanically by
  `optimize.select_ptq` over the same candidate-directory contract.

## 4. Registered reading of the outcome

M2's question (DESIGN §8): *does QAT recover PTQ accuracy while retaining INT8
speed/size?*

- **Recovers**: best arm's ORT primary > M1's **0.3527**.
- **Fully recovers**: best arm's ORT primary ≥ the M0 ORT reference **0.3667**
  (ratio ≥ 1.0).
- If the best arm's primary ≤ M1's, QAT **failed to recover** — recorded as a
  finding (M1 remains the INT8 candidate for D6), not retried with off-range
  learning rates or a changed budget.
- The D1 material-drop lines (0.95x primary, −10% per-domain vs the M0 ORT
  reference) trigger the same debugging obligation, for consistency.

## 5. Policy and gates

The selected arm calibrates its bobcat policy under the frozen §6.3 rule from
its own ORT scores (`calibrate_candidate`), binds by hash, and must pass
P3/P4 (the D1 gates, same tolerances) before entering `comparison.jsonl` as
M2. All three arms' calibrations are committed as evidence either way.

## 6. What this pre-registration does not permit

- No learning rates outside [1e-5, 5e-5]; no budget changes after a result is
  seen; no switching the initialization checkpoint.
- No test labels anywhere in D2.
- Amendments require recording the trigger measurement in this file, per
  standing practice.
