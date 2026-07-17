# D5 — M4 registration (committed before any M4 number exists)

M4 = the validated QAT recipe applied to the selected M3 pruned checkpoint
(DESIGN §8.4). Registered before the first M4 metric, in the D1–D4 pattern.

## 1. What M4 is, and what it is not

- **Init: the M3 c30 checkpoint** (`d4_m3_c30_20260717T052416Z/best.pt`, sha
  `89bbb9e7…`), hash-verified — never M0, never M1/M2. The pruned architecture
  is rebuilt with `prune.apply_widths` from the widths recorded in that
  checkpoint, then the c30 weights are loaded, then QAT structure is inserted.
- **Recipe: exactly M2's validated arm**, no new search. M2's LR sweep already
  chose 5e-5 as the recovery-maximising rate; DESIGN §8.4 says "apply the
  validated QAT procedure", so M4 runs **one arm at lr 5e-5**, 6 epochs,
  batch 64, AdamW wd 1e-4, AMP off, observers frozen after epoch 1, seed 42 —
  the frozen M2 recipe verbatim (`results/optimize/m2_qat/preregistration.md`).
  No LR search: re-searching would be a new experiment, not "the validated
  procedure".
- **Observer calibration: the frozen D1 manifest** (`9fab904c…`), shared with
  M1/M2 by registration — the same corpus decides the activation scales.
- Best epoch by the frozen §7.2 rule under active fake-quant, exactly as M2.

## 2. Export, evaluation, calibration

The unchanged D2 export path: output-side QDQ, ReLU6 absorbed with the
exactness proof on real ranges, per-channel S8 weights / per-tensor S8
activations, scalar-QDQ fix, and the **weight fold** into INT8 initializers
(the bitwise-proven fold D2 established — M4 must be a real INT8 graph, not a
float graph carrying rounded weights). Then `evaluate_onnx` through deployment
ORT (CPU EP, batch 1), `calibrate_candidate` under frozen §6.3 →
`bobcat_m4_qat_lr5e-5_v1`.

## 3. The comparison row and MAC/param counting

M4's architecture is c30's — QAT changes representation, not shape — so its
params and MACs **equal M3's** and are taken from the committed M3 row
(`comparison.py --base-model-id M3`), exactly as M1/M2 copied M0's. The row's
`kind` is `pruned_qat`; its bytes and validation come from M4's own artifact
and evaluation. Only the selected, gated M4 enters `comparison.jsonl`.

## 4. Parity gates

P3 (quantized variant — M4 is INT8, so check 1 is the integer-execution
coverage gate, *not* the FP32 physical-shape gate) and P4, the registered
corpus gates unchanged. No candidate reaching a failed gate deploys.

## 5. The verdict rule (DESIGN §8.4, registered)

**M4 does not win by default.** After gating, M4 is compared to M2 on the
validation Pareto axes (bobcat F2 / MACs / size):

- if M4 is dominated by M2 (M2's primary ≥ and MACs ≤ and bytes ≤, one
  strict), **M2 remains the optimized front-runner** and M4 is recorded as a
  measured negative — a more complicated stack that did not earn its
  complexity;
- if M4 is non-dominated, it joins the D6 shortlist beside M2, and Pi latency
  (D6/F-phase) settles which ships.

The registration commits to *recording whichever happens*; "combined is
Pareto-superior" is a hypothesis, not a plan.

## 6. What this registration forbids

Searching any LR other than the validated 5e-5; initializing M4 from anything
but the c30 checkpoint; changing the QAT recipe from M2's; selecting M4 over
M2 on anything but §5; reading test data anywhere in D5.
