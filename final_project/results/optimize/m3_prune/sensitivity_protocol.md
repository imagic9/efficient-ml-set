# D3 — pruning-sensitivity protocol (registered before any sensitivity number exists)

Committed before the first sensitivity evaluation runs, in the same spirit as
D1/D2's pre-registrations: the measurement rule is fixed first, so the curves
cannot quietly become an argument for whatever they turn out to show.

## 1. What is pruned, and what is never pruned

Per DESIGN §8.3, pruning roots are exactly the **16 expansion `1x1`
convolutions** of MobileNetV2's `t=6` inverted-residual blocks
(`features.2 … features.17`). Each removal is one dependency-solver-verified
group coupling: expansion conv out-channels → expansion BN → depthwise conv
(in = out = groups) → depthwise BN → projection conv in-channels.

Fixed, never mutated: the stem (`features.0`), the whole `t=1` block
(`features.1` — its depthwise width is coupled to the stem), every projection
output / residual width, the final `1x1` (`features.18`, 1280), and the
16-output classifier. The `torch-pruning` group for every root is verified to
touch nothing outside the contract before any weight moves; a violation aborts
the run.

## 2. The measurement

- **Ratios per group:** 0.125, 0.25, 0.375, 0.5 — requested; every surviving
  width is rounded by the solver (`round_to=8`), and the **realized** width,
  ratio, and MAC drop are recorded separately from the request.
- **One group at a time**, from the M0 checkpoint (`42079c36…`), no
  fine-tuning. 16 groups × 4 ratios = 64 evaluations.
- **Importance:** group L1 magnitude (`GroupMagnitudeImportance(p=1)`) — the
  hw1 criterion, group-extended. Deterministic given the checkpoint.
- **Metric:** bobcat frame F2 at the fixed 0.5 yardstick on cis-val-clean and
  trans-val, their mean (the frozen §7.2 primary), and sequence-balanced
  recall recorded alongside. Never accuracy; never test data.
- **Regime:** torch on gx10 GPU with TF32 disabled (the issue-#30 floor),
  batch from the M0 run's own config. These numbers are a **relative
  instrument** — the ranking feeds D4's per-group allocation; the ladder's
  reported numbers for actual candidates come from ORT evaluation of exported
  graphs, as for M1/M2.

## 3. The sensitivity index

One registered scalar per group for the report's ranking: **mean Δprimary
across the four ratios** (baseline primary minus pruned primary, averaged).
The full per-ratio curves are the D4 input; the scalar exists so "which groups
are fragile" has one pre-declared answer instead of a post-hoc choice of
column.

D3 takes **no selection decision**: it produces the curves, the ranking, and a
reproducible config. D4 will build ~15/30/45% MAC-reduction candidates from
these curves (allocation rule registered there, before D4 numbers exist).

## 4. Profiling

Parameters/MACs via `torch_pruning.utils.count_ops_and_params` at the frozen
`(1, 3, 192, 256)` input — the same counter before and after every mutation, so
requested-vs-realized reduction is internally consistent. The unpruned M0
reference is also recorded against the analytic C1a counter
(`input_cost.macs_at` = 293,402,624) with the divergence between the two
counters stated, since the ladder's `comparison.jsonl` uses the analytic
convention for M0/M1/M2.

## 5. Invariants after every mutation (abort on violation)

1. depthwise `groups == in_channels == out_channels` for every block;
2. every surviving expansion width ≥ 8 and a multiple of 8;
3. projection outputs, stem, `features.1`, final conv, classifier widths
   unchanged (bit-for-bit against the frozen architecture table);
4. forward and backward execute at `(1, 3, 192, 256)`;
5. ONNX export succeeds under the P0 opset-17 contract and the exported
   graph's changed shapes match the mutated module widths.

## 6. Environment

`torch-pruning` 1.6.1 (the `requirements.lock` pin; the wheel's
`__version__` string reports 1.6.0 — a known upstream packaging quirk,
recorded here so nobody chases it later), torch 2.11, gx10. The gx10 venv was
aligned to the lock before any D3 evidence ran.
