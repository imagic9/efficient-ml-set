# D4 — M3 candidate registration (committed before any M3 number exists)

D3 produced the curves and explicitly took no decision. This document is the
decision rule set for D4, fixed before the first candidate is created, in the
same pattern as D1/D2's pre-registrations.

## 1. Allocation rule: greedy marginal damage per MAC

Targets: **15%, 30%, 45%** MAC reduction (of the torch-pruning-counter
baseline 312,467,472 — the counter D3's evidence is measured in). Candidates
are named `c15`, `c30`, `c45`.

For each target, per-group removals are chosen by a deterministic greedy loop:

1. the removable quantum is **8 channels** (the alignment unit);
2. each group's damage at a width is linear interpolation of its measured D3
   curve (Δprimary at realized ratios, with (0, 0) prepended) on the
   channels-removed axis — single-group measurements, so additivity across
   groups is an assumption, stated here rather than hidden; fine-tuning exists
   to repair exactly this;
3. each group's MAC gain per channel is its measured `mac_reduction@0.5`
   scaled linearly (exact for this architecture: every conv in a group scales
   linearly in the group's width);
4. repeatedly remove the quantum with the **lowest marginal Δprimary per MAC
   saved**, until the cumulative reduction reaches the target;
5. **cap: no group beyond realized ratio 0.5** — beyond the measured range is
   extrapolation, not evidence.

The cap makes the all-groups envelope **42.99%** (measured, sum of
`mac_reduction@0.5`), so the 45% target is *unreachable* within evidence:
**`c45` is registered as the envelope candidate** (every group at the cap),
and its requested-vs-realized gap is part of the finding, not a failure to
hide. Requested and realized reductions are recorded separately for all three
(DESIGN §8.3 step 4).

Before fine-tuning each candidate: physical channel removal is verified by the
D3 invariant suite, **every surviving width asserted ≥ 8 and ≡ 0 (mod 8)**,
group coupling proven, and the pruned-no-fine-tune validation primary recorded
(the recovery delta is evidence about fine-tuning, free at this point).

## 2. Fine-tune recipe (the fixed data/loss contract)

- Data, loss, augmentation, batching: **exactly the frozen §7.2 contract** —
  weighted CE (`ignore_index=-1`), train + empty-supplement, the B3 cache,
  batch 64, workers 8, seed 42.
- Init: the physically pruned M0 checkpoint (`42079c36…`), hash-verified.
- Optimiser: AdamW, **lr 3e-4** (M0's own full-phase LR — no new search space),
  weight decay 1e-4, cosine annealing over the run.
- AMP **on**, exactly as M0's own training (`train.py`'s GradScaler path);
  the §6.3 amendment governs *scoring*, which stays deployment-ORT.
- Budget: **max 15 epochs**, early stopping after **4** epochs without
  improvement under the frozen §7.2 selection rule (strict lexicographic
  improvement, earliest kept on ties). Per-epoch checkpoint selection by the
  same rule. No LR search: one arm per candidate — the LR is M0's own, and
  three candidates × one arm is the registered budget.

## 3. Evaluation, calibration, rows

The unchanged D-chain, per candidate: export at the P0 opset (17) with the
physical-shape check (exported conv-shape multiset must equal the pruned
module's), `evaluate_onnx` through deployment ORT (CPU EP, batch 1),
`calibrate_candidate` under frozen §6.3 → `bobcat_m3_prune_c{15,30,45}_v1`
policies (the `bobcat_<root>_<method>_v1` convention every D-phase driver
shares).
All three candidates' numbers are recorded; **only the selected M3 enters
`comparison.jsonl`** (the M1/M2 pattern).

Ladder-convention MACs for the M3 row are measured with the ladder's own
counter (`FlopCounterMode // 2`, generalised to take the pruned module);
params counted from the hash-verified checkpoint through the pruned
architecture. The tp-counter numbers stay in D3/D4 internal evidence; the two
conventions are never mixed (D3's rule, carried forward).

## 4. M3 selection rule (the validation Pareto frontier, made mechanical)

Inputs: the three candidates' deployment-ORT primaries, ladder-convention
MACs, and artifact bytes; the M0 ORT reference primary (0.3667) as the anchor.

1. Drop any candidate that failed a gate (export, invariants, calibration
   refusing to write a policy).
2. Drop dominated candidates: candidate A dominates B if A's primary ≥ B's
   and A's MACs ≤ B's, with at least one strict.
3. Among the non-dominated: **select the largest realized MAC reduction whose
   primary ≥ 0.95 × the M0 ORT reference** (the D1 material-drop line,
   reused verbatim — 0.3484 at the current reference).
4. If none reaches the line, select the highest-primary non-dominated
   candidate and record that the line failed — a pruning-hurts-more verdict is
   a result, not an emergency.

The selected candidate becomes **M3**, gets the parity gates and the
comparison row, and is the sole QAT init for M4 (D5). Selection reads
validation only; test stays sealed.

## 5. Parity gates for the selected M3

FP32 has no integer-coverage question, so P3's check 1 is replaced by the
physical one (the driver reads the candidate's own `kind` — no flag to
forget); the rest are verbatim from D1/D2:

1. exported graph valid under the P0 contract, its conv-shape multiset equals
   the candidate's recorded pruned shapes (the D3 export check, now on the
   deployable artifact), **and** no integer kernel executed (an FP32 artifact
   that quantized itself somewhere would be a different candidate);
2. full validation re-run through ORT equals the recorded candidate exactly;
3. ORT Python vs C++ on the shared fixtures under the registered gates
   (logits ≤ 1e-4, identical argmax, identical decisions with the 1e-4
   carve-out);
4. no silent fallback: every C++ infer record names this artifact hash and
   policy id.

Then **P4** on both full validation splits, the registered corpus gates
unchanged. No candidate reaching a failed gate may be deployed (DESIGN §10).

## 6. What this registration forbids

Re-opening the ratio search after seeing candidate metrics; fine-tuning any
candidate with a different LR/budget than §2; selecting on anything but §4;
reading test data anywhere in D4.
