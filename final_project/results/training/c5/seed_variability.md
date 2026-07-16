# C5 — baseline training variability, and issue #18's answer

Three runs of DESIGN §7.2's frozen recipe, differing in nothing but the seed —
verified by `validate.seed_variability` (recipe fields, dataset manifests, and the
class map are compared before anything is aggregated). Machine-readable numbers:
[`seed_variability.json`](seed_variability.json). Seed 42 is the pre-registered
primary and *remains M0 regardless of these numbers*; seeds 17 and 73 confirm, they
do not compete.

## At the selected checkpoints (F2@0.5, the §7.2 selection yardstick)

| seed | run | best epoch | selection score | cis F2 | trans F2 | trans event capture |
|---:|---|---:|---:|---:|---:|---:|
| 17 | `c5_m0_fp32_seed17_20260716T173021Z` | 11 | 0.3737 | 0.6130 | 0.1343 | 0.1925 |
| 42 | `c2_m0_fp32_seed42_20260716T061203Z` (M0) | 11 | 0.3663 | 0.6272 | 0.1054 | 0.1849 |
| 73 | `c5_m0_fp32_seed73_20260716T174821Z` | 12 | 0.3872 | 0.6716 | 0.1029 | 0.1811 |
| | **mean ± std (n=3)** | | 0.3757 ± 0.0106 | **0.6373 ± 0.0305** | **0.1142 ± 0.0175** | 0.1862 ± 0.0058 |

Std is the n-1 sample estimate over three runs — a scale indicator, not a
distribution claim; §6.3's seq_id-cluster bootstrap remains the uncertainty measure
for any single model.

## Issue #18: seed noise or the recipe?

The registered question: M0 (seed 42) reached trans F2 0.1054 while the C1a
ablation arm — same contract, shorter comparison budget — hit 0.2684. Issue #18
registered the reading before these runs: *"If seeds 17/73 land near seed 42's
trans F2, the gap is the recipe; if they scatter across 0.10-0.27, it is noise."*

**They landed near seed 42. The evidence supports the recipe.**

- Best-checkpoint trans F2 across seeds: 0.1343 / 0.1054 / 0.1029 — a spread of
  0.031, nowhere near 0.27.
- Stronger: the **maximum trans F2 over every phase-B epoch** is 0.1343 / 0.1086 /
  0.1029. Forty phase-B epochs across three seeds of the full recipe and not one
  crosses 0.135, while the short-budget arm's single run crossed 0.25 once.
- cis F2 moves the other way (0.6130-0.6716, all above the C1a arm's 0.5875): the
  full budget reliably buys cis accuracy and reliably does not buy trans recall —
  the shape of fitting the seen cameras harder.

Two caveats stay attached to the verdict, both from the issue:

1. The C1a arm is itself **one seed**, and its 0.2684 was a one-epoch spike between
   0.1251 and 0.1310. What is *established* is the full recipe's tight trans
   ceiling (~0.11-0.13 across three seeds); the arm's 0.2684 is one observation of
   a different budget, not a measured distribution.
2. The two recipes differ in coupled ways (head steps 1,445 vs 1,055; cosine
   `T_max` 7,225 vs 4,945; total budget). The evidence indicts the budget as a
   whole; it does not localize which component.

## What follows, and what must not

Nothing retrains. §7.2 is pre-registered, M0 stands, and every D-phase candidate
initializes from M0 — this comparison changes none of that. Tuning `head_epochs`
or the schedule to close a validation gap would be tuning on the domain the
project uses to make decisions (issue #18's "what must not happen"). If the recipe
is ever reconsidered, it is a DESIGN §7.2 amendment with its own pre-registration,
and DESIGN §18's registered outcome stands meanwhile: trans recall is poor and is
reported honestly.

## Provenance

- Confirmation seeds trained this session under the frozen recipe with no
  overrides: `configs/train/m0_fp32_seed{17,73}.yaml` (PR #34); both runs'
  `selection_audit.json` re-derive their selected epochs from the rule (`agrees:
  true` for both).
- Both new `predictions.npz` are scored in the deployment regime
  (`cudnn_tf32: False` recorded in-file), per the DESIGN §6.3 amendment of
  2026-07-16 (issue #30). Seed 42's npz predates the amendment and stays as
  committed; cross-seed numbers here come from the in-training histories, where
  all three seeds share one regime.
