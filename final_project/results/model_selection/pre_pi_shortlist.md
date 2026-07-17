# Pre-Pi deployable shortlist (DESIGN §8.5)

Mechanically derived from `comparison.jsonl` by `wildlife_trigger.optimize.pre_pi_shortlist`. Every row already carries a passing parity gate (`comparison.py` refuses to write a failing one).

## The recall operating rule (§8.5 step 2)

no candidate meets the 90% sequence-balanced bobcat-recall rule (all recall_floor_infeasible); §8.5 step 2 fallback — retain the best documented candidates by dominance.

## Candidates (validation, deployment ORT)

| model | kind | primary (mean bobcat F2) | cis F2 | trans F2 | MACs | bytes | status |
|---|---|---:|---:|---:|---:|---:|---|
| M0 (**baseline**) | fp32_baseline | 0.3663 | 0.6272 | 0.1054 | 293,402,624 | 8,950,645 | recall_floor_infeasible |
| M1 (rejected) | int8_ptq | 0.3527 | 0.6015 | 0.1039 | 293,402,624 | 2,620,130 | recall_floor_infeasible |
| M2 (**shortlist**) | int8_qat | 0.3832 | 0.6499 | 0.1166 | 293,402,624 | 2,536,267 | recall_floor_infeasible |
| M3 (rejected) | pruned_fp32 | 0.3583 | 0.5879 | 0.1287 | 205,614,080 | 7,035,950 | recall_floor_infeasible |
| M4 (**shortlist**) | pruned_qat | 0.3730 | 0.6529 | 0.0930 | 205,614,080 | 2,014,806 | recall_floor_infeasible |

## Rejections (dominated on §8.5's three axes)

- **M1** dominated by M2, M4 — lower or equal on primary F2 (0.3527) with no advantage on both MACs and size.
- **M3** dominated by M4 — lower or equal on primary F2 (0.3583) with no advantage on both MACs and size.

## Shortlist frozen for the Pi

**M0 · M2 · M4**

M0 is the mandatory FP32 baseline (§12.2). The optimized front is every non-dominated candidate; Pi latency (F-phase), never gx10 latency (§12.4), chooses the final model from this set.

Float-fallback check: integer execution is proven per-candidate by the committed coverage verdicts (P3 check 1); gx10 latency is not used to rank Cortex-A76 (§12.4).
