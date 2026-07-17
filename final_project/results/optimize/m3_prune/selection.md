# D4 — M3 candidate selection

Rule: `results/optimize/m3_prune/m3_registration.md` §4 — applied mechanically by `wildlife_trigger.optimize.select_m3`.

Reference (M0 FP32 through deployment ORT): primary 0.3667; the recovery line is 0.95 x that = 0.3484.

| candidate | primary | cis F2 | trans F2 | pre-FT primary | ladder MACs | tp MAC cut | params | bytes | threshold | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| c30 | 0.3583 | 0.5879 | 0.1287 | 0.0000 | 205,614,080 | 30.02% | 1,761,720 | 7,035,950 | 0.800026 | recall_floor_infeasible |
| c15 | 0.3259 | 0.5502 | 0.1016 | 0.0130 | 248,620,160 | 15.31% | 1,957,320 | 7,810,036 | 0.619709 | recall_floor_infeasible |
| c45 | 0.3166 | 0.5912 | 0.0420 | 0.0000 | 166,765,568 | 42.99% | 1,340,912 | 5,364,784 | 0.669527 | recall_floor_infeasible |

Non-dominated: c30, c45; dominated: c15

**Selected: c30** (`d4_m3_c30`) — largest realized MAC reduction among non-dominated candidates with primary >= 0.95 x reference (0.3484).

Recovery line met: **True**.
