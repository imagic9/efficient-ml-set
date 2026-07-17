# Model card — M3, structured-pruned FP32 (c30, 30% MAC target)

**One sentence:** M0's MobileNetV2 with expansion channels physically removed
to a 30% MAC reduction by sensitivity-guided structured pruning, then
fine-tuned back — 1,761,720 params (21.5% under M0), 205,614,080 MACs (29.9%
under M0), the first candidate on the ladder whose **architecture**, not just
its weights' representation, is smaller.

| | |
|---|---|
| Architecture | MobileNetV2 with 14 of 16 expansion groups narrowed (see widths below); depthwise/projection couplings and residual widths preserved; every surviving width a multiple of 8 |
| Starts from | M0 checkpoint `42079c36…` (epoch 11), physically pruned — never a mask |
| Input | `[1, 3, 192, 256]` NCHW float32, the frozen §5.5 preprocessing — unchanged |
| Deployable artifact | `results/optimize/m3_prune/c30/model.onnx`, opset 17, FP32, **7,035,950 B** (M0 8,950,645 → 21.4% smaller; larger than the INT8 M1/M2 because it is FP32) |
| Fine-tune run | `d4_m3_c30_20260717T052416Z`, best epoch **2** of an early-stopped run, frozen §7.2 rule |
| Params / MACs | 1,761,720 params (−21.5%); 205,614,080 ladder MACs (−29.9%); tp-counter MAC cut 30.02% |
| Policy | `artifacts/policies/bobcat_m3_prune_c30_v1.json`, threshold **0.800026**, status **`recall_floor_infeasible`** |
| Selection | `results/optimize/m3_prune/selection.{json,md}` — mechanical, pre-registered (`m3_registration.md`) |

## How M3 was produced (D4, DESIGN §8.3)

1. **D3 first** gave the per-group sensitivity curves (16 groups × 4 ratios,
   no fine-tune); D3 took no decision.
2. **Registration before any M3 number** (`m3_registration.md`): greedy
   marginal-damage-per-MAC allocation (quantum 8, capped at the measured 0.5
   ratio), targets 15/30/45%, fine-tune at M0's own 3e-4 for ≤15 epochs
   (patience 4) under the frozen §7.2 contract, mechanical Pareto selection
   with the D1 0.95 recovery line.
3. Three candidates (`c15`, `c30`, `c45`) created, physically pruned under the
   D3 contract (%8 asserted before any tuning), fine-tuned, scored through
   deployment ORT, calibrated. `select_m3` chose **c30**: the largest realized
   MAC reduction among non-dominated candidates clearing the recovery line.

**The allocation left the fragile groups alone.** `features.15` (960-wide,
D3-rank-2 fragile) and `features.7` are untouched; the robust mid-network
groups `features.8/9/10` were halved and `features.12` cut two-thirds — the
marginal-damage rule spending its budget exactly where D3 said it was cheap.
Surviving expansion widths: `features.{2:48, 3:72, 4:104, 5:144, 6:144,
7:192, 8:192, 9:192, 10:192, 11:288, 12:288, 13:360, 14:504, 15:960, 16:840,
17:544}`.

**Findings for the record.** (a) **Pruning without fine-tuning is
catastrophic here** — the pruned-untuned primary is 0.000 (c30 and c45) and
0.013 (c15). All of M3's accuracy is the fine-tune repairing the cut, not the
cut being gentle. (b) **Recovery is non-monotonic in the cut**: c30 (30%)
recovered to primary 0.3583, *above* c15 (15%) at 0.3259 — so c15 is dominated
by c30 on both axes, and "a gentler cut is a safer cut" is false once
fine-tuning is in play. (c) The frontier **bends down past 30%**: c45 at 43%
recovered only to 0.3166 and missed the recovery line. 30% is the knee on this
architecture.

## Metrics

Validation, scored through deployment ORT (CPU EP, batch 1); test stays
sealed. Reference: M0's ONNX through the same evaluator.

### At the 0.5 selection yardstick

| | M3 cis | M0-ref cis | M3 trans | M0-ref trans |
|---|---:|---:|---:|---:|
| bobcat F2 | 0.5879 | 0.6280 | 0.1287 | 0.1054 |
| frame recall | 0.7708 | — | 0.1072 | — |
| precision | 0.3016 | — | 0.6589 | — |
| false-fire | 0.0837 | — | 0.0472 | — |
| average precision | 0.4946 | 0.5948 | 0.5967 | 0.5256 |
| event capture | 0.68 | — | — | — |

Primary (mean F2@0.5): **0.3583** vs reference 0.3667 → ratio 0.977, above the
0.95 recovery line. **M3 buys a 30% MAC reduction for a 2.3% primary cost** —
and trans F2 actually *rises* (0.1287 vs 0.1054), while cis falls (0.5879 vs
0.6280). Structured pruning + fine-tune redistributed capacity toward the
harder unseen-camera domain here; that is a measured property of this run, not
a general claim.

### At the calibrated operating point 0.800026 (`bobcat_m3_prune_c30_v1`)

| | cis-val-clean | trans-val |
|---|---:|---:|
| bobcat F2 | 0.5221 | 0.0219 |
| frame recall | 0.5903 | 0.0177 |
| sequence-balanced recall | 0.58 | 0.0176 |
| false-fire | 0.0498 | 0.0043 |
| event capture | 0.68 | 0.0453 |

**The primary rule is NOT met** — the same registered status as the rest of
the ladder (`recall_floor_infeasible`); no threshold inside the 5% per-domain
false-fire budget reaches the 90% sequence-recall floor on both domains. The
high operating threshold (0.80) and the wide bootstrap interval
(**[0.704, 0.9399]**) both say the same thing: M3's trans-domain behaviour is
weak and unstable at the operating point, even though its trans F2 at the
yardstick reads well. D6 should weigh the interval, not the point.

## Intended use and limitations

Same intended use and out-of-scope list as [m0_fp32.md](m0_fp32.md). M3 is the
smallest-MAC *architecture* on the ladder but the second-largest *file* (FP32,
7.03 MB) — its value is latency on the Pi, which only Pi measurement can
confirm; gx10 latency is used solely to detect float fallback (DESIGN §12.4).
M4 (D5) applies the validated QAT recipe to exactly this checkpoint and is
where the pruned architecture becomes a small INT8 file.

## Parity and deployment evidence (P3/P4, gates registered before measurement)

FP32 has no integer-execution question, so P3 check 1 is the physical gate
(`m3_registration.md` §5):

| gate | result |
|---|---|
| Graph + physical shapes (P3.1) | exported conv-shape multiset equals the candidate record; no integer kernel executed (it is FP32) |
| Metrics reproduce (P3.2) | full validation re-run equals the recorded candidate exactly |
| ORT py↔cpp fixtures (P3.3) | clean under the registered gates (≤1e-4, identical argmax/decisions) |
| Binding (P3.4) | every C++ infer record names this artifact hash and policy id |
| C++ dataset parity (P4) | both full splits: 4,939 frames, worst gap 5.96e-08, 0 decision diffs, matrices equal |

Evidence: `results/optimize/m3_prune/c30/p3_quantized.json`,
`results/optimize/m3_prune/c30/p4_dataset_parity.json`.

## Hashes

| artifact | sha256 |
|---|---|
| `model.onnx` (deployable, pruned FP32) | `c7529ee608ed6e393fba517f5c0bc482e54ff5864b549c06d89c7fc4ce7b6a50` |
| source M0 checkpoint | `42079c362013898c3354a65bbf8ced4524504c0dfc20cb6efaa00dabe9209074` |
| D3 sensitivity report (allocation input) | `692053324e08bad688198fc2d97f89f22843f2993234c28c251eb333de6765cd` |

The `.onnx` and fine-tune checkpoints live on gx10 until the G5 release; every
hash above is committed.

## License

Same as [m0_fp32.md](m0_fp32.md).

## Machine-readable row

`results/model_selection/comparison.jsonl`, `model_id: M3` — written by
`wildlife_trigger.comparison --candidate`, params/MACs re-measured from the
hash-verified pruned checkpoint (not copied from M0's row).
