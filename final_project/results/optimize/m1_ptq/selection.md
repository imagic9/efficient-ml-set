# D1 PTQ candidate selection

Rule: `results/optimize/m1_ptq/preregistration.md` — applied mechanically by `wildlife_trigger.optimize.select_ptq`.

Reference (M0 FP32 through deployment ORT): primary 0.3667, cis F2 0.6280, trans F2 0.1054.

| method | primary | cis F2 | trans F2 | bytes | threshold | status |
|---|---:|---:|---:|---:|---:|---|
| percentile | 0.3527 | 0.6015 | 0.1039 | 2,620,130 | 0.496375 | recall_floor_infeasible |
| minmax | 0.3520 | 0.5958 | 0.1083 | 2,620,211 | 0.475551 | recall_floor_infeasible |
| entropy | 0.3520 | 0.5958 | 0.1083 | 2,620,211 | 0.475551 | recall_floor_infeasible |

**Selected: percentile** (`d1_m1_ptq_percentile`).

Material-drop check: primary ratio 0.9618 vs the 0.95 line; cis_val_clean +4.2%, trans_val +1.4% vs the -10% line → quantization debugging not triggered.

QOperator: not warranted (QOperator is generated only if S8S8 QDQ coverage on the ARM64 host shows float Conv/Gemm/MatMul surviving optimization (preregistration §5)).
