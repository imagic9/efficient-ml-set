# D1 PTQ candidate selection

Rule: `results/optimize/m1_ptq/preregistration.md` — applied mechanically by `wildlife_trigger.optimize.select_ptq`.

Reference (M0 FP32 through deployment ORT): primary 0.3667, cis F2 0.6280, trans F2 0.1054.

| method | primary | cis F2 | trans F2 | bytes | threshold | status |
|---|---:|---:|---:|---:|---:|---|
| lr5e-5 | 0.3832 | 0.6499 | 0.1166 | 2,536,267 | 0.650390 | recall_floor_infeasible |
| lr1e-5 | 0.3704 | 0.6446 | 0.0963 | 2,536,335 | 0.644010 | recall_floor_infeasible |
| lr3e-5 | 0.3628 | 0.6340 | 0.0916 | 2,536,369 | 0.530991 | recall_floor_infeasible |

**Selected: lr5e-5** (`d2_m2_qat_lr5e-5`).

Material-drop check: primary ratio 1.0451 vs the 0.95 line; cis_val_clean -3.5%, trans_val -10.5% vs the -10% line → quantization debugging not triggered.

QOperator: not warranted (QOperator is generated only if S8S8 QDQ coverage on the ARM64 host shows float Conv/Gemm/MatMul surviving optimization (preregistration §5)).
