# D3 — expansion-group pruning sensitivity (no fine-tune)

Protocol: `results/optimize/m3_prune/sensitivity_protocol.md` — applied mechanically.

Baseline (M0 `42079c362013…`, torch GPU TF32-off): primary **0.3667**, 312,467,472 MACs / 2,244,368 params (torch-pruning counter; the ladder's analytic M0 reference is 293,402,624).

Sensitivity index = mean Δprimary over requested ratios [0.125, 0.25, 0.375, 0.5] (registered §3). Higher = more fragile.

| rank | group | width | index | Δ@0.125 | Δ@0.25 | Δ@0.375 | Δ@0.5 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `features.3.conv.0.0` | 144 | +0.2997 | +0.1981 | +0.3276 | +0.3064 | +0.3667 |
| 2 | `features.15.conv.0.0` | 960 | +0.2967 | +0.1571 | +0.3091 | +0.3581 | +0.3624 |
| 3 | `features.14.conv.0.0` | 576 | +0.2741 | +0.0859 | +0.2770 | +0.3667 | +0.3667 |
| 4 | `features.2.conv.0.0` | 96 | +0.2655 | +0.1505 | +0.2279 | +0.3274 | +0.3560 |
| 5 | `features.16.conv.0.0` | 960 | +0.2422 | +0.0788 | +0.2914 | +0.3287 | +0.2698 |
| 6 | `features.4.conv.0.0` | 144 | +0.2220 | +0.0469 | +0.1660 | +0.3126 | +0.3624 |
| 7 | `features.7.conv.0.0` | 192 | +0.2216 | +0.0948 | +0.1041 | +0.3209 | +0.3667 |
| 8 | `features.11.conv.0.0` | 384 | +0.2186 | +0.1067 | +0.0904 | +0.3107 | +0.3667 |
| 9 | `features.5.conv.0.0` | 192 | +0.1942 | +0.0727 | +0.1106 | +0.2276 | +0.3659 |
| 10 | `features.17.conv.0.0` | 960 | +0.1549 | +0.0401 | +0.1048 | +0.1385 | +0.3360 |
| 11 | `features.13.conv.0.0` | 576 | +0.1276 | +0.0675 | +0.0068 | +0.1072 | +0.3288 |
| 12 | `features.8.conv.0.0` | 384 | +0.0928 | +0.0358 | +0.0415 | +0.1106 | +0.1834 |
| 13 | `features.10.conv.0.0` | 384 | +0.0867 | +0.0575 | +0.0895 | +0.0896 | +0.1101 |
| 14 | `features.6.conv.0.0` | 192 | +0.0793 | +0.0331 | +0.0258 | +0.1399 | +0.1183 |
| 15 | `features.12.conv.0.0` | 576 | +0.0673 | +0.0221 | +0.0422 | +0.0849 | +0.1203 |
| 16 | `features.9.conv.0.0` | 384 | +0.0565 | +0.0585 | +0.0524 | +0.0685 | +0.0467 |

Full per-ratio curves (realized widths, MACs, per-domain metrics): `sensitivity.json`. D3 takes no selection decision; D4's allocation rule is registered separately before its numbers exist.
