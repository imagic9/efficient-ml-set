# Submission manifest

> **Status: placeholder.** Completed at G5, once every artifact exists and every
> number traces to a machine-readable result. Nothing here is filled in from
> memory — DESIGN §9.2 forbids reconstructing a reported value by hand.

This file is the single index a reviewer opens first: every link, hash and
reproduction command for the final submission.

## Canonical locations

| Item | Location |
|---|---|
| Public repository | https://github.com/imagic9/efficient-ml-set |
| Release tag | `TBD` (`v1.0-final`) |
| Final commit | `TBD` |
| Report | `report/final_report.pdf` |
| Slides | `slides/final_presentation.pptx` / `.pdf` |
| Pi deployment bundle | `TBD` (GitHub Release) |

## Models

| ID | Description | File | SHA-256 |
|---|---|---|---|
| M0 | FP32 baseline | `TBD` | `TBD` |
| final | selected optimized model | `TBD` | `TBD` |

## Headline results

Filled from `results/` at G1-G2. Every value must cite its source file.

| Metric | M0 | Final | Source |
|---|---|---|---|
| Pi p95 end-to-end latency | `TBD` | `TBD` | `results/pi/` |
| Pi end-to-end FPS | `TBD` | `TBD` | `results/pi/` |
| Peak RSS | `TBD` | `TBD` | `results/pi/` |
| Model size | `TBD` | `TBD` | `results/` |
| cis-test bobcat recall | `TBD` | `TBD` | `results/evaluation/` |
| trans-test bobcat recall | `TBD` | `TBD` | `results/evaluation/` |

## Reproduction

```bash
# TBD at G5 — setup, data, train, export, C++ build, benchmark, demo
```

## Assignment mapping

| Rubric area | Points | Evidence |
|---|---:|---|
| Model training & optimization strategy | 15 | `TBD` |
| C++ inference implementation | 15 | `TBD` |
| Benchmarking & metrics | 10 | `TBD` |
| Results analysis & presentation | 10 | `TBD` |

## Definition of Done

DESIGN §19 holds the authoritative checklist. G5 runs every item before this file
is considered complete.
