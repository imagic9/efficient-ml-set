#!/usr/bin/env python3
"""Gate E6 — the C++ application is correct before any performance claim (PLAN E6).

E6 is a consolidation gate, not a re-run: it cites the parity evidence each shortlisted
model already earned and the E6-specific experiments, and asserts they hold together.
A performance number (Phase F, on the Pi) is only meaningful once the thing being timed
is proven correct, so this gate stands between the two.

For each shortlisted model (M0, M2, M4) it binds the parity chain by sha256 to the
frozen artifact and checks each verdict:

  - **P1** preprocessing parity — model-independent (the 256x192 contract is shared),
    proven once on M0 and cited for all three;
  - **P2 / p_ort_cpp** (FP32 M0) or **P3** (INT8 M2/M4) — the ORT/Python numeric gate
    appropriate to the model's dtype;
  - **P4** — C++ dataset parity over the validation manifests.

Then the E6 experiments:

  - **QEMU ISA parity** (pre-rental cortex-a76 rehearsal) passed;
  - **native-vs-target** build-and-test passed (5/5 both, bit-identical decisions);
  - **reduced-decode drift** ran validly and recorded its decision (reduced decode
    rejected -> shipping full decode);
  - the **optimization matrix** collated cleanly (diagnostic; not a correctness gate);
  - both the **ALL and EXTENDED** optimized graphs were retained for inspection, so the
    graph-level comparison rests on artifacts, not on inferring execution from node
    names (integer execution itself is P0's ort_coverage, cited).

Gate E6 passes iff the parity chains and the two correctness experiments (QEMU,
native-vs-target) pass and the decode-drift experiment was valid. The matrix and the
decode-drift *decision* concern optional knobs; the shipping pipeline is unchanged
(fused, full decode, ORT_ENABLE_ALL, arena on), so neither can fail the gate.

Usage (gx10, driven by scripts/run_e6_gate.sh):
    python -m wildlife_trigger.validate.e6_gate \\
        --freeze results/model_selection/pre_pi_freeze.json \\
        --e6-dir results/e6 --output results/e6/e6_gate.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..runs import atomic_write_json, sha256_file


def cite(path: Path, expect_sha: str | None) -> dict:
    """Read one evidence file's verdict and (where it names a model) its sha binding."""
    entry = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        entry["passed"] = False
        entry["error"] = "evidence file missing"
        return entry
    d = json.loads(path.read_text())
    entry["gate"] = d.get("gate")
    entry["passed"] = bool(d.get("verdict", {}).get("passed"))
    onnx_sha = (d.get("onnx") or {}).get("sha256")
    if expect_sha and onnx_sha:
        entry["sha_bound"] = onnx_sha == expect_sha
        if not entry["sha_bound"]:
            entry["passed"] = False
            entry["error"] = f"evidence sha {onnx_sha[:12]} != frozen {expect_sha[:12]}"
    return entry


def model_chain(model: dict, results_root: Path) -> dict:
    mid = model["model_id"]
    kind = model.get("kind")
    sha = model["onnx"].get("sha256")
    onnx_path = Path(model["onnx"]["artifact"])
    run_dir = onnx_path.parent                      # the candidate's directory
    run_id = run_dir.name

    chain: dict[str, dict] = {}
    # P1 preprocessing parity is model-independent; cite M0's (the shared contract).
    m0_parity = results_root / "parity" / run_id if kind == "fp32_baseline" else None

    if kind == "fp32_baseline":
        parity = results_root / "parity" / run_id
        chain["P1_preprocess"] = cite(parity / "p1_preprocess.json", None)
        chain["P2_fp32"] = cite(parity / "p2_fp32.json", sha)
        chain["p_ort_cpp"] = cite(parity / "p_ort_cpp.json", None)
        chain["P4_dataset"] = cite(results_root / "e4" / "p4_dataset_parity_m0.json", sha)
    else:
        # INT8: P1 is the shared preprocessing contract (cited via M0 by the caller),
        # P3 is the quantized numeric gate, P4 the dataset parity.
        chain["P3_quantized"] = cite(run_dir / "p3_quantized.json", sha)
        chain["P4_dataset"] = cite(run_dir / "p4_dataset_parity.json", sha)

    passed = all(c["passed"] for c in chain.values())
    return {"model": mid, "kind": kind, "sha256": sha, "chain": chain,
            "shared_p1": str(m0_parity / "p1_preprocess.json") if m0_parity else "M0 P1 (shared preprocessing contract)",
            "passed": passed}


def cite_experiment(path: Path) -> tuple[dict, dict | None]:
    entry = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        entry["passed"] = False
        entry["error"] = "missing"
        return entry, None
    d = json.loads(path.read_text())
    entry["passed"] = bool(d.get("verdict", {}).get("passed"))
    return entry, d


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--e6-dir", type=Path, default=Path("results/e6"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    freeze = json.loads(args.freeze.read_text())
    models = [m for m in freeze["models"] if m["model_id"] in ("M0", "M2", "M4")]

    failures: list[str] = []

    # 1. Parity chains, per shortlisted model.
    chains = [model_chain(m, args.results_root) for m in models]
    for c in chains:
        if not c["passed"]:
            bad = [k for k, v in c["chain"].items() if not v["passed"]]
            failures.append(f"{c['model']} parity chain incomplete: {bad}")

    # 2. E6 experiments.
    qemu, _ = cite_experiment(args.e6_dir / "qemu_parity.json")
    nvt, _ = cite_experiment(args.e6_dir / "native_vs_target.json")
    matrix, _ = cite_experiment(args.e6_dir / "optimization_matrix.json")
    if not qemu["passed"]:
        failures.append("QEMU ISA parity did not pass")
    if not nvt["passed"]:
        failures.append("native-vs-target build-and-test did not pass")
    if not matrix["passed"]:
        failures.append("optimization matrix did not collate cleanly")

    drift_path = args.e6_dir / "decode_drift.json"
    drift = {"path": str(drift_path), "exists": drift_path.exists()}
    if not drift_path.exists():
        failures.append("decode-drift evidence missing")
        drift["experiment_valid"] = False
    else:
        dd = json.loads(drift_path.read_text())
        drift["experiment_valid"] = bool(dd.get("experiment_valid"))
        drift["reduced_decode_adopted"] = dd.get("reduced_decode_adopted")
        drift["conclusion"] = dd.get("conclusion")
        if not drift["experiment_valid"]:
            failures.append("decode-drift experiment was invalid (runs incomplete)")

    # 3. Retained ALL/EXTENDED optimized graphs (artifacts, not node-name inference).
    graphs_dir = args.e6_dir / "graphs"
    graphs = {}
    for level in ("all", "extended"):
        g = graphs_dir / f"opt_{level}.onnx"
        graphs[level] = {"path": str(g), "exists": g.exists(),
                         "sha256": sha256_file(g) if g.exists() else None,
                         "bytes": g.stat().st_size if g.exists() else 0}
        if not g.exists():
            failures.append(f"optimized graph for ORT_ENABLE_{level.upper()} not retained")
    profiles = sorted(str(p) for p in graphs_dir.glob("prof_*"))

    passed = not failures
    report = {
        "gate": "E6 — the C++ application is correct before performance claims (PLAN E6)",
        "shipping_pipeline": {
            "preprocess": "fused", "decode": "full", "graph_optimization": "all",
            "cpu_arena": "on", "note": "unchanged by the optimization experiments; "
                                       "reduced decode measured and rejected.",
        },
        "parity_chains": chains,
        "experiments": {
            "qemu_isa_parity": qemu,
            "native_vs_target": nvt,
            "optimization_matrix_diagnostic": matrix,
            "reduced_decode_drift": drift,
        },
        "retained_graphs": {"levels": graphs, "profiles": profiles,
                            "integer_execution_proof": "P0 ort_coverage (cited); the "
                            "retained graphs are for inspection, never shipped"},
        "verdict": {"passed": passed, "failures": failures},
    }
    atomic_write_json(args.output, report)

    for c in chains:
        marks = " ".join(f"{k.split('_')[0]}={'P' if v['passed'] else 'F'}"
                         for k, v in c["chain"].items())
        print(f"{c['model']} ({c['kind']}): {'PASS' if c['passed'] else 'FAIL'} — {marks}")
    print(f"QEMU ISA parity: {'PASS' if qemu['passed'] else 'FAIL'}")
    print(f"native-vs-target: {'PASS' if nvt['passed'] else 'FAIL'}")
    print(f"optimization matrix (diagnostic): {'OK' if matrix['passed'] else 'FAIL'}")
    print(f"reduced-decode drift: valid={drift.get('experiment_valid')}, "
          f"adopted={drift.get('reduced_decode_adopted')}")
    print(f"retained graphs: ALL={graphs['all']['exists']}, EXTENDED={graphs['extended']['exists']}")
    for f in failures:
        print(f"    FAIL: {f}")
    print(f"\nGATE E6 {'PASSED' if passed else 'FAILED'}; wrote {args.output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
