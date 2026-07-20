#!/usr/bin/env python3
"""F3 freeze manifest — archive the frozen deployment state before test evaluation.

After final_decision.md selects M2, this records the exact, immutable artifacts and
runtime configuration that F4/F5 must run without change (PLAN F3 "no artifact or
configuration changes after this point"). Reads results/model_selection/pre_pi_freeze.json
(the hash-locked shortlist bundle) and re-verifies the on-disk sha256 of the selected
model + baseline so the freeze cannot record a stale hash. Writes results/f3/freeze_manifest.json.

Usage:  python3 scripts/build_freeze_manifest.py <git_commit>
"""
import hashlib
import json
import sys
from pathlib import Path

GIT_COMMIT = sys.argv[1] if len(sys.argv) > 1 else "UNKNOWN"
PPF = json.loads(Path("results/model_selection/pre_pi_freeze.json").read_text())


def sha256(path):
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def find(model_id):
    for m in PPF["models"]:
        if m["model_id"] == model_id:
            return m
    raise KeyError(model_id)


winner = find("M2")     # final optimized model (final_decision.md, DESIGN §8.4)
baseline = find("M0")   # mandatory FP32 baseline (§12.2)

# Re-verify on-disk hashes against the frozen bundle — a freeze must not bless a moved byte.
for m in (winner, baseline):
    onx = m["onnx"]
    disk = sha256(onx["artifact"])
    m["_onnx_sha256_ondisk"] = disk
    m["_onnx_sha256_matches_freeze"] = (disk == onx["sha256"])

manifest = {
    "kind": "f3_freeze_manifest",
    "schema_version": 1,
    "design": "PLAN F3 / DESIGN §8.4-§8.5, §12.3",
    "frozen_at_git_commit": GIT_COMMIT,
    "decision": "results/model_selection/final_decision.md",
    "final_optimized_model": {
        "model_id": "M2",
        "kind": winner["kind"],
        "onnx": winner["onnx"]["artifact"],
        "onnx_sha256": winner["onnx"]["sha256"],
        "onnx_sha256_ondisk": winner["_onnx_sha256_ondisk"],
        "onnx_sha256_matches_freeze": winner["_onnx_sha256_matches_freeze"],
        "policy": winner["policy"]["path"],
        "policy_id": winner["policy"]["policy_id"],
        "policy_sha256": winner["policy"]["sha256"],
        "threshold": winner["policy"]["threshold"],
        "status": winner["policy"]["status"],
        "macs": winner["macs"],
        "params": winner["params"],
    },
    "baseline_model": {
        "model_id": "M0",
        "kind": baseline["kind"],
        "onnx": baseline["onnx"]["artifact"],
        "onnx_sha256": baseline["onnx"]["sha256"],
        "onnx_sha256_ondisk": baseline["_onnx_sha256_ondisk"],
        "onnx_sha256_matches_freeze": baseline["_onnx_sha256_matches_freeze"],
        "policy": baseline["policy"]["path"],
        "policy_id": baseline["policy"]["policy_id"],
        "threshold": baseline["policy"]["threshold"],
        "status": baseline["policy"]["status"],
    },
    "class_map": PPF["class_map"],
    "preprocessing": PPF["preprocessing"],
    "runtime_config_frozen": {
        "intra_op_threads": 1,
        "decode": "full",
        "graph_optimization": "all (ORT_ENABLE_ALL)",
        "cpu_arena": "on",
        "preprocess": "fused",
        "width": 256,
        "height": 192,
        "onnxruntime_version": "1.27.0",
        "note": "the exact config every P1-P4 parity gate validated; F4 Pi-vs-gx10 parity is "
                "config-matched. threads=3 measured better (F2) but not folded in, to keep the "
                "FP32-baseline comparison parity-clean.",
    },
    "benchmark_manifest": PPF["benchmark"],
    "pi_latency_f2": {
        "host": "Raspberry Pi CM5 (BCM2712 / Cortex-A76 @ 2.4 GHz, 8 GB), Ubuntu 24.04",
        "M0_fp32_p50_ms_t1": 48.23, "M0_fp32_fps_t1": 20.7,
        "M2_int8qat_p50_ms_t1": 21.30, "M2_int8qat_fps_t1": 47.0,
        "speedup_end_to_end_t1": 2.26,
        "source": "results/f2/f2_summary.json",
    },
    "confirmation_seeds": {
        "seeds": [17, 73],
        "transformation": "M2 QAT (lr 5e-5, 6 epochs) from the frozen M0 checkpoint",
        "gating": False,
        "note": "variability measurement only; never replaces the seed-42 deployment artifact; "
                "must finish before Gate G (PLAN F3).",
    },
    "test_labels": "SEALED — no test manifest named; opened only after this freeze (F4).",
    "invariant": "No artifact or configuration change is permitted after this manifest (PLAN F3).",
}

out = Path("results/f3/freeze_manifest.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(f"wrote {out}")
ok = (winner["_onnx_sha256_matches_freeze"] and baseline["_onnx_sha256_matches_freeze"])
print(f"  winner  M2 sha matches freeze: {winner['_onnx_sha256_matches_freeze']}")
print(f"  baseline M0 sha matches freeze: {baseline['_onnx_sha256_matches_freeze']}")
print(f"  FREEZE {'OK' if ok else 'FAILED — on-disk hash moved!'}")
sys.exit(0 if ok else 1)
