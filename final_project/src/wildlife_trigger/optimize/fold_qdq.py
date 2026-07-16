#!/usr/bin/env python3
"""Fold constant weight-quantization into INT8 initializers (DESIGN §8.2).

The direct QDQ fake-quant export stores every weight as FP32 behind a
`QuantizeLinear -> DequantizeLinear` pair. ORT executes that graph as integer —
P0's verdict, unchanged — but only after constant-folding `Q(w)` at session
load; the *file* still carries float weights, which made the first M2 export
9,096,154 bytes: larger than M0 itself, 3.5x M1, and literally "a float graph
carrying rounded weights", the exact artifact DESIGN §8.2 forbids shipping.

The fold is ORT's own Basic-level offline optimization — constant folding and
redundant-node elimination only, no fused or hardware-specific kernels (those
begin at Extended; the NchwcTransformer warning that applies to ORT_ENABLE_ALL
saves does not apply here). Measured on the first M2 artifact: 9,096,154 ->
2,536,433 bytes, 87 INT8 initializers, every node in the ai.onnx domain, and
**bitwise identical outputs** — which this module refuses to take on faith:

- the folded graph must pass the full ONNX checker;
- its default-domain opset must equal the source's (the P0 contract);
- its outputs must equal the source's **exactly** on seeded random probes —
  both files resolve to the same int8 kernels, so any difference at all means
  the fold changed semantics and the artifact must not ship.

ORT's save also declares unused opset domains (com.microsoft etc.) in the
header; they are pruned so the artifact declares exactly what it contains.
"""

from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

EQUIVALENCE_PROBES = 8


def prune_unused_opset_imports(model: onnx.ModelProto) -> list[str]:
    """Drop opset declarations no node uses; the default domain always stays."""
    used = {node.domain or "" for node in model.graph.node}
    removed = [
        entry.domain
        for entry in model.opset_import
        if entry.domain not in used and entry.domain != ""
    ]
    kept = [e for e in model.opset_import if e.domain in used or e.domain == ""]
    del model.opset_import[:]
    model.opset_import.extend(kept)
    return removed


def verify_bitwise_equivalence(
    source: Path, folded: Path, input_shape: tuple[int, ...], seed: int = 0
) -> float:
    """Both graphs, same probes, exact equality — or a refusal."""
    a = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(str(folded), providers=["CPUExecutionProvider"])
    (spec,) = a.get_inputs()
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(EQUIVALENCE_PROBES):
        x = rng.standard_normal(input_shape).astype(np.float32)
        (out_a,) = a.run(None, {spec.name: x})
        (out_b,) = b.run(None, {spec.name: x})
        worst = max(worst, float(np.abs(out_a - out_b).max()))
    if worst != 0.0:
        raise RuntimeError(
            f"folded graph differs from the source (worst abs {worst:.3e}); a "
            "constant fold must be exact, so this artifact must not ship"
        )
    return worst


def fold_qdq_weights(
    source: Path, output: Path, input_shape: tuple[int, ...]
) -> dict:
    source_model = onnx.load(str(source), load_external_data=False)
    source_opset = {e.domain: e.version for e in source_model.opset_import}

    with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
        raw_folded = Path(scratch) / "folded.onnx"
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.optimized_model_filepath = str(raw_folded)
        ort.InferenceSession(str(source), options, providers=["CPUExecutionProvider"])

        model = onnx.load(str(raw_folded), load_external_data=False)

    removed_domains = prune_unused_opset_imports(model)
    onnx.save(model, str(output))

    onnx.checker.check_model(str(output), full_check=True)
    folded_opset = {e.domain: e.version for e in model.opset_import}
    if folded_opset.get("") != source_opset.get(""):
        raise RuntimeError(
            f"fold changed the default-domain opset {source_opset.get('')} -> "
            f"{folded_opset.get('')}; the P0 contract does not move here"
        )
    foreign = [d for d in folded_opset if d not in ("", "ai.onnx")]
    if foreign:
        raise RuntimeError(
            f"folded graph still declares non-standard domains {foreign}; "
            "refusing to ship an artifact that needs more than stock ONNX"
        )

    worst = verify_bitwise_equivalence(source, output, input_shape)

    initializer_dtypes = Counter(
        onnx.TensorProto.DataType.Name(init.data_type).lower()
        for init in model.graph.initializer
    )
    return {
        "tool": "wildlife_trigger.optimize.fold_qdq",
        "source_bytes": source.stat().st_size,
        "folded_bytes": output.stat().st_size,
        "initializer_dtypes": dict(sorted(initializer_dtypes.items())),
        "pruned_unused_opset_domains": sorted(removed_domains),
        "opset": folded_opset,
        "equivalence": {
            "probes": EQUIVALENCE_PROBES,
            "worst_abs_diff": worst,
            "exact": True,
        },
    }
