#!/usr/bin/env python3
"""Export a PyTorch model to ONNX under the P0 opset contract.

DESIGN §8 fixes one provisional export opset — 17 — across M0-M4, so that a
latency or accuracy difference between candidates is a property of the candidate
and not of the opset it happened to be exported at. Opset 9, inherited from the
legacy course spike, is forbidden outright.

This module enforces that contract at the only place it can be enforced cheaply:
the export call. It also re-reads the written file and reports what is actually
in it, because `torch.onnx.export` is asked for an opset — it is not a promise
that the graph carries one. The verification is what P0 records, not the request.

Usage:
    python -m wildlife_trigger.models.export --output artifacts/m0_fp32.onnx
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import onnx
import torch
import torch.nn as nn

# The P0-accepted opset. DESIGN §8: provisional, one value for every candidate,
# changed only if P0 proves a concrete incompatibility — and then every export and
# parity fixture is re-run.
P0_OPSET = 17

# Not merely "old": the legacy course spike exported at 9, and an opset-9 graph
# has no native QDQ quantization support, so it would quietly force a different
# quantization meaning than the one this project measures. Named explicitly so the
# rejection message can say why rather than "too low".
FORBIDDEN_OPSETS = {9}

# The default ONNX domain. Its opset is the one the contract is about; a graph may
# legitimately carry other domains (e.g. com.microsoft after quantization).
DEFAULT_DOMAIN = ""


class OpsetContractError(RuntimeError):
    """The export opset violates the P0 contract, as requested or as produced."""


def check_opset_request(opset: int) -> None:
    """Reject a forbidden or off-contract opset before spending an export on it."""
    if opset in FORBIDDEN_OPSETS:
        raise OpsetContractError(
            f"opset {opset} is forbidden by DESIGN §8: it is the legacy course "
            "spike's opset and predates native QDQ quantization support, so a "
            "graph exported at it would carry a different quantization meaning "
            f"than the rest of the comparison. Use opset {P0_OPSET}."
        )
    if opset != P0_OPSET:
        raise OpsetContractError(
            f"opset {opset} is not the P0 contract opset {P0_OPSET}. DESIGN §8 "
            "requires one opset across M0-M4 so candidates stay comparable. "
            "Changing it is a design decision that re-runs every export and "
            "parity fixture — not an export-call argument."
        )


def graph_opsets(model: onnx.ModelProto) -> dict[str, int]:
    """Map domain -> opset version as actually recorded in the file."""
    return {entry.domain: entry.version for entry in model.opset_import}


def verify_exported_opset(path: Path, expected: int = P0_OPSET) -> dict[str, int]:
    """Confirm the written graph carries the opset that was requested.

    Separate from `check_opset_request` on purpose. The request is an argument;
    this is evidence. An exporter that silently upgraded the default domain would
    otherwise be discovered on the Pi instead of here.
    """
    opsets = graph_opsets(onnx.load(str(path), load_external_data=False))
    actual = opsets.get(DEFAULT_DOMAIN)
    if actual != expected:
        raise OpsetContractError(
            f"{path} carries default-domain opset {actual}, not the requested "
            f"{expected}. Full opset_import: {opsets}. The export silently "
            "produced a different contract than it was asked for; do not use "
            "this artifact."
        )
    return opsets


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(path: Path) -> dict:
    """Summarise a written ONNX file: what P0 records as having been produced."""
    model = onnx.load(str(path), load_external_data=False)
    op_types: dict[str, int] = {}
    for node in model.graph.node:
        op_types[node.op_type] = op_types.get(node.op_type, 0) + 1

    def io_spec(values) -> list[dict]:
        specs = []
        for value in values:
            tensor = value.type.tensor_type
            specs.append(
                {
                    "name": value.name,
                    "dtype": onnx.TensorProto.DataType.Name(tensor.elem_type),
                    "shape": [
                        dim.dim_param if dim.HasField("dim_param") else dim.dim_value
                        for dim in tensor.shape.dim
                    ],
                }
            )
        return specs

    return {
        "path": str(path),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "ir_version": model.ir_version,
        "producer": f"{model.producer_name} {model.producer_version}".strip(),
        "opset_import": graph_opsets(model),
        "inputs": io_spec(model.graph.input),
        "outputs": io_spec(model.graph.output),
        "node_count": len(model.graph.node),
        "op_types": dict(sorted(op_types.items())),
    }


def export_onnx(
    model: nn.Module,
    output: Path,
    example: torch.Tensor,
    opset: int = P0_OPSET,
    input_names: tuple[str, ...] = ("input",),
    output_names: tuple[str, ...] = ("logits",),
    dynamo: bool = False,
    full_check: bool = True,
) -> dict:
    """Export `model` to ONNX and return the verified description of the result.

    `dynamo=False` selects the TorchScript exporter deliberately, and the P0 spike
    is where that choice gets tested rather than assumed: torch 2.11 defaults to
    the dynamo exporter, whose QDQ fake-quant lowering is the specific behaviour
    A3 has to establish for M2. Both exporters are reachable from this one
    argument so the spike can compare them on identical inputs.

    Batch dimension stays static. A dynamic batch would let ORT pick a different
    kernel at run time than the one benchmarked, and the deployed application
    infers one image at a time (DESIGN §9.1).
    """
    check_opset_request(opset)
    output.parent.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            torch.onnx.export(
                model,
                example,
                str(output),
                opset_version=opset,
                input_names=list(input_names),
                output_names=list(output_names),
                do_constant_folding=True,
                dynamo=dynamo,
            )
    finally:
        model.train(was_training)

    # check_model before verify_opset: a structurally invalid graph should be
    # reported as invalid, not as an opset mismatch.
    onnx.checker.check_model(str(output), full_check=full_check)
    verify_exported_opset(output, expected=opset)

    description = describe(output)
    description["exporter"] = "dynamo" if dynamo else "torchscript"
    return description


def main() -> int:
    from wildlife_trigger.models.mobilenet import build_mobilenet_v2, example_input

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--opset", type=int, default=P0_OPSET)
    parser.add_argument(
        "--dynamo",
        action="store_true",
        help="Use the dynamo exporter instead of TorchScript (P0 compares both).",
    )
    parser.add_argument(
        "--describe-json",
        type=Path,
        help="Also write the verified description here, for run provenance.",
    )
    args = parser.parse_args()

    model = build_mobilenet_v2(num_classes=args.num_classes, pretrained=True)
    description = export_onnx(
        model,
        args.output,
        example_input(),
        opset=args.opset,
        dynamo=args.dynamo,
    )

    print(json.dumps(description, indent=2))
    if args.describe_json:
        args.describe_json.parent.mkdir(parents=True, exist_ok=True)
        args.describe_json.write_text(json.dumps(description, indent=2) + "\n")
        print(f"wrote {args.describe_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
