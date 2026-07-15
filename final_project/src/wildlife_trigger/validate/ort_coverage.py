#!/usr/bin/env python3
"""Decide whether ORT actually executes a graph as integer — with evidence.

DESIGN §8.1/§8.2 state the rule this module exists to enforce: *do not rely on one
version-specific kernel name such as `QLinearConv` as the sole proof of INT8
execution*. A graph can contain QuantizeLinear/DequantizeLinear pairs, pass the
ONNX checker, carry INT8 initializers, and still run every convolution in FP32 —
ORT simply dequantizes, computes in float, and requantizes. That model is smaller
on disk and no faster, and nothing about its file structure says so.

Three independent signals are collected, because each can lie alone:

1. **The session-optimized graph.** What ORT decided to run after fusion, not what
   we exported. This is where a Conv that failed to fuse remains visibly a Conv
   with a DequantizeLinear feeding it.
2. **Initializer dtypes.** INT8 weights prove storage, never execution — recorded
   precisely so a reader can see storage and execution disagree.
3. **The ORT profile.** Per-node `input_type_shape` records the dtype each kernel
   was actually handed at run time. This is the only signal produced by execution
   rather than by inspection, and it is what settles the question.

Usage:
    python -m wildlife_trigger.validate.ort_coverage --model m2_qat.onnx \
        --optimized-out m2_opt.onnx --report m2_coverage.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

# Ops that only exist because a quantized kernel was selected. Presence is
# suggestive; the profile's dtypes are what confirm. The list spans the ONNX and
# com.microsoft domains and is deliberately broader than what MobileNetV2 can
# produce, so an unexpected-but-legitimate kernel is not scored as a float
# fallback.
INTEGER_OPS = frozenset(
    {
        "QLinearConv",
        "ConvInteger",
        "QLinearMatMul",
        "MatMulInteger",
        "MatMulIntegerToFloat",
        "QGemm",
        "QLinearAdd",
        "QLinearMul",
        "QLinearAveragePool",
        "QLinearGlobalAveragePool",
        "QLinearConcat",
        "QLinearLeakyRelu",
        "QLinearSigmoid",
        "DynamicQuantizeLSTM",
        "DynamicQuantizeMatMul",
    }
)

# The heavy compute of MobileNetV2. If one of these survives optimization as a
# float op inside a supposedly quantized model, that is the finding.
FLOAT_COMPUTE_OPS = frozenset({"Conv", "FusedConv", "Gemm", "MatMul"})

# Bookkeeping around the real work. Their presence is expected in any QDQ graph;
# what matters is how many, and where.
QDQ_OPS = frozenset({"QuantizeLinear", "DequantizeLinear"})

INTEGER_DTYPES = frozenset({"int8", "uint8", "int32"})


def save_optimized_graph(
    model: Path,
    optimized_out: Path,
    profile_prefix: Path | None = None,
    intra_op_threads: int = 1,
) -> tuple[ort.InferenceSession, str | None]:
    """Create a session at ORT_ENABLE_ALL, persisting what ORT chose to run.

    `ORT_ENABLE_ALL` matches DESIGN §8's starting point; ORT_ENABLE_EXTENDED is an
    explicitly named E6 candidate and is not silently substituted here.

    Single-threaded by default so the profile attributes time to kernels rather
    than to thread scheduling. Correctness work only — DESIGN §12.4 forbids
    treating any of this as a latency result.
    """
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.optimized_model_filepath = str(optimized_out)
    options.intra_op_num_threads = intra_op_threads
    if profile_prefix is not None:
        profile_prefix.parent.mkdir(parents=True, exist_ok=True)
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_prefix)

    optimized_out.parent.mkdir(parents=True, exist_ok=True)
    session = ort.InferenceSession(
        str(model), options, providers=["CPUExecutionProvider"]
    )
    return session, (str(profile_prefix) if profile_prefix else None)


def run_fixture(
    session: ort.InferenceSession, input_bin: Path | None = None, seed: int = 0
) -> np.ndarray:
    """Run one deterministic input so the profile has execution to report.

    `input_bin` reads the shared blob `validate.fixture` wrote — the same bytes the
    C++ probe reads. Pass it whenever the outputs will be compared across call
    sites: generating a fresh array here instead, even from a fixed seed, means the
    two runtimes saw different inputs and any output difference says nothing.
    """
    spec = session.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in spec.shape]

    if input_bin is not None:
        data = np.fromfile(input_bin, dtype=np.float32)
        expected = int(np.prod(shape))
        if data.size != expected:
            raise ValueError(
                f"fixture {input_bin} holds {data.size} float32 but the model "
                f"wants {expected} for shape {shape}"
            )
        data = data.reshape(shape)
    else:
        data = np.random.default_rng(seed).standard_normal(shape, dtype=np.float32)

    return session.run(None, {spec.name: data})[0]


def graph_histogram(path: Path) -> dict:
    """Op-type counts and initializer dtypes of a graph on disk."""
    model = onnx.load(str(path), load_external_data=False)
    ops = Counter(node.op_type for node in model.graph.node)
    initializer_dtypes = Counter(
        onnx.TensorProto.DataType.Name(init.data_type).lower()
        for init in model.graph.initializer
    )
    domains = Counter(node.domain or "ai.onnx" for node in model.graph.node)
    return {
        "op_types": dict(sorted(ops.items())),
        "node_count": sum(ops.values()),
        "domains": dict(sorted(domains.items())),
        "initializer_dtypes": dict(sorted(initializer_dtypes.items())),
    }


def profile_coverage(profile_path: Path) -> dict:
    """Per-op execution counts and the dtypes each kernel was actually given.

    ORT writes one JSON array of trace events. Only `cat == "Node"` events carry
    `args.op_name` and `args.input_type_shape`; the rest are session-level spans.
    """
    events = json.loads(profile_path.read_text())
    per_op: dict[str, Counter] = {}
    providers: Counter = Counter()

    for event in events:
        if event.get("cat") != "Node":
            continue
        args = event.get("args") or {}
        op_name = args.get("op_name")
        if not op_name:
            continue
        providers[args.get("provider", "unknown")] += 1

        dtypes: list[str] = []
        for entry in args.get("input_type_shape") or []:
            dtypes.extend(entry.keys())
        per_op.setdefault(op_name, Counter()).update(dtypes or ["<no-input-dtype>"])

    return {
        "profile": str(profile_path),
        "executed_ops": {
            op: dict(sorted(counts.items())) for op, counts in sorted(per_op.items())
        },
        "execution_providers": dict(providers),
        "node_events": sum(sum(c.values()) for c in per_op.values()),
    }


def verdict(optimized: dict, executed: dict) -> dict:
    """Combine the signals into a defensible answer, and show the working.

    `integer_execution` is deliberately conjunctive: an integer kernel must be
    present *and* no heavy float compute may survive. A graph that ran 40 of 52
    convolutions as QLinearConv and 12 as Conv is not "quantized" — it is a
    finding, and the twelve are named.
    """
    op_types = optimized["op_types"]
    integer_kernels = {op: n for op, n in op_types.items() if op in INTEGER_OPS}
    float_compute = {op: n for op, n in op_types.items() if op in FLOAT_COMPUTE_OPS}
    qdq_left = {op: n for op, n in op_types.items() if op in QDQ_OPS}

    executed_ops = executed.get("executed_ops", {})
    integer_kernels_run = {
        op: dtypes for op, dtypes in executed_ops.items() if op in INTEGER_OPS
    }
    float_compute_run = {
        op: dtypes for op, dtypes in executed_ops.items() if op in FLOAT_COMPUTE_OPS
    }
    # A float kernel handed only integer inputs would be a contradiction worth
    # seeing, so dtypes are reported rather than reduced to a boolean.
    float_compute_dtypes = sorted(
        {dtype for dtypes in float_compute_run.values() for dtype in dtypes}
    )

    return {
        "integer_kernels_in_graph": integer_kernels,
        "integer_kernels_executed": integer_kernels_run,
        "float_compute_in_graph": float_compute,
        "float_compute_executed": float_compute_run,
        "float_compute_input_dtypes": float_compute_dtypes,
        "qdq_nodes_remaining": qdq_left,
        "integer_execution": bool(integer_kernels_run) and not float_compute_run,
        "explanation": (
            "integer_execution is true only when at least one integer kernel "
            "actually executed AND no Conv/Gemm/MatMul float kernel executed. "
            "Presence of QuantizeLinear/DequantizeLinear proves nothing on its "
            "own: ORT can dequantize, compute in float, and requantize."
        ),
    }


def analyse(
    model: Path,
    workdir: Path,
    label: str,
    input_bin: Path | None = None,
    intra_op_threads: int = 1,
) -> dict:
    """Full pipeline: optimize, run a fixture, read the profile, decide."""
    workdir.mkdir(parents=True, exist_ok=True)
    optimized_path = workdir / f"{label}.optimized.onnx"
    session, _ = save_optimized_graph(
        model,
        optimized_path,
        profile_prefix=workdir / f"{label}.profile",
        intra_op_threads=intra_op_threads,
    )
    output = run_fixture(session, input_bin=input_bin)
    # end_profiling returns the real filename; ORT appends its own timestamp, so
    # guessing the path from the prefix would break on the next ORT release.
    profile_file = Path(session.end_profiling())

    optimized = graph_histogram(optimized_path)
    exported = graph_histogram(model)
    executed = profile_coverage(profile_file)

    return {
        "label": label,
        "model": str(model),
        "onnxruntime_version": ort.__version__,
        "exported_graph": exported,
        "optimized_graph": optimized,
        "optimized_graph_path": str(optimized_path),
        "execution": executed,
        "input_fixture": str(input_bin) if input_bin else "<generated, seed 0>",
        "output_summary": {
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "mean": float(output.mean()),
            "std": float(output.std()),
            # The class, which must match across call sites even though the logits
            # differ in their last bits.
            "argmax": int(output.argmax()),
        },
        "verdict": verdict(optimized, executed),
    }


def analyse_artifacts(
    exported: Path, optimized: Path, profile: Path, label: str
) -> dict:
    """Reach the same verdict from files another call site produced.

    The C++ probe runs inside the target container, which has no onnx/onnxruntime
    Python. It writes the optimized graph and the profile; this reads them. The
    verdict logic is therefore shared between the Python and C++ evidence rather
    than reimplemented in C++ — asking the question twice is only worth anything
    if both askers mean the same thing by it.
    """
    optimized_graph = graph_histogram(optimized)
    executed = profile_coverage(profile)
    return {
        "label": label,
        "model": str(exported),
        "call_site": "c++",
        "exported_graph": graph_histogram(exported),
        "optimized_graph": optimized_graph,
        "optimized_graph_path": str(optimized),
        "execution": executed,
        "verdict": verdict(optimized_graph, executed),
    }


def main() -> int:
    """Exit 0 = integer execution, 2 = ran but not integer, 1 = could not run.

    Three codes rather than two, because "the model does not execute as integer"
    and "the analysis crashed" are different findings and shell `$?` cannot tell
    them apart otherwise. This is not theoretical: an ORT load failure exited 1
    during A3, the report file was left holding the *previous* run's verdict, and
    the stale numbers looked like a plausible result.

    A model ORT refuses to load is evidence, so the report is written either way.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--workdir", type=Path, help="Run a session here (Python).")
    parser.add_argument(
        "--from-artifacts",
        action="store_true",
        help="Do not run a session; judge the optimized graph and profile that "
        "another call site (the C++ probe) already produced.",
    )
    parser.add_argument("--optimized", type=Path, help="With --from-artifacts.")
    parser.add_argument("--profile", type=Path, help="With --from-artifacts.")
    parser.add_argument(
        "--input-bin",
        type=Path,
        help="Shared fixture blob to run, the same one the C++ probe reads. "
        "Required for any cross-call-site output comparison to mean anything.",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if args.from_artifacts:
        if not (args.optimized and args.profile):
            parser.error("--from-artifacts requires --optimized and --profile")
    elif not args.workdir:
        parser.error("--workdir is required unless --from-artifacts is given")

    def emit(report: dict) -> None:
        print(json.dumps(report, indent=2))
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
            print(f"wrote {args.report}")

    try:
        report = (
            analyse_artifacts(args.model, args.optimized, args.profile, args.label)
            if args.from_artifacts
            else analyse(args.model, args.workdir, args.label, input_bin=args.input_bin)
        )
    except Exception as exc:
        emit(
            {
                "label": args.label,
                "model": str(args.model),
                "onnxruntime_version": ort.__version__,
                "error": f"{type(exc).__name__}: {exc}",
                "verdict": {
                    "integer_execution": False,
                    "explanation": "ONNX Runtime could not load or run this model.",
                },
            }
        )
        return 1

    emit(report)
    return 0 if report["verdict"]["integer_execution"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
