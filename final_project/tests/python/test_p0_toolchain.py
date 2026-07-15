"""Tests for the P0 toolchain path: opset contract, QDQ repair, verdict logic (A3).

These assert the properties P0 depends on rather than the shape of the API. Each
one corresponds to a way A3 could have passed while being wrong, and most of them
correspond to a way it actually did during development:

  - the opset guard must reject opset 9 and any off-contract opset;
  - the verdict must not call a QDQ-shaped float graph "integer execution";
  - the verdict must not call an FP32 graph integer (the negative control);
  - the scalar fix must repair per-tensor scales and leave per-channel alone.

The heavy end-to-end evidence (real exports, real ORT sessions, QEMU) belongs to
`scripts/run_p0_spike.sh`, which needs gx10 and the target container. These stay
fast and hermetic so they can run anywhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from wildlife_trigger.models import export
from wildlife_trigger.optimize.qdq_scalar import (
    PerChannelAmbiguity,
    scalarize_per_tensor_qdq,
)
from wildlife_trigger.validate import ort_coverage


# --- the opset contract ------------------------------------------------------


def test_opset_9_is_rejected_by_name() -> None:
    # DESIGN §8 forbids the legacy course spike's opset specifically, and the
    # message has to say why: "too low" would invite someone to try 10.
    with pytest.raises(export.OpsetContractError, match="forbidden"):
        export.check_opset_request(9)


def test_off_contract_opset_is_rejected_even_though_it_is_newer() -> None:
    # 13 supports QDQ perfectly well. It is still wrong, because DESIGN §8 requires
    # one opset across M0-M4 so candidates stay comparable.
    with pytest.raises(export.OpsetContractError, match="not the P0 contract"):
        export.check_opset_request(13)


def test_contract_opset_is_accepted() -> None:
    export.check_opset_request(export.P0_OPSET)


def test_verify_exported_opset_catches_a_graph_that_disagrees(tmp_path: Path) -> None:
    """The request is an argument; the artifact is the evidence.

    An exporter that silently produced a different opset than asked would otherwise
    be discovered on the Pi.
    """
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    path = tmp_path / "opset13.onnx"
    onnx.save(model, str(path))

    with pytest.raises(export.OpsetContractError, match="carries default-domain"):
        export.verify_exported_opset(path, expected=17)


# --- the verdict -------------------------------------------------------------


def test_qdq_shaped_float_graph_is_not_integer_execution() -> None:
    """The exact A3 failure: QDQ everywhere, every convolution still float.

    This is what DESIGN §8.2 means by "a float graph carrying rounded weights". A
    verdict that counted QuantizeLinear nodes would call this quantized.
    """
    optimized = {
        "op_types": {
            "FusedConv": 45,
            "Conv": 5,
            "QLinearConv": 2,
            "QuantizeLinear": 51,
            "DequantizeLinear": 136,
        }
    }
    executed = {
        "executed_ops": {
            "FusedConv": {"float": 145},
            "Conv": {"float": 15},
            "QLinearConv": {"int8": 8},
        }
    }

    result = ort_coverage.verdict(optimized, executed)
    assert result["integer_execution"] is False
    # The surviving float kernels must be named, not merely counted: "which ones"
    # is the whole finding.
    assert "FusedConv" in result["float_compute_executed"]


def test_fp32_graph_is_not_integer_execution() -> None:
    # The negative control. A verdict stuck at True would make every other row in
    # the P0 gate meaningless.
    optimized = {"op_types": {"FusedConv": 45, "Conv": 7, "Gemm": 1}}
    executed = {"executed_ops": {"FusedConv": {"float": 145}, "Gemm": {"float": 2}}}

    assert ort_coverage.verdict(optimized, executed)["integer_execution"] is False


def test_fully_quantized_graph_is_integer_execution() -> None:
    optimized = {
        "op_types": {"QLinearConv": 52, "QLinearAdd": 10, "QGemm": 1, "Flatten": 1}
    }
    executed = {
        "executed_ops": {
            "QLinearConv": {"int8": 208, "int32": 52},
            "QGemm": {"int8": 4},
        }
    }

    assert ort_coverage.verdict(optimized, executed)["integer_execution"] is True


def test_integer_kernels_alone_do_not_prove_integer_execution() -> None:
    """A mixed graph is a finding, not a pass.

    40 of 52 convolutions on the fast path is exactly the kind of result that gets
    rounded up to "quantized" in a report. It is not.
    """
    optimized = {"op_types": {"QLinearConv": 40, "Conv": 12}}
    executed = {
        "executed_ops": {"QLinearConv": {"int8": 40}, "Conv": {"float": 12}}
    }

    assert ort_coverage.verdict(optimized, executed)["integer_execution"] is False


# --- the QDQ scalar repair ---------------------------------------------------


def _qdq_model(scale_shape: list[int], with_axis: bool) -> onnx.ModelProto:
    scale = numpy_helper.from_array(
        np.full(scale_shape or [], 0.02, dtype=np.float32), "scale"
    )
    zero_point = numpy_helper.from_array(
        np.zeros(scale_shape or [], dtype=np.int8), "zp"
    )
    kwargs = {"axis": 0} if with_axis else {}
    node = helper.make_node("QuantizeLinear", ["x", "scale", "zp"], ["y"], **kwargs)
    graph = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("y", TensorProto.INT8, [1, 2])],
        initializer=[scale, zero_point],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def test_per_tensor_scale_of_shape_one_becomes_a_scalar(tmp_path: Path) -> None:
    """torch exports [1]; ORT demands rank 0 and refuses to load otherwise."""
    source = tmp_path / "in.onnx"
    onnx.save(_qdq_model([1], with_axis=False), str(source))

    result = scalarize_per_tensor_qdq(source, tmp_path / "out.onnx")
    assert result["scalarized_tensors"] == 2  # scale and zero_point

    fixed = onnx.load(str(tmp_path / "out.onnx"))
    for init in fixed.graph.initializer:
        assert list(init.dims) == [], f"{init.name} is still rank {len(init.dims)}"


def test_scalarizing_changes_rank_but_never_value(tmp_path: Path) -> None:
    source = tmp_path / "in.onnx"
    onnx.save(_qdq_model([1], with_axis=False), str(source))
    scalarize_per_tensor_qdq(source, tmp_path / "out.onnx")

    fixed = onnx.load(str(tmp_path / "out.onnx"))
    scale = next(i for i in fixed.graph.initializer if i.name == "scale")
    # Silently rescaling a quantization parameter would be undetectable downstream.
    assert numpy_helper.to_array(scale).item() == pytest.approx(0.02)


def test_per_channel_scales_are_left_alone(tmp_path: Path) -> None:
    """Squeezing one would turn per-channel into per-tensor and lose accuracy."""
    source = tmp_path / "in.onnx"
    onnx.save(_qdq_model([4], with_axis=True), str(source))

    result = scalarize_per_tensor_qdq(source, tmp_path / "out.onnx")
    assert result["scalarized_tensors"] == 0
    assert result["per_channel_tensors_left_alone"] == 2

    fixed = onnx.load(str(tmp_path / "out.onnx"))
    scale = next(i for i in fixed.graph.initializer if i.name == "scale")
    assert list(scale.dims) == [4]


def test_single_channel_per_channel_refuses_to_guess(tmp_path: Path) -> None:
    """Shape [1] plus an explicit axis is genuinely ambiguous.

    MobileNetV2 has no single-channel convolution so this never fires today. If a
    future architecture makes it fire, it must stop rather than silently change the
    quantization scheme.
    """
    source = tmp_path / "in.onnx"
    onnx.save(_qdq_model([1], with_axis=True), str(source))

    with pytest.raises(PerChannelAmbiguity, match="refusing to guess"):
        scalarize_per_tensor_qdq(source, tmp_path / "out.onnx")
