#!/usr/bin/env python3
"""Initial ORT Python vs ORT C++ parity, in two layers (PLAN C4, DESIGN §10).

Both sides run the *same pinned ORT 1.27.0* — pins.env holds the Python wheel to
the C++ tarball's version precisely so this comparison measures our two call
sites and never two releases. Two layers, because they isolate different wiring:

- **logits layer** — `ort_probe` (C++) and `InferenceSession` (Python) fed the
  identical `.bin` tensors P1 dumped. No preprocessing, no policy: pure
  session wiring. Registered gates: max abs <= 1e-4, identical argmax.
- **decision layer** — the real `infer` CLI (decode -> preprocess -> ORT ->
  policy, with the re-bound policy the C++ loader now accepts) against a
  Python recomputation of the same pipeline on the same image. Same logits
  gate, plus identical top-1 and identical SHUTTER_TRIGGER under the 1e-4
  threshold carve-out. This is the first end-to-end run of the deployable
  artifact chain: ONNX + class map + calibrated policy, all hash-bound.

"Initial" is P2's word for it: corpus-scale C++ parity is P4's job with the
dataset runner; this proves the seam before anything is built on it.

Usage (normally via scripts/run_c4_parity.sh ort <run_dir>):
    python -m wildlife_trigger.validate.ort_cpp_parity \
        --run results/training/c2/<run_id> \
        --cpp-logits-dir results/parity/<run_id>/ort/logits \
        --p1-dir results/parity/<run_id>/p1 \
        --infer-dir results/parity/<run_id>/ort \
        --policy artifacts/policies/bobcat_v1.json \
        --output results/parity/<run_id>/p_ort_cpp.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from ..data.preprocess import PreprocessConfig, preprocess_file
from ..runs import atomic_write_json, resolve_run_id, sha256_file

# Registered in DESIGN §10's P2 amendment (2026-07-16).
LOGITS_MAX_ABS = 1e-4
DECISION_CARVE_OUT = 1e-4


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def compare_logits_layer(
    session: ort.InferenceSession,
    p1_dir: Path,
    cpp_logits_dir: Path,
    shape: tuple[int, ...],
) -> list[dict]:
    """C++ ort_probe vs Python InferenceSession on identical input tensors."""
    results = []
    for cpp_path in sorted(cpp_logits_dir.glob("*.bin")):
        name = cpp_path.stem
        tensor = np.fromfile(p1_dir / f"{name}.fused.bin", dtype=np.float32).reshape(shape)
        cpp_logits = np.fromfile(cpp_path, dtype=np.float32)
        (python_logits,) = session.run(None, {"input": tensor[None, ...]})
        python_logits = python_logits[0]
        if cpp_logits.shape != python_logits.shape:
            raise RuntimeError(
                f"{cpp_path} holds {cpp_logits.shape} logits, python produced "
                f"{python_logits.shape}; these are not the same model"
            )
        max_abs = float(np.abs(cpp_logits - python_logits).max())
        failures = []
        if max_abs > LOGITS_MAX_ABS:
            failures.append(f"logits max abs {max_abs:.2e} exceeds the gate")
        if int(cpp_logits.argmax()) != int(python_logits.argmax()):
            failures.append("argmax differs between the two ORT call sites")
        results.append(
            {
                "fixture": name,
                "logits_max_abs": max_abs,
                "argmax_match": int(cpp_logits.argmax()) == int(python_logits.argmax()),
                "passed": not failures,
                "failures": failures,
            }
        )
    if not results:
        raise RuntimeError(f"no C++ logits found under {cpp_logits_dir}")
    return results


def compare_decision_layer(
    session: ort.InferenceSession,
    infer_dir: Path,
    config: PreprocessConfig,
    class_names: list[str],
    threshold: float,
    image_root: Path,
) -> list[dict]:
    """The real `infer` CLI against a Python recomputation of the same pipeline."""
    bobcat = class_names.index("bobcat")
    results = []
    for infer_path in sorted(infer_dir.glob("infer_*.json")):
        record = json.loads(infer_path.read_text())
        cpp_logits = np.array(record["logits"], dtype=np.float32)

        image = record["image"]
        image_path = Path(image[len("/work/") :] if image.startswith("/work/") else image)
        tensor, _ = preprocess_file(image_root / image_path, config)
        (python_logits,) = session.run(None, {"input": tensor[None, ...]})
        python_logits = python_logits[0]

        python_probability = float(softmax(python_logits)[bobcat])
        python_fire = python_probability >= threshold
        cpp_fire = bool(record["decision"]["SHUTTER_TRIGGER"])
        near_threshold = abs(python_probability - threshold) <= DECISION_CARVE_OUT

        max_abs = float(np.abs(cpp_logits - python_logits).max())
        failures = []
        if max_abs > LOGITS_MAX_ABS:
            failures.append(f"logits max abs {max_abs:.2e} exceeds the gate")
        if record["decision"]["top1"]["index"] != int(python_logits.argmax()):
            failures.append("top-1 differs between C++ CLI and python ORT")
        if cpp_fire != python_fire and not near_threshold:
            failures.append(
                f"SHUTTER_TRIGGER differs outside the carve-out "
                f"(python p={python_probability:.6f})"
            )
        results.append(
            {
                "fixture": infer_path.stem,
                "image": str(image_path),
                "logits_max_abs": max_abs,
                "cpp_shutter_trigger": cpp_fire,
                "python_shutter_trigger": python_fire,
                "python_bobcat_probability": python_probability,
                "within_threshold_tolerance": near_threshold,
                "policy_id": record["policy_id"],
                "model_sha256": record["model_sha256"],
                "passed": not failures,
                "failures": failures,
            }
        )
    if not results:
        raise RuntimeError(f"no infer_*.json found under {infer_dir}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--p1-dir", required=True, type=Path)
    parser.add_argument("--cpp-logits-dir", required=True, type=Path)
    parser.add_argument("--infer-dir", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--image-root", type=Path, default=Path("."),
                        help="root the infer JSONs' /work-relative image paths resolve against")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    history = json.loads((args.run / "history.json").read_text())
    onnx_path = args.onnx or args.run / f"{history['run_name']}.onnx"
    onnx_sha256 = sha256_file(onnx_path)

    policy = json.loads(args.policy.read_text())
    if policy.get("model_sha256") != onnx_sha256:
        raise RuntimeError(
            "the policy is not bound to this ONNX — run the decision layer only "
            "after rebind_policy, or the C++ side would have refused what the "
            "Python side quietly accepted"
        )
    (target,) = [t for t in policy["targets"] if t["class"] == "bobcat"]

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    config = PreprocessConfig(
        width=history["config"]["width"], height=history["config"]["height"]
    )
    shape = (3, config.height, config.width)

    logits_layer = compare_logits_layer(session, args.p1_dir, args.cpp_logits_dir, shape)
    decision_layer = compare_decision_layer(
        session, args.infer_dir, config, history["class_names"],
        float(target["threshold"]), args.image_root,
    )

    report = {
        "gate": "initial ORT python-vs-C++ parity (DESIGN 10, registered 2026-07-16)",
        "run_id": resolve_run_id(args.run, history["run_name"]),
        "onnx_sha256": onnx_sha256,
        "policy_id": policy["policy_id"],
        "ort_version_python": ort.__version__,
        "tolerances": {
            "logits_max_abs": LOGITS_MAX_ABS,
            "decision_carve_out": DECISION_CARVE_OUT,
        },
        "logits_layer": {
            "fixtures": len(logits_layer),
            "worst_max_abs": max(r["logits_max_abs"] for r in logits_layer),
            "results": logits_layer,
        },
        "decision_layer": {
            "fixtures": len(decision_layer),
            "worst_max_abs": max(r["logits_max_abs"] for r in decision_layer),
            "results": decision_layer,
        },
        "verdict": {
            "passed": all(r["passed"] for r in logits_layer + decision_layer),
            "failed_fixtures": [
                r["fixture"] for r in logits_layer + decision_layer if not r["passed"]
            ],
        },
    }
    atomic_write_json(args.output, report)
    print(
        f"ORT py-vs-cpp {'PASSED' if report['verdict']['passed'] else 'FAILED'} "
        f"(logits layer: {len(logits_layer)} tensors, worst "
        f"{report['logits_layer']['worst_max_abs']:.2e}; decision layer: "
        f"{len(decision_layer)} images, worst "
        f"{report['decision_layer']['worst_max_abs']:.2e})"
    )
    print(f"wrote {args.output}")
    return 0 if report["verdict"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
