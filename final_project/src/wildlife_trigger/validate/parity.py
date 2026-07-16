#!/usr/bin/env python3
"""P2 — FP32 model parity: PyTorch vs ONNX Runtime on the frozen fixture set.

The question this gate answers: is the exported graph *the calibrated model*, to
within numeric noise that cannot change a decision — and where it legitimately
could (a probability within 1e-4 of the threshold), is every such frame named
rather than averaged away?

Both frameworks are fed the **same preprocessed tensor** (one `preprocess_file`
call per fixture), so the comparison isolates the model. Preprocessing has its
own gate (P1), and mixing the two would let a preprocessing wobble masquerade as
kernel drift.

## The pre-registered gates (DESIGN §10, C4 amendment, 2026-07-16)

- logits: max abs <= 5e-4 per fixture (expected ~1e-5; the headroom is for
  kernel reassociation, not for bugs);
- top-1: identical on every fixture, no carve-out;
- bobcat fire/no-fire at the calibrated threshold: identical, except fixtures
  whose torch probability lies within 1e-4 of the threshold — those are listed
  by image id as within-tolerance-of-threshold.

## The consistency guard, as corrected the same day (issue #30)

Its original form — torch-CPU probability within 1e-3 of the committed
predictions.npz — FAILED on 53/200 fixtures and was mis-specified, not tight:
the npz holds the checkpoint's output under cuDNN TF32 at the loader's batch
size (torch 2.11's default), while the exported graph is true FP32. Same
weights, different convolution arithmetic. The corrected guard, per the DESIGN
§10 verdict:

- weight identity is proven by the checkpoint hash chain (unchanged);
- where CUDA is available, the npz value is reproduced **under its own regime**
  (CUDA, cuDNN TF32 on, the recorded batch size) to <= 1e-4 on the
  worst-gapped sample of fixtures;
- the torch-CPU vs npz gap is *reported* per fixture — it is the measured
  calibration-vs-deployment numeric gap, the number issue #30 is about — and
  is not gated.

Usage:
    python -m wildlife_trigger.validate.parity \
        --run results/training/c2/c2_m0_fp32_seed42_20260716T061203Z \
        --fixtures tests/fixtures/p2_fixtures.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from ..data.preprocess import PreprocessConfig, preprocess_file
from ..models.mobilenet import build_mobilenet_v2
from ..runs import BEST_CHECKPOINT, atomic_write_json, resolve_run_id, sha256_file

# The registered gates. Changing any of these is a DESIGN §10 amendment — the
# npz guard already required one (issue #30), and its record is in the DESIGN.
LOGITS_MAX_ABS = 5e-4
DECISION_CARVE_OUT = 1e-4
NPZ_REGIME_MAX_ABS = 1e-4  # reproduced under the npz's own regime, not across regimes
NPZ_REGIME_SAMPLE = 8


def load_model(run_dir: Path, history: dict) -> torch.nn.Module:
    hashes = json.loads((run_dir / "hashes.json").read_text())
    checkpoint_path = run_dir / BEST_CHECKPOINT
    measured = sha256_file(checkpoint_path)
    if measured != hashes["checkpoint:best"]["sha256"]:
        raise RuntimeError(
            f"{checkpoint_path} does not hash to the run's record; whatever it "
            "holds now is not the calibrated model"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_mobilenet_v2(num_classes=len(history["class_names"]), pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def npz_probabilities(run_dir: Path, bobcat: int) -> dict[str, float]:
    """image_id -> the committed (CUDA) bobcat probability, both domains."""
    data = np.load(run_dir / "predictions.npz", allow_pickle=False)
    lookup: dict[str, float] = {}
    for domain in ("cis_val_clean", "trans_val"):
        probabilities = data[f"{domain}/probabilities"][:, bobcat]
        for image_id, probability in zip(data[f"{domain}/image_ids"], probabilities):
            lookup[str(image_id)] = float(probability)
    return lookup


def compare_fixture(
    fixture: dict,
    image_path: Path,
    config: PreprocessConfig,
    model: torch.nn.Module,
    session: ort.InferenceSession,
    class_names: list[str],
    threshold: float,
    npz_lookup: dict[str, float],
) -> dict:
    measured = sha256_file(image_path)
    if measured != fixture["sha256"]:
        raise RuntimeError(
            f"{image_path} does not hash to the frozen fixture record; P2 would "
            "be comparing different bytes than it froze"
        )

    tensor, _ = preprocess_file(image_path, config)
    batch = tensor[None, ...]

    with torch.inference_mode():
        torch_logits = model(torch.from_numpy(batch)).numpy()[0]
    (ort_logits,) = session.run(None, {"input": batch})
    ort_logits = ort_logits[0]

    bobcat = class_names.index("bobcat")
    torch_probability = float(torch.softmax(torch.from_numpy(torch_logits), 0)[bobcat])
    ort_probability = float(torch.softmax(torch.from_numpy(ort_logits), 0)[bobcat])

    logits_max_abs = float(np.abs(torch_logits - ort_logits).max())
    torch_top1 = class_names[int(torch_logits.argmax())]
    ort_top1 = class_names[int(ort_logits.argmax())]
    torch_fire = torch_probability >= threshold
    ort_fire = ort_probability >= threshold
    near_threshold = abs(torch_probability - threshold) <= DECISION_CARVE_OUT
    npz_gap = abs(torch_probability - npz_lookup[fixture["image_id"]])

    failures = []
    if logits_max_abs > LOGITS_MAX_ABS:
        failures.append(f"logits max abs {logits_max_abs:.2e} exceeds the gate")
    if torch_top1 != ort_top1:
        failures.append(f"top-1 differs: torch={torch_top1} ort={ort_top1}")
    if torch_fire != ort_fire and not near_threshold:
        failures.append(
            f"decision differs outside the carve-out: torch p={torch_probability:.6f}"
        )
    # npz_gap is reported, never gated: it is the TF32-vs-FP32 regime distance
    # (issue #30), not a defect of this fixture.

    return {
        "image_id": fixture["image_id"],
        "split": fixture["split"],
        "reason": fixture["reason"],
        "logits_max_abs": logits_max_abs,
        "torch_bobcat_probability": torch_probability,
        "ort_bobcat_probability": ort_probability,
        "top1": torch_top1,
        "top1_match": torch_top1 == ort_top1,
        "fire_match": torch_fire == ort_fire,
        "within_threshold_tolerance": near_threshold,
        "npz_probability_gap": npz_gap,
        "passed": not failures,
        "failures": failures,
    }


def verify_npz_regime(
    model: torch.nn.Module,
    results: list[dict],
    fixtures_by_id: dict[str, dict],
    images_dir: Path,
    config: PreprocessConfig,
    npz_lookup: dict[str, float],
    bobcat: int,
    batch_size: int,
    sample: int = NPZ_REGIME_SAMPLE,
) -> dict:
    """Reproduce the committed npz values under their own numeric regime.

    CUDA, cuDNN TF32 on, the recorded batch size — the conditions
    `dump_predictions` actually ran under (issue #30). The sample is the
    worst-gapped fixtures, because those are precisely the ones a wrong-weights
    explanation would have to account for: if TF32 batching explains even them,
    the identity question is closed behaviourally as well as cryptographically.
    """
    if not torch.cuda.is_available():
        return {
            "status": "skipped_no_cuda",
            "note": "regime reproduction needs the device the npz was written on",
        }

    ranked = sorted(results, key=lambda r: -r["npz_probability_gap"])[:sample]
    saved_tf32 = torch.backends.cudnn.allow_tf32
    cuda_model = model.cuda()
    torch.backends.cudnn.allow_tf32 = True
    try:
        checks = []
        for row in ranked:
            fixture = fixtures_by_id[row["image_id"]]
            tensor, _ = preprocess_file(images_dir / fixture["file_name"], config)
            batch = torch.from_numpy(tensor[None, ...]).repeat(batch_size, 1, 1, 1)
            with torch.inference_mode():
                logits = cuda_model(batch.cuda())[0]
            reproduced = float(torch.softmax(logits, 0)[bobcat])
            gap = abs(reproduced - npz_lookup[row["image_id"]])
            checks.append(
                {
                    "image_id": row["image_id"],
                    "cpu_gap": row["npz_probability_gap"],
                    "regime_gap": gap,
                    "passed": gap <= NPZ_REGIME_MAX_ABS,
                }
            )
    finally:
        torch.backends.cudnn.allow_tf32 = saved_tf32
        model.cpu()

    return {
        "status": "ran",
        "regime": f"cuda, cudnn_tf32=on, batch={batch_size}",
        "max_abs_allowed": NPZ_REGIME_MAX_ABS,
        "sample": len(checks),
        "worst_regime_gap": max(c["regime_gap"] for c in checks),
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--onnx", type=Path,
                        help="default: <run>/<run_name>.onnx, the export tool's output")
    parser.add_argument("--images-dir", type=Path,
                        help="raw image root; default: the run config's images_dir")
    parser.add_argument("--output", type=Path,
                        help="default: results/parity/<run_id>/p2_fp32.json")
    args = parser.parse_args()

    history = json.loads((args.run / "history.json").read_text())
    class_names = history["class_names"]
    frozen = json.loads(args.fixtures.read_text())
    threshold = frozen["threshold"]
    run_id = resolve_run_id(args.run, history["run_name"])
    if frozen["run_id"] != run_id:
        raise RuntimeError(
            f"fixtures were frozen from {frozen['run_id']} but this run is "
            f"{run_id}; P2 must compare the model its fixtures came from"
        )

    onnx_path = args.onnx or args.run / f"{history['run_name']}.onnx"
    model = load_model(args.run, history)
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=ort.SessionOptions(),
        providers=["CPUExecutionProvider"],
    )

    config = PreprocessConfig(
        width=history["config"]["width"], height=history["config"]["height"]
    )
    images_dir = args.images_dir or Path(history["config"]["images_dir"])
    npz_lookup = npz_probabilities(args.run, class_names.index("bobcat"))

    results = [
        compare_fixture(
            fixture, images_dir / fixture["file_name"], config, model, session,
            class_names, threshold, npz_lookup,
        )
        for fixture in frozen["fixtures"]
    ]

    carved = [r["image_id"] for r in results if r["within_threshold_tolerance"]]
    regime = verify_npz_regime(
        model,
        results,
        {f["image_id"]: f for f in frozen["fixtures"]},
        images_dir,
        config,
        npz_lookup,
        class_names.index("bobcat"),
        batch_size=history["config"]["batch_size"],
    )
    npz_gaps = sorted(r["npz_probability_gap"] for r in results)
    report = {
        "gate": "P2 FP32 parity (DESIGN 10, tolerances registered 2026-07-16; "
        "npz guard corrected same day, issue #30)",
        "run_id": run_id,
        "onnx": {"path": str(onnx_path), "sha256": sha256_file(onnx_path)},
        "checkpoint_sha256": json.loads((args.run / "hashes.json").read_text())[
            "checkpoint:best"
        ]["sha256"],
        "fixtures_sha256": sha256_file(args.fixtures),
        "threshold": threshold,
        "tolerances": {
            "logits_max_abs": LOGITS_MAX_ABS,
            "decision_carve_out": DECISION_CARVE_OUT,
            "npz_regime_max_abs": NPZ_REGIME_MAX_ABS,
        },
        "versions": {
            "torch": torch.__version__,
            "onnxruntime": ort.__version__,
        },
        "fixtures": len(results),
        "worst_logits_max_abs": max(r["logits_max_abs"] for r in results),
        # The measured calibration-vs-deployment numeric distance (issue #30):
        # reported with the evidence, gated only under its own regime below.
        "npz_probability_gap": {
            "worst": npz_gaps[-1],
            "median": npz_gaps[len(npz_gaps) // 2],
        },
        "npz_regime_check": regime,
        "within_threshold_tolerance": carved,
        "verdict": {
            "passed": all(r["passed"] for r in results)
            and regime.get("passed", True),
            "failed_fixtures": [r["image_id"] for r in results if not r["passed"]],
        },
        "results": results,
    }
    output = args.output or Path("results/parity") / run_id / "p2_fp32.json"
    atomic_write_json(output, report)

    regime_line = (
        f"regime check {regime['status']}"
        if regime["status"] != "ran"
        else f"regime worst {regime['worst_regime_gap']:.2e} "
        f"({'ok' if regime['passed'] else 'FAILED'})"
    )
    print(
        f"P2 {'PASSED' if report['verdict']['passed'] else 'FAILED'} "
        f"({report['fixtures']} fixtures; worst logits gap "
        f"{report['worst_logits_max_abs']:.2e}; npz gap median "
        f"{report['npz_probability_gap']['median']:.2e} worst "
        f"{report['npz_probability_gap']['worst']:.2e}; {regime_line}; "
        f"{len(carved)} within threshold tolerance)"
    )
    print(f"wrote {output}")
    return 0 if report["verdict"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
