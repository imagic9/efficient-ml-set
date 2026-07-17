#!/usr/bin/env python3
"""D2: train one M2 QAT arm from the M0 checkpoint (DESIGN §8.2, PLAN D2).

The quantization structure is the P0-pinned contract implemented in
`optimize.qat` (output-side QDQ, ReLU6 absorbed exactly, per-channel S8
weights, per-tensor S8 activations). What this module adds is the *real*
training path the spike deliberately did not have:

- initialization from **M0's selected checkpoint**, hash-verified — never M1
  (DESIGN §8.2), never the ImageNet factory weights;
- BN folded **at initialization** from the checkpoint's statistics — frozen
  from step 0, the registered reading of §8.2's freeze bullet
  (`results/optimize/m2_qat/preregistration.md` §2);
- observers initialized on the **frozen D1 calibration manifest** (real
  training frames, not noise), updating through epoch 1, frozen after;
- the frozen §7.2 data/loss contract (weighted CE, ignore_index=-1, the same
  augmentation and loaders `train.py` uses), **AMP off** — fake-quant in fp16
  would train scales against arithmetic the deployed INT8 graph never performs;
- per-epoch validation under fake-quant, best epoch by the frozen §7.2 rule;
- export of the best epoch to a candidate directory speaking the same file
  contract D1 established, so `evaluate_onnx` → `calibrate_candidate` →
  `select_ptq` → P3/P4 run downstream unchanged.

One invocation trains ONE learning-rate arm; the pre-registered arms are
1e-5 / 3e-5 / 5e-5 and nothing outside DESIGN §8.2's documented range.

Usage (gx10):
    python -m wildlife_trigger.optimize.qat_train \
        --config configs/optimize/m2_qat.yaml --lr 1e-5
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from .. import runs
from ..data.dataset import WildlifeDataset, class_weights, load_class_names
from ..data.preprocess import PreprocessConfig
from ..metrics import is_better_checkpoint
from ..models.export import describe, export_onnx
from ..models.mobilenet import build_mobilenet_v2, example_input
from ..train import build_datasets, evaluate, score_of
from ..validate import ort_coverage
from .fold_qdq import fold_qdq_weights
from .prune import apply_widths
from .qat import (
    build_qat_model,
    set_export_mode,
    set_observers,
    verify_relu6_removal_is_exact,
)
from .qdq_scalar import scalarize_per_tensor_qdq

MIN_LR, MAX_LR = 1e-5, 5e-5  # DESIGN §8.2's documented range; nothing outside it


@dataclass
class QatConfig:
    """Duck-types the fields `train.build_datasets` reads, plus the QAT recipe."""

    # Provenance — all hash-pinned, all refused on mismatch.
    source_run_id: str = ""
    source_checkpoint: str = ""
    source_checkpoint_sha256: str = ""
    calibration_manifest: str = "data/manifests/calibration_1024.jsonl"
    calibration_manifest_sha256: str = ""

    # M4 (D5): when set, the source is a *pruned* MobileNetV2 (the M3 c30
    # checkpoint), so the architecture is rebuilt with these expansion widths
    # before the weights load and before the QAT structure is inserted. Empty
    # for M2, whose source is the unpruned M0 checkpoint.
    pruned_widths: dict = field(default_factory=dict)

    # Candidate identity — parameterized so M4 differs from M2 in the evidence
    # without a second copy of the trainer. M2's defaults keep D2 byte-stable.
    candidate_prefix: str = "d2_m2_qat"
    candidate_kind: str = "int8_qat"
    candidate_model_id: str = "M2-candidate"
    candidate_design: str = "8.2"
    # The run-directory / history run_name stem. Defaults to M2's so D2 stays
    # byte-stable; M4 sets "m4_qat". Nothing downstream keys on it (candidate_id
    # is the key), but the evidence should not call an M4 run "m2_qat".
    run_name_stem: str = "m2_qat"

    # Data (the frozen §7.2 contract; same defaults as TrainConfig)
    manifests_dir: str = "data/manifests"
    images_dir: str = "data/raw/extracted/eccv_18_all_images_sm"
    supplement_manifest: str | None = "data/manifests/cct_empty_train_v1.jsonl"
    supplement_dir: str = "data/images/empty_supplement"
    cache_dir: str = "data/cache"
    classes_config: str = "configs/data/classes.yaml"
    width: int = 256
    height: int = 192
    exclude_empty_class: bool = False  # read by build_datasets; never true for M2

    # The registered recipe (preregistration §2)
    seed: int = 42
    epochs: int = 6
    batch_size: int = 64
    weight_decay: float = 1e-4
    workers: int = 8
    observer_freeze_after_epoch: int = 1

    output_root: str = "results/optimize/m2_qat"
    phase: str = "D2"


def load_config(path: Path) -> QatConfig:
    raw = yaml.safe_load(path.read_text())
    unknown = [k for k in raw if k not in QatConfig.__dataclass_fields__]
    if unknown:
        raise ValueError(f"{path} has unknown keys: {unknown}")
    config = QatConfig(**raw)
    for field in ("source_run_id", "source_checkpoint", "source_checkpoint_sha256",
                  "calibration_manifest_sha256"):
        if not getattr(config, field):
            raise ValueError(f"{path} must pin {field}")
    return config


def arm_label(lr: float) -> str:
    return f"lr{lr:.0e}".replace("e-0", "e-")


def build_source_architecture(config: QatConfig, class_names: list[str]) -> nn.Module:
    """The FP32 architecture the source weights load into.

    Unpruned for M2 (the M0 checkpoint); the pruned c30 architecture for M4 —
    `apply_widths` reproduces exactly the channel counts the checkpoint carries
    before any weight loads, so the same trainer serves both. The QAT structure
    inserted afterwards (`build_qat_model`) is shape-agnostic: it rewrites
    convolutions by scanning children, not by their channel counts.
    """
    base = build_mobilenet_v2(num_classes=len(class_names), pretrained=False)
    if config.pruned_widths:
        apply_widths(base, dict(config.pruned_widths))
    return base


def load_m0_base(config: QatConfig, class_names: list[str]) -> tuple[nn.Module, dict]:
    """The source checkpoint as a 16-class network, proven by hash.

    Named for its M2 role; M4 passes a pruned source. The hash pin is the
    contract either way — the fake-quant scales are trained against whatever
    these weights are, so the wrong file would poison every scale silently.
    """
    checkpoint_path = Path(config.source_checkpoint)
    measured = runs.sha256_file(checkpoint_path)
    if measured != config.source_checkpoint_sha256:
        raise RuntimeError(
            f"{checkpoint_path} hashes to {measured[:12]}… but the config pins "
            f"{config.source_checkpoint_sha256[:12]}…; QAT initializes from the "
            "pinned checkpoint and nothing else (DESIGN §8.2/§8.4)"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("class_names") and checkpoint["class_names"] != class_names:
        raise RuntimeError(
            "the checkpoint's class order differs from the frozen classes config; "
            "every fake-quant scale would be trained against the wrong animal"
        )
    if config.pruned_widths and checkpoint.get("widths") not in (None, {}):
        if checkpoint["widths"] != dict(config.pruned_widths):
            raise RuntimeError(
                "the config's pruned_widths differ from the checkpoint's own "
                "recorded widths; M4's architecture must be exactly the M3 "
                "checkpoint's, not a re-derived one"
            )
    base = build_source_architecture(config, class_names)
    base.load_state_dict(checkpoint["model"])
    return base, {
        "path": str(checkpoint_path),
        "sha256": measured,
        "epoch": checkpoint.get("epoch"),
        "run_id": checkpoint.get("run_id"),
        "pruned_widths": dict(config.pruned_widths) or None,
    }


def calibrate_on_manifest(
    model: nn.Module, config: QatConfig, class_names: list[str], device: torch.device
) -> dict:
    """Observer warm-up on the frozen D1 manifest — real frames, not noise.

    Observers on, fake-quant off (the spike's rule): ranges must describe the
    FP32 activations the M0 weights actually produce on training data.
    """
    manifest = Path(config.calibration_manifest)
    measured = runs.sha256_file(manifest)
    if measured != config.calibration_manifest_sha256:
        raise RuntimeError(
            f"{manifest} hashes to {measured[:12]}…, not the pinned "
            f"{config.calibration_manifest_sha256[:12]}…; the calibration corpus "
            "is frozen and shared with M1 by registration"
        )
    dataset = WildlifeDataset(
        manifest,
        class_names,
        PreprocessConfig(width=config.width, height=config.height),
        Path(config.images_dir),
        cache_root=Path(config.cache_dir),
        train=False,
        image_root_overrides={"empty_supplement": Path(config.supplement_dir)},
    )
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.workers, pin_memory=True,
    )
    set_observers(model, observe=True, fake_quant=False)
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            model(batch["image"].to(device, non_blocking=True))
    set_observers(model, observe=True, fake_quant=True)  # epoch 1 keeps observing
    return {"manifest": str(manifest), "sha256": measured, "images": len(dataset)}


def train_arm(config: QatConfig, lr: float) -> dict:
    if not (MIN_LR <= lr <= MAX_LR):
        raise ValueError(
            f"lr {lr} is outside DESIGN §8.2's documented range "
            f"[{MIN_LR}, {MAX_LR}]; the pre-registration forbids searching there"
        )

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label = arm_label(lr)

    ctx = runs.RunContext.create(
        phase=config.phase,
        name=f"{config.run_name_stem}_{label}",
        config={**asdict(config), "lr": lr},
        results_root=Path(config.output_root) / "runs",
    )

    class_names = load_class_names(Path(config.classes_config))
    base, checkpoint_info = load_m0_base(config, class_names)

    model, structure = build_qat_model(base=base)
    model.to(device)

    data = build_datasets(config, class_names)
    train_records = [r for part in data["train_parts"] for r in part.records]
    weights = class_weights(train_records, class_names).to(device)

    manifests = Path(config.manifests_dir)
    ctx.record_hashes(
        {
            "manifest:train": manifests / "train.jsonl",
            "manifest:cis_val_clean": manifests / "cis_val_clean.jsonl",
            "manifest:trans_val": manifests / "trans_val.jsonl",
            "manifest:empty_supplement": Path(config.supplement_manifest),
            "manifest:calibration": Path(config.calibration_manifest),
            "config:classes": Path(config.classes_config),
            "checkpoint:m0_source": Path(config.source_checkpoint),
        },
        class_names=class_names,
    )

    calibration = calibrate_on_manifest(model, config, class_names, device)

    # The exactness proof, on the *real* observed ranges — a ReLU6 whose range
    # escaped [0, 6] on this data would make the export non-equivalent. Observers
    # are frozen around the check so its synthetic probes cannot leak into the
    # ranges the manifest just calibrated, then re-opened for epoch 1.
    set_observers(model, observe=False, fake_quant=True)
    equivalence = verify_relu6_removal_is_exact(model)
    if not equivalence["exact"]:
        raise RuntimeError(f"ReLU6 removal is not exact on real ranges: {equivalence}")
    set_observers(model, observe=True, fake_quant=True)

    loaders = {
        "train": DataLoader(
            data["train"], batch_size=config.batch_size, shuffle=True,
            num_workers=config.workers, pin_memory=True, drop_last=True,
            persistent_workers=config.workers > 0,
        ),
        **{
            name: DataLoader(
                dataset, batch_size=config.batch_size, shuffle=False,
                num_workers=config.workers, pin_memory=True,
                persistent_workers=config.workers > 0,
            )
            for name, dataset in data["validation"].items()
        },
    }
    validation_loaders = {k: v for k, v in loaders.items() if k != "train"}

    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-1)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=config.weight_decay
    )
    total_steps = len(loaders["train"]) * config.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=total_steps)

    history: list[dict] = []
    best: dict = {"score": None, "epoch": -1}
    started = time.time()
    step = 0

    for epoch in range(1, config.epochs + 1):
        if epoch == config.observer_freeze_after_epoch + 1:
            # Preregistration §2: ranges observe through epoch 1, frozen after.
            set_observers(model, observe=False, fake_quant=True)

        model.train()
        epoch_loss, batches = 0.0, 0
        for batch in loaders["train"]:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            # No autocast anywhere: AMP is registered OFF for QAT.
            loss = criterion(model(images), targets)
            loss.backward()
            optimiser.step()
            scheduler.step()
            epoch_loss += float(loss.detach())
            batches += 1
            step += 1

        results = evaluate(model, validation_loaders, class_names, device)
        score = score_of(results)
        entry = {
            "epoch": epoch,
            "step": step,
            "lr": lr,
            "train_loss": epoch_loss / max(batches, 1),
            "selection_score": score,
            "cis_val_clean": dict(results["cis_val_clean"]["target"]),
            "trans_val": dict(results["trans_val"]["target"]),
            "macro_f1": {
                d: results[d]["classes"]["support_aware_macro_f1"] for d in results
            },
            "observers_frozen": epoch > config.observer_freeze_after_epoch,
            "elapsed_s": round(time.time() - started, 1),
        }
        history.append(entry)
        print(
            f"epoch {epoch}/{config.epochs}  loss {entry['train_loss']:.4f}  "
            f"F2@0.5 cis {entry['cis_val_clean']['frame_f2']:.4f} "
            f"trans {entry['trans_val']['frame_f2']:.4f}  "
            f"score {score['primary']:.4f}"
        )

        state = {
            "run_id": ctx.run_id,
            "model": model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "epoch": epoch,
            "score": score,
            "class_names": class_names,
            "config": {**asdict(config), "lr": lr},
        }
        runs.save_checkpoint(ctx.checkpoint_path, state)
        if is_better_checkpoint(score, best["score"]):
            best = {"score": score, "epoch": epoch}
            runs.save_checkpoint(ctx.best_checkpoint_path, state)

    history_record = {
        "run_name": f"{config.run_name_stem}_{label}",
        "run_id": ctx.run_id,
        "lr": lr,
        "best_epoch": best["epoch"],
        "best_selection_score": best["score"],
        "class_names": class_names,
        "config": {**asdict(config), "lr": lr},
        "initialized_from": checkpoint_info,
        "qat_structure": structure,
        "observer_calibration": calibration,
        "relu6_removal_equivalence": equivalence,
        "history": history,
    }
    runs.atomic_write_json(ctx.run_dir / "history.json", history_record)

    export_record = export_candidate(config, ctx, label, lr, best, class_names,
                                     checkpoint_info, calibration, structure)
    ctx.finish(status="completed", best_epoch=best["epoch"],
               best_selection_score=best["score"])
    return {**history_record, "export": export_record}


def export_candidate(
    config: QatConfig,
    ctx,
    label: str,
    lr: float,
    best: dict,
    class_names: list[str],
    checkpoint_info: dict,
    calibration: dict,
    structure: dict,
) -> dict:
    """Best epoch -> genuinely quantized ONNX in a D1-contract candidate dir."""
    checkpoint = runs.load_checkpoint(ctx.best_checkpoint_path)
    if checkpoint["epoch"] != best["epoch"]:
        raise RuntimeError(
            f"best.pt holds epoch {checkpoint['epoch']} but the loop selected "
            f"{best['epoch']}; refusing to export a different model than selected"
        )

    base = build_source_architecture(config, class_names)
    model, _ = build_qat_model(base=base)
    model.load_state_dict(checkpoint["model"])
    model.eval().cpu()
    # A best epoch inside the observation window carries observer_enabled=1 in
    # its buffers; tracing the export with live observers would silently move
    # the ranges on the example input. Frozen, always, before export.
    set_observers(model, observe=False, fake_quant=True)

    candidate_dir = Path(config.output_root) / label
    candidate_dir.mkdir(parents=True, exist_ok=True)
    raw_export = candidate_dir / "model.raw.onnx"
    fakequant_path = candidate_dir / "model.fakequant.onnx"
    model_path = candidate_dir / "model.onnx"

    set_export_mode(model, True)
    try:
        export_onnx(
            model, raw_export,
            example_input((1, 3, config.height, config.width)),
            dynamo=False,
        )
    finally:
        set_export_mode(model, False)
    scalar_fix = scalarize_per_tensor_qdq(raw_export, fakequant_path)

    # DESIGN §8.2: a real INT8 graph, not a float graph carrying rounded
    # weights. The fake-quant export stores FP32 weights behind Q/DQ (9.1 MB on
    # the first M2 run); the fold turns them into INT8 initializers, proves
    # bitwise equivalence, and is what ships. The fakequant intermediate stays
    # on disk (gitignored) so the difference remains inspectable.
    fold = fold_qdq_weights(
        fakequant_path, model_path, (1, 3, config.height, config.width)
    )

    description = describe(model_path)
    coverage = ort_coverage.analyse(
        model_path, candidate_dir, f"{config.candidate_prefix}_{label}"
    )
    runs.atomic_write_json(candidate_dir / "coverage.json", coverage)

    candidate = {
        "tool": "wildlife_trigger.optimize.qat_train",
        "design": config.candidate_design,
        "candidate_id": f"{config.candidate_prefix}_{label}",
        "model_id": config.candidate_model_id,
        "kind": config.candidate_kind,
        "method": label,
        "lr": lr,
        "source_run_id": config.source_run_id,
        "source_checkpoint": checkpoint_info,
        "qat_run_id": ctx.run_id,
        "best_epoch": best["epoch"],
        # M4 carries the pruned widths so its comparison row and P-gates can see
        # the architecture is c30's; empty for M2.
        "pruned_widths": dict(config.pruned_widths) or None,
        "calibration": {
            "manifest": calibration["manifest"],
            "sha256": calibration["sha256"],
            "images": calibration["images"],
            "order": "observer warm-up, manifest order; observed through epoch 1",
        },
        "input": {"width": config.width, "height": config.height},
        "qat_structure": structure,
        "qdq_scalar_fix": scalar_fix,
        "weight_fold": fold,
        "model": {
            **description,
            "quantization": {
                "format": "QDQ",
                "scheme": "S8S8",
                "per_channel": True,
                "calibration_method": f"qat_{label}",
            },
        },
        "integer_execution": coverage["verdict"]["integer_execution"],
    }
    runs.atomic_write_json(candidate_dir / "candidate.json", candidate)
    print(
        f"{label}: exported best epoch {best['epoch']} -> {model_path} "
        f"({description['size_bytes']:,} B, integer_execution="
        f"{candidate['integer_execution']})"
    )
    return candidate


def reexport_arm(config: QatConfig, run_dir: Path) -> dict:
    """Re-run only the export path over a completed arm's best checkpoint.

    Exists for export-representation fixes (like the weight fold): the
    training is untouched — the history is read back, its recorded best epoch
    is what gets exported, and a run whose history disagrees with its best.pt
    is refused exactly as during training.
    """
    history = json.loads((run_dir / "history.json").read_text())
    if history["config"]["lr"] != history["lr"]:
        raise RuntimeError(f"{run_dir} history is inconsistent about its lr")

    class _Ctx:
        best_checkpoint_path = run_dir / "best.pt"
        run_id = history["run_id"]

    return export_candidate(
        config,
        _Ctx,
        arm_label(history["lr"]),
        history["lr"],
        {"epoch": history["best_epoch"], "score": history["best_selection_score"]},
        history["class_names"],
        history["initialized_from"],
        history["observer_calibration"],
        history["qat_structure"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--lr", type=float)
    parser.add_argument(
        "--export-only",
        type=Path,
        metavar="RUN_DIR",
        help="skip training; re-export the best checkpoint of this completed "
        "arm run through the current export path (weight fold included)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.export_only:
        candidate = reexport_arm(config, args.export_only)
        print(f"re-exported {candidate['candidate_id']} "
              f"(best epoch {candidate['best_epoch']})")
        return 0

    if args.lr is None:
        parser.error("--lr is required unless --export-only is given")
    result = train_arm(config, args.lr)
    print(
        f"arm {arm_label(args.lr)}: best epoch {result['best_epoch']}, "
        f"primary {result['best_selection_score']['primary']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
