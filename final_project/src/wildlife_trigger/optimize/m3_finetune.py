#!/usr/bin/env python3
"""D4: create and fine-tune one M3 pruned candidate (DESIGN §8.3, PLAN D4).

One invocation owns one candidate (`c15`, `c30`, `c45`) end to end:

1. **allocation** — the registered greedy marginal-damage rule
   (`results/optimize/m3_prune/m3_registration.md` §1) over the committed,
   hash-pinned D3 sensitivity report; `c45` is the envelope candidate by
   registration, since the 45% target exceeds the measured cap envelope;
2. **physical pruning** — `prune_expansion` under the full D3 contract (group
   verification, invariant suite, export check), realized widths asserted to
   equal the allocation exactly and every survivor a multiple of 8 *before*
   any fine-tuning (PLAN D4's explicit box);
3. **recovery reference** — the pruned-but-untuned validation primary is
   recorded first; the fine-tune's recovery delta is free evidence here and
   unobtainable later;
4. **fine-tune** — the frozen §7.2 data/loss contract, M0's own full-phase LR
   (3e-4), AMP on exactly as M0's own training, max 15 epochs with
   patience-4 early stopping under the frozen §7.2 selection rule
   (registration §2 — no LR search, one arm per candidate);
5. **export** — best epoch through a fresh `apply_widths` architecture into a
   D1-contract candidate directory (`model.onnx`, `candidate.json`), so
   `evaluate_onnx` → `calibrate_candidate` → `select_m3` → P3/P4 run
   downstream unchanged. Ladder-convention MACs (`macs_of_model`) and the
   tp-counter profile are both recorded, never mixed.

Usage (gx10):
    python -m wildlife_trigger.optimize.m3_finetune \
        --config configs/optimize/m3_prune.yaml --target c15
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
from ..data.dataset import class_weights, load_class_names
from ..metrics import is_better_checkpoint
from ..models.export import describe, export_onnx
from ..models.mobilenet import build_mobilenet_v2, example_input
from ..train import build_datasets, evaluate, score_of
from ..validate.input_cost import macs_of_model
from .prune import (
    allocate_greedy,
    allocation_envelope,
    apply_widths,
    check_invariants,
    evaluate_at_yardstick,
    profile,
    prune_expansion,
)


@dataclass
class M3Config:
    """Duck-types the fields `train.build_datasets` reads, plus the D4 recipe.

    The same YAML file also carries D3's `m0_run`/`ratios`/`round_to` keys —
    one config per optimization family, per DESIGN §9.1 — so those are
    accepted here and simply not used by the fine-tune path.
    """

    # Provenance — all hash-pinned, all refused on mismatch.
    source_run_id: str = ""
    source_checkpoint: str = ""
    source_checkpoint_sha256: str = ""
    sensitivity_report: str = "results/optimize/m3_prune/sensitivity.json"
    sensitivity_report_sha256: str = ""

    # D3's keys (same file, different tool). Unused here.
    m0_run: str = ""
    ratios: list = field(default_factory=list)
    round_to: int = 8

    # Data (the frozen §7.2 contract; same defaults as TrainConfig)
    manifests_dir: str = "data/manifests"
    images_dir: str = "data/raw/extracted/eccv_18_all_images_sm"
    supplement_manifest: str | None = "data/manifests/cct_empty_train_v1.jsonl"
    supplement_dir: str = "data/images/empty_supplement"
    cache_dir: str = "data/cache"
    classes_config: str = "configs/data/classes.yaml"
    width: int = 256
    height: int = 192
    exclude_empty_class: bool = False

    # The registered recipe (m3_registration.md §§1-2)
    targets: dict = field(
        default_factory=lambda: {"c15": 0.15, "c30": 0.30, "c45": 0.45}
    )
    cap: float = 0.5
    quantum: int = 8
    seed: int = 42
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 15
    early_stopping_patience: int = 4
    amp: bool = True
    batch_size: int = 64
    workers: int = 8

    output_root: str = "results/optimize/m3_prune"
    phase: str = "D4"


def load_config(path: Path) -> M3Config:
    raw = yaml.safe_load(path.read_text())
    unknown = [k for k in raw if k not in M3Config.__dataclass_fields__]
    if unknown:
        raise ValueError(f"{path} has unknown keys: {unknown}")
    config = M3Config(**raw)
    for name in (
        "source_run_id",
        "source_checkpoint",
        "source_checkpoint_sha256",
        "sensitivity_report_sha256",
    ):
        if not getattr(config, name):
            raise ValueError(f"{path} must pin {name}")
    return config


def load_sensitivity(config: M3Config) -> dict:
    path = Path(config.sensitivity_report)
    measured = runs.sha256_file(path)
    if measured != config.sensitivity_report_sha256:
        raise RuntimeError(
            f"{path} hashes to {measured[:12]}… but the config pins "
            f"{config.sensitivity_report_sha256[:12]}…; the allocation must read "
            "the committed D3 evidence, not whatever sits at that path today"
        )
    return json.loads(path.read_text())


def load_m0_base(config: M3Config, class_names: list[str]) -> tuple[nn.Module, dict]:
    """The M0 checkpoint as a 16-class network, proven to be the M0 checkpoint."""
    checkpoint_path = Path(config.source_checkpoint)
    measured = runs.sha256_file(checkpoint_path)
    if measured != config.source_checkpoint_sha256:
        raise RuntimeError(
            f"{checkpoint_path} hashes to {measured[:12]}… but the config pins "
            f"{config.source_checkpoint_sha256[:12]}…; M3 prunes M0's selected "
            "checkpoint and nothing else (DESIGN §8.3)"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("class_names") and checkpoint["class_names"] != class_names:
        raise RuntimeError(
            "the checkpoint's class order differs from the frozen classes config"
        )
    base = build_mobilenet_v2(num_classes=len(class_names), pretrained=False)
    base.load_state_dict(checkpoint["model"])
    return base, {
        "path": str(checkpoint_path),
        "sha256": measured,
        "epoch": checkpoint.get("epoch"),
        "run_id": checkpoint.get("run_id"),
    }


def run_candidate(config: M3Config, target_label: str) -> dict:
    if target_label not in config.targets:
        raise ValueError(
            f"{target_label!r} is not a registered target; the registration "
            f"names {sorted(config.targets)} and nothing else"
        )
    target_fraction = float(config.targets[target_label])

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ctx = runs.RunContext.create(
        phase=config.phase,
        name=f"m3_{target_label}",
        config={**asdict(config), "target": target_label},
        results_root=Path(config.output_root) / "runs",
    )

    class_names = load_class_names(Path(config.classes_config))
    report = load_sensitivity(config)

    # 1. The registered allocation, from committed evidence only.
    allocation = allocate_greedy(
        report, target_fraction, cap=config.cap, quantum=config.quantum
    )
    envelope = allocation_envelope(report)
    print(
        f"{target_label}: target {target_fraction:.0%}, predicted "
        f"{allocation['predicted_reduction']:.4f} "
        f"(envelope {envelope:.4f}, exhausted={allocation['envelope_exhausted']})"
    )

    # 2. Physical pruning under the full D3 contract.
    model, checkpoint_info = load_m0_base(config, class_names)
    block_ratios = {int(k.split(".")[1]): v for k, v in allocation["ratios"].items()}
    prune_report = prune_expansion(
        model,
        block_ratios,
        round_to=config.quantum,
        num_classes=len(class_names),
        verify_groups=True,
        export_check=True,
    )
    for name, width in allocation["widths"].items():
        realized = prune_report["invariants"]["expansion_widths"][name]
        if realized != width:
            raise RuntimeError(
                f"{name}: solver realized {realized}, allocation wanted {width}; "
                "requested-vs-realized may differ from the *target*, never from "
                "the allocation"
            )
        if width < 8 or width % 8:
            raise RuntimeError(f"{name}: width {width} violates the %8 contract")

    manifests = Path(config.manifests_dir)
    ctx.record_hashes(
        {
            "manifest:train": manifests / "train.jsonl",
            "manifest:cis_val_clean": manifests / "cis_val_clean.jsonl",
            "manifest:trans_val": manifests / "trans_val.jsonl",
            "manifest:empty_supplement": Path(config.supplement_manifest),
            "config:classes": Path(config.classes_config),
            "checkpoint:m0_source": Path(config.source_checkpoint),
            "evidence:sensitivity": Path(config.sensitivity_report),
        },
        class_names=class_names,
    )

    model.to(device)

    # 3. The recovery reference: pruned, untuned, deployment-regime torch eval.
    # TF32 is disabled only around the scoring — the §6.3 amendment governs
    # scoring; the fine-tune itself runs under torch's defaults, exactly as
    # M0's own training did.
    from ..validate.dump_predictions import build_validation_loaders

    yardstick_loaders = build_validation_loaders(
        {**asdict(config), "batch_size": config.batch_size, "workers": config.workers},
        class_names,
    )
    tf32_state = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        pre_ft = evaluate_at_yardstick(model, yardstick_loaders, device)
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = tf32_state
    print(f"{target_label}: pruned-untuned primary {pre_ft['primary']:.4f}")

    # 4. Fine-tune under the frozen contract.
    data = build_datasets(config, class_names)
    train_records = [r for part in data["train_parts"] for r in part.records]
    weights = class_weights(train_records, class_names).to(device)

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
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=len(loaders["train"]) * config.max_epochs
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=config.amp and device.type == "cuda"
    )

    history: list[dict] = []
    best: dict = {"score": None, "epoch": -1}
    started = time.time()

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        epoch_loss, batches = 0.0, 0
        for batch in loaders["train"]:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                loss = criterion(model(images), targets)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
            epoch_loss += float(loss.detach())
            batches += 1

        results = evaluate(model, validation_loaders, class_names, device)
        score = score_of(results)
        entry = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(batches, 1),
            "selection_score": score,
            "cis_val_clean": dict(results["cis_val_clean"]["target"]),
            "trans_val": dict(results["trans_val"]["target"]),
            "macro_f1": {
                d: results[d]["classes"]["support_aware_macro_f1"] for d in results
            },
            "elapsed_s": round(time.time() - started, 1),
        }
        history.append(entry)
        print(
            f"epoch {epoch}/{config.max_epochs}  loss {entry['train_loss']:.4f}  "
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
            "widths": allocation["widths"],
            "config": {**asdict(config), "target": target_label},
        }
        runs.save_checkpoint(ctx.checkpoint_path, state)
        if is_better_checkpoint(score, best["score"]):
            best = {"score": score, "epoch": epoch}
            runs.save_checkpoint(ctx.best_checkpoint_path, state)
        elif epoch - best["epoch"] >= config.early_stopping_patience:
            print(
                f"early stop: no §7.2 improvement since epoch {best['epoch']} "
                f"({config.early_stopping_patience} epochs)"
            )
            break

    history_record = {
        "run_name": f"m3_{target_label}",
        "run_id": ctx.run_id,
        "target": target_label,
        "target_fraction": target_fraction,
        "best_epoch": best["epoch"],
        "best_selection_score": best["score"],
        "class_names": class_names,
        "config": {**asdict(config), "target": target_label},
        "initialized_from": checkpoint_info,
        "allocation": allocation,
        "envelope": envelope,
        "prune_report": {
            key: prune_report[key]
            for key in (
                "requested", "realized", "round_to", "profile_before",
                "profile_after", "mac_reduction", "param_reduction",
            )
        },
        "pre_finetune_yardstick": pre_ft,
        "history": history,
    }
    runs.atomic_write_json(ctx.run_dir / "history.json", history_record)

    export_record = export_candidate(
        config, ctx, target_label, best, class_names, checkpoint_info,
        allocation, prune_report, pre_ft,
    )
    ctx.finish(status="completed", best_epoch=best["epoch"],
               best_selection_score=best["score"])
    return {**history_record, "export": export_record}


def export_candidate(
    config: M3Config,
    ctx,
    target_label: str,
    best: dict,
    class_names: list[str],
    checkpoint_info: dict,
    allocation: dict,
    prune_report: dict,
    pre_ft: dict,
) -> dict:
    """Best epoch -> deployable pruned FP32 ONNX in a D1-contract candidate dir."""
    checkpoint = runs.load_checkpoint(ctx.best_checkpoint_path)
    if checkpoint["epoch"] != best["epoch"]:
        raise RuntimeError(
            f"best.pt holds epoch {checkpoint['epoch']} but the loop selected "
            f"{best['epoch']}; refusing to export a different model than selected"
        )
    if checkpoint["widths"] != allocation["widths"]:
        raise RuntimeError(
            "best.pt's recorded widths differ from this run's allocation; the "
            "checkpoint is not this candidate's architecture"
        )

    model = build_mobilenet_v2(num_classes=len(class_names), pretrained=False)
    apply_widths(model, checkpoint["widths"])
    model.load_state_dict(checkpoint["model"])
    model.eval().cpu()
    invariants = check_invariants(model, num_classes=len(class_names))

    candidate_dir = Path(config.output_root) / target_label
    candidate_dir.mkdir(parents=True, exist_ok=True)
    model_path = candidate_dir / "model.onnx"

    export_onnx(
        model, model_path,
        example_input((1, 3, config.height, config.width)),
        dynamo=False,
    )
    description = describe(model_path)

    # The physical evidence P3's FP32 check 1 verifies against the artifact:
    # the exported conv-shape multiset, recorded from the verified module.
    conv_shapes = sorted(
        list(m.weight.shape)
        for m in model.modules()
        if isinstance(m, nn.Conv2d)
    )

    ladder_macs = macs_of_model(model, config.width, config.height)
    tp_profile = profile(model)
    params = sum(p.numel() for p in model.parameters())

    candidate = {
        "tool": "wildlife_trigger.optimize.m3_finetune",
        "design": "8.3",
        "candidate_id": f"d4_m3_{target_label}",
        "model_id": "M3-candidate",
        "kind": "pruned_fp32",
        "method": target_label,
        "seed": config.seed,
        "source_run_id": config.source_run_id,
        "source_checkpoint": checkpoint_info,
        "finetune_run_id": ctx.run_id,
        "finetune_run_dir": str(ctx.run_dir),
        "best_epoch": best["epoch"],
        "best_checkpoint_sha256": runs.sha256_file(ctx.best_checkpoint_path),
        "input": {"width": config.width, "height": config.height},
        "pruning": {
            "registration": "results/optimize/m3_prune/m3_registration.md",
            "target_fraction": allocation["target_fraction"],
            "predicted_reduction": allocation["predicted_reduction"],
            "envelope_exhausted": allocation["envelope_exhausted"],
            "widths": allocation["widths"],
            "removals": allocation["removals"],
            "realized_mac_reduction_tp": prune_report["mac_reduction"],
            "param_reduction": prune_report["param_reduction"],
            "profile_tp": tp_profile,
            "params": params,
            "macs_ladder_convention": ladder_macs,
            "exported_conv_shapes": conv_shapes,
            "pre_finetune_primary": pre_ft["primary"],
            "invariants": invariants,
        },
        "model": description,
    }
    runs.atomic_write_json(candidate_dir / "candidate.json", candidate)
    print(
        f"{target_label}: exported best epoch {best['epoch']} -> {model_path} "
        f"({description['size_bytes']:,} B, {ladder_macs:,} ladder MACs, "
        f"{params:,} params)"
    )
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--target", required=True, help="c15 | c30 | c45")
    args = parser.parse_args()

    config = load_config(args.config)
    result = run_candidate(config, args.target)
    print(
        f"done: {result['run_id']} best epoch {result['best_epoch']} "
        f"primary {result['best_selection_score']['primary']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
