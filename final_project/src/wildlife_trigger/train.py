#!/usr/bin/env python3
"""C1 — the training engine (DESIGN §7.2).

Recipe, all of it from DESIGN §7.2 and deviations recorded in the run config:
phase A trains the head for 5 epochs with the backbone frozen, phase B fine-tunes
everything for at most 30; AdamW, lr 1e-3 head / 3e-4 full, weight decay 1e-4, cosine
decay, early stopping patience 6.

Two things this engine refuses to do:

**It does not select on accuracy.** `empty` dominates the corpus, so the most accurate
model is the one best at predicting that nothing happened. The checkpoint score is mean
bobcat F2 across cis-val-clean and trans-val, with sequence-balanced recall and
support-aware macro F1 as tie-breaks (DESIGN §7.2).

**It does not touch test.** Only train, cis-val-clean and trans-val are reachable from
here (DESIGN §5.4). cis-test and trans-test are sealed until every model, threshold and
runtime decision is frozen.

The step budget is fixed rather than the epoch count, because the empty-supplement
ablation changes the training set from 13,546 to 18,546 images. Matching epochs would
give the supplement arm 36.9% more optimizer steps and confound "empty data helps" with
"this arm trained longer" (DESIGN §5.2).

Usage:
    python -m wildlife_trigger.train --config configs/train/m0_fp32.yaml
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

from wildlife_trigger import metrics as M
from wildlife_trigger.data.dataset import (
    ConcatManifestDataset,
    WildlifeDataset,
    class_weights,
    load_class_names,
)
from wildlife_trigger.data.preprocess import PreprocessConfig
from wildlife_trigger.models.mobilenet import build_mobilenet_v2

TARGET_CLASS = "bobcat"


@dataclass
class TrainConfig:
    run_name: str
    seed: int = 42

    # Data
    manifests_dir: str = "data/manifests"
    images_dir: str = "data/raw/extracted/eccv_18_all_images_sm"
    supplement_manifest: str | None = "data/manifests/cct_empty_train_v1.jsonl"
    supplement_dir: str = "data/images/empty_supplement"
    cache_dir: str = "data/cache"
    classes_config: str = "configs/data/classes.yaml"

    # Input geometry (DESIGN §5.5; C1a resolves 256x192 vs 224x224)
    width: int = 256
    height: int = 192

    # DESIGN §7.2
    head_epochs: int = 5
    max_epochs: int = 30
    batch_size: int = 64
    head_lr: float = 1e-3
    full_lr: float = 3e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 6
    amp: bool = True
    workers: int = 8

    # DESIGN §5.2: the ablation matches optimizer steps, not epochs. None = derive from
    # this run's own dataset size.
    #
    # `head_steps` matters as much as `max_steps` and is easy to forget. The supplement
    # arm has 289 steps/epoch against the no-empty arm's 211, so "phase A = 5 epochs"
    # silently gives it 1,445 head steps versus 1,055 — a 37% larger head budget inside
    # a comparison that is supposed to be step-matched. Matching the total while
    # mismatching the phases is not matching.
    max_steps: int | None = None
    head_steps: int | None = None

    # The no-empty arm trains a 15-output head on a training set with no `empty` frames.
    exclude_empty_class: bool = False

    output_dir: str = "results/training"


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_datasets(config: TrainConfig, class_names: list[str]) -> dict:
    root = Path(".")
    preprocess = PreprocessConfig(width=config.width, height=config.height)
    manifests = root / config.manifests_dir

    train_parts = [
        WildlifeDataset(
            manifests / "train.jsonl",
            class_names,
            preprocess,
            root / config.images_dir,
            cache_root=root / config.cache_dir,
            train=True,
            seed=config.seed,
        )
    ]
    if config.supplement_manifest and not config.exclude_empty_class:
        train_parts.append(
            WildlifeDataset(
                Path(config.supplement_manifest),
                class_names,
                preprocess,
                root / config.supplement_dir,
                cache_root=root / config.cache_dir,
                train=True,
                seed=config.seed + 1,
            )
        )

    validation = {
        name: WildlifeDataset(
            manifests / f"{name}.jsonl",
            class_names,
            preprocess,
            root / config.images_dir,
            cache_root=root / config.cache_dir,
            train=False,  # deterministic: no augmentation, ever
        )
        for name in ("cis_val_clean", "trans_val")
    }

    return {
        "train": ConcatManifestDataset(train_parts) if len(train_parts) > 1 else train_parts[0],
        "train_parts": train_parts,
        "validation": validation,
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    class_names: list[str],
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """Score both validation domains. inference_mode, never just no_grad."""
    model.eval()
    target_index = class_names.index(TARGET_CLASS)
    results = {}

    for domain, loader in loaders.items():
        probabilities, present, seq_ids = [], [], []
        for batch in loader:
            logits = model(batch["image"].to(device, non_blocking=True))
            probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            present.append(batch["present"].numpy())
            seq_ids.extend(loader.dataset.records[i]["seq_id"] for i in batch["index"].tolist())

        probabilities = np.concatenate(probabilities)
        present = np.concatenate(present)

        results[domain] = {
            "target": M.target_presence_metrics(
                probabilities[:, target_index], present[:, target_index], seq_ids, threshold
            ),
            "classes": M.per_class_metrics(probabilities, present, class_names, seq_ids),
            "_probabilities": probabilities,
            "_present": present,
            "_seq_ids": seq_ids,
        }
    return results


def score_of(results: dict) -> dict:
    return M.selection_score(
        results["cis_val_clean"]["target"],
        results["trans_val"]["target"],
        (
            results["cis_val_clean"]["classes"]["support_aware_macro_f1"]
            + results["trans_val"]["classes"]["support_aware_macro_f1"]
        )
        / 2,
    )


def run(config: TrainConfig) -> dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = resolve_device()

    class_names = load_class_names(Path(config.classes_config))
    if config.exclude_empty_class:
        # The 15-output arm: DESIGN §5.2's control. `empty` is removed from the head
        # entirely rather than left as a class that never sees a positive.
        class_names = [n for n in class_names if n != "empty"]

    data = build_datasets(config, class_names)
    train_records = [r for part in data["train_parts"] for r in part.records]
    weights = class_weights(train_records, class_names).to(device)

    loaders = {
        "train": DataLoader(
            data["train"],
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=config.workers > 0,
        ),
        **{
            name: DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.workers,
                pin_memory=True,
                persistent_workers=config.workers > 0,
            )
            for name, dataset in data["validation"].items()
        },
    }
    validation_loaders = {k: v for k, v in loaders.items() if k != "train"}

    model = build_mobilenet_v2(num_classes=len(class_names), pretrained=True).to(device)

    # ignore_index=-1: the multi-class frames carry target -1 and are skipped by CE
    # while remaining in the dataset for target-presence evaluation (DESIGN B3).
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-1)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")

    steps_per_epoch = len(loaders["train"])
    max_steps = config.max_steps or steps_per_epoch * config.max_epochs
    head_steps = config.head_steps or steps_per_epoch * config.head_epochs
    if head_steps >= max_steps:
        raise ValueError(
            f"head_steps ({head_steps}) >= max_steps ({max_steps}): phase B would "
            "never run and the backbone would never be fine-tuned."
        )

    output = Path(config.output_dir) / config.run_name
    output.mkdir(parents=True, exist_ok=True)

    history = []
    best = {"score": -1.0, "epoch": -1}
    step = 0
    images_seen = 0
    non_empty_seen = 0
    empty_index = class_names.index("empty") if "empty" in class_names else None
    started = time.time()

    def set_phase(phase: str) -> torch.optim.Optimizer:
        """Phase A freezes the backbone; phase B trains everything (DESIGN §7.2)."""
        frozen = phase == "A"
        for name, parameter in model.named_parameters():
            parameter.requires_grad = (not frozen) or name.startswith("classifier")
        trainable = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            trainable,
            lr=config.head_lr if frozen else config.full_lr,
            weight_decay=config.weight_decay,
        )

    optimiser = set_phase("A")
    scheduler = None
    phase = "A"

    for epoch in range(1000):  # bounded by max_steps and early stopping, not by this
        model.train()
        epoch_loss, batches = 0.0, 0
        for batch in loaders["train"]:
            if step >= max_steps:
                break

            # The phase boundary is a step count, not an epoch count, so two arms with
            # different dataset sizes get identical head and fine-tune budgets.
            if phase == "A" and step >= head_steps:
                phase = "B"
                optimiser = set_phase("B")
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimiser, T_max=max(1, max_steps - step)
                )
                print(f"  phase A -> B at step {step}", flush=True)

            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                loss = criterion(model(images), targets)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            epoch_loss += float(loss.detach())
            batches += 1
            step += 1
            images_seen += len(targets)
            if empty_index is not None:
                non_empty_seen += int((targets != empty_index).sum())
            else:
                non_empty_seen += len(targets)

        results = evaluate(model, validation_loaders, class_names, device)
        score = score_of(results)

        entry = {
            "epoch": epoch,
            "phase": phase,
            "step": step,
            "train_loss": epoch_loss / max(batches, 1),
            "selection_score": score,
            "cis_val_clean": {
                k: v for k, v in results["cis_val_clean"]["target"].items()
            },
            "trans_val": {k: v for k, v in results["trans_val"]["target"].items()},
            "macro_f1": {
                d: results[d]["classes"]["support_aware_macro_f1"] for d in results
            },
            "elapsed_s": round(time.time() - started, 1),
        }
        history.append(entry)
        print(
            f"epoch {epoch:2d} [{phase}] step {step:5d}/{max_steps}  "
            f"loss {entry['train_loss']:.4f}  "
            f"bobcatF2 cis {entry['cis_val_clean']['frame_f2']:.4f} "
            f"trans {entry['trans_val']['frame_f2']:.4f}  "
            f"score {score['primary']:.4f}",
            flush=True,
        )

        if score["primary"] > best["score"] and phase == "B":
            # Phase A checkpoints are never selected: the backbone has not moved, so a
            # head-only model that happens to score well early would be chosen over a
            # properly fine-tuned one and the run's whole phase B would be discarded.
            best = {"score": score["primary"], "epoch": epoch}
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimiser": optimiser.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler else None,
                    "epoch": epoch,
                    "step": step,
                    "score": score,
                    "class_names": class_names,
                    "config": asdict(config),
                },
                output / "best.pt",
            )

        if epoch - best["epoch"] >= config.early_stopping_patience and phase == "B":
            print(f"early stopping: no improvement for {config.early_stopping_patience} epochs")
            break
        if step >= max_steps:
            print(f"step budget reached: {step}/{max_steps}")
            break

    torch.save({"model": model.state_dict(), "step": step}, output / "last.pt")

    summary = {
        "run_name": config.run_name,
        "config": asdict(config),
        "class_names": class_names,
        "class_weights": weights.cpu().tolist(),
        "best_epoch": best["epoch"],
        "best_score": best["score"],
        # DESIGN §5.2 requires all four for the ablation to be interpretable.
        "budget": {
            "steps": step,
            "max_steps": max_steps,
            "head_steps": head_steps,
            "steps_per_epoch": steps_per_epoch,
            "effective_epochs": round(step / steps_per_epoch, 2),
            "images_seen": images_seen,
            # DESIGN §5.2: this is the value that makes the compute-matched supplement
            # arm's LOWER animal exposure explicit, rather than letting "empty data
            # helps" quietly mean "this arm saw 37% fewer animals".
            "non_empty_images_seen": non_empty_seen,
            "train_images": len(data["train"]),
        },
        "history": history,
        "elapsed_s": round(time.time() - started, 1),
        "device": str(device),
    }
    (output / "history.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nbest epoch {best['epoch']} score {best['score']:.4f} -> {output}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    # action="extend", not the default "store": with plain nargs="*" a second --override
    # silently REPLACES the first rather than adding to it, so half an experiment's
    # settings vanish without a word. That is not hypothetical — it launched a C1a arm
    # with the wrong step budget and the wrong dataset.
    parser.add_argument(
        "--override", nargs="*", action="extend", default=[], help="key=value"
    )
    args = parser.parse_args()

    raw = yaml.safe_load(args.config.read_text())
    for override in args.override:
        key, _, value = override.partition("=")
        if key not in raw and key not in TrainConfig.__dataclass_fields__:
            # A typo'd key would otherwise be accepted and ignored, and the run would
            # quietly use the config's default.
            parser.error(f"unknown config key: {key!r}")
        current = raw.get(key)
        if isinstance(current, bool) or value in ("true", "false"):
            raw[key] = value == "true"
        elif isinstance(current, int) or (value.isdigit() and not isinstance(current, str)):
            raw[key] = int(value)
        elif isinstance(current, float):
            raw[key] = float(value)
        else:
            raw[key] = value

    run(TrainConfig(**raw))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
