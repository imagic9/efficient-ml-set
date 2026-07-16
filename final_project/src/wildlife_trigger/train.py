#!/usr/bin/env python3
"""C1 — the training engine (DESIGN §7.2).

Recipe, all of it from DESIGN §7.2 and deviations recorded in the run config:
phase A trains the head for 5 epochs with the backbone frozen, phase B fine-tunes
everything for at most 30; AdamW, lr 1e-3 head / 3e-4 full, weight decay 1e-4, cosine
decay, early stopping patience 6.

Two things this engine refuses to do:

**It does not select on accuracy.** `empty` dominates the corpus, so the most accurate
model is the one best at predicting that nothing happened. The checkpoint score is mean
bobcat F2 across cis-val-clean and trans-val at the fixed 0.5 yardstick, with
sequence-balanced recall and support-aware macro F1 as tie-breaks (DESIGN §7.2 — which
also records the AP amendment that challenged this rule and was reverted by its own
pre-registered test, issue #19). Bobcat AP stays recorded per epoch beside F2: it does
not select, but its per-checkpoint bootstrap is ~4.5x tighter and the report will want
it.

**It does not touch test.** Only train, cis-val-clean and trans-val are reachable from
here (DESIGN §5.4). cis-test and trans-test are sealed until every model, threshold and
runtime decision is frozen.

The step budget is fixed rather than the epoch count, because the empty-supplement
ablation changes the training set from 13,546 to 18,546 images. Matching epochs would
give the supplement arm 36.9% more optimizer steps and confound "empty data helps" with
"this arm trained longer" (DESIGN §5.2).

**Every run writes through `runs.RunContext`** (issue #10), so DESIGN §9.2's provenance
and PLAN C2's third bullet are satisfied by construction: an immutable run id, the
resolved config, the environment, the git state, a log that outlives the ssh session,
and the hashes of every manifest, class map, cache and checkpoint the run touched. This
is not paperwork. M0 is the baseline every optimized candidate is measured against, and
a number whose inputs cannot be named afterwards cannot be defended.

Usage:
    python -m wildlife_trigger.train --config configs/train/m0_fp32.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from wildlife_trigger import metrics as M
from wildlife_trigger import runs
from wildlife_trigger.data.dataset import (
    ConcatManifestDataset,
    WildlifeDataset,
    class_weights,
    load_class_names,
)
from wildlife_trigger.data.preprocess import PreprocessConfig
from wildlife_trigger.models.mobilenet import build_mobilenet_v2

TARGET_CLASS = "bobcat"

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    run_name: str
    seed: int = 42

    # The phase this run belongs to. It names the run id and the directory under
    # `output_dir`, so a result can be traced to the PLAN task that asked for it.
    phase: str = "C2"

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

    # DESIGN §7.1's ImageNet initialisation. False exists so the engine can be exercised
    # without the torchvision download, and because §7.2 requires a deviation this large
    # to be *recorded in the run config* rather than implied by a code path.
    pretrained: bool = True

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

    # Root for run directories, not the run directory itself: `RunContext` owns the
    # layout below it (`<output_dir>/<phase>/<run_id>/`, DESIGN §14's results/training).
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

        target = M.target_presence_metrics(
            probabilities[:, target_index], present[:, target_index], seq_ids, threshold
        )
        # Threshold-free, next to the thresholded metrics on purpose: the selection rule
        # reads this (DESIGN §7.2 as amended, issue #19), and recording both per epoch is
        # what lets the amendment's stability test be run on real trajectories instead of
        # asserted.
        target["average_precision"] = M.average_precision(
            probabilities[:, target_index], present[:, target_index]
        )

        results[domain] = {
            "target": target,
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


def cache_fingerprints(datasets: list[WildlifeDataset]) -> dict:
    """Which pixels each split actually read.

    `null` means the split decoded its own JPEGs because no cache was there. That is a
    real difference between two runs of the same config and it belongs in the record:
    the cache is a derived artifact, and its fingerprint is what ties this run's pixels
    to a manifest and a preprocessing config (DESIGN §5.5).
    """
    return {
        dataset.manifest.stem: (
            None
            if dataset.cache_meta is None
            else {
                "fingerprint": dataset.cache_meta["fingerprint"],
                "image_id_order_sha256": dataset.cache_meta["image_id_order_sha256"],
                "images": dataset.cache_meta["images"],
            }
        )
        for dataset in datasets
    }


def run(config: TrainConfig) -> dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = resolve_device()

    # First, before the data: a run that dies building datasets still has to be able to
    # say which code and which machine it died on.
    ctx = runs.RunContext.create(
        phase=config.phase,
        name=config.run_name,
        config=asdict(config),
        results_root=Path(config.output_dir),
    )

    class_names = load_class_names(Path(config.classes_config))
    if config.exclude_empty_class:
        # The 15-output arm: DESIGN §5.2's control. `empty` is removed from the head
        # entirely rather than left as a class that never sees a positive.
        class_names = [n for n in class_names if n != "empty"]

    data = build_datasets(config, class_names)
    train_records = [r for part in data["train_parts"] for r in part.records]
    weights = class_weights(train_records, class_names).to(device)

    # DESIGN §9.2's dataset hashes, recorded before the first optimizer step rather than
    # after the last: this is what makes "which data was this trained on" answerable for
    # a run that never finishes.
    manifests = Path(config.manifests_dir)
    ctx.record_hashes(
        {
            "manifest:train": manifests / "train.jsonl",
            "manifest:cis_val_clean": manifests / "cis_val_clean.jsonl",
            "manifest:trans_val": manifests / "trans_val.jsonl",
            # Null for the no-empty arm, which is the point: DESIGN §5.2's control is
            # visible in the hashes, not only in the config that requested it.
            "manifest:empty_supplement": (
                Path(config.supplement_manifest)
                if config.supplement_manifest and not config.exclude_empty_class
                else None
            ),
            "config:classes": Path(config.classes_config),
        },
        caches=cache_fingerprints(
            [*data["train_parts"], *data["validation"].values()]
        ),
        class_names=class_names,
    )

    # Surfaced, not buried: the no-empty arm legitimately sees `empty` in validation and
    # models it as "no animal present". Anything else appearing here is a bug.
    unmodelled = {
        name: dataset.unmodelled_labels
        for name, dataset in data["validation"].items()
        if dataset.unmodelled_labels
    }
    if unmodelled:
        LOGGER.info("labels present in data but not modelled by this head: %s", unmodelled)
        unexpected = {
            name: [l for l in labels if l != "empty"] for name, labels in unmodelled.items()
        }
        unexpected = {k: v for k, v in unexpected.items() if v}
        if unexpected:
            raise RuntimeError(
                f"unmodelled labels other than 'empty': {unexpected}. Only DESIGN "
                "§5.2's 15-output no-empty arm may drop a class, and only `empty`."
            )

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

    model = build_mobilenet_v2(
        num_classes=len(class_names), pretrained=config.pretrained
    ).to(device)

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

    output = ctx.run_dir

    history = []
    # The whole score vector, not its first component: `is_better_checkpoint` needs the
    # tie-breaks to be able to break a tie. None until phase B offers the first one.
    best: dict = {"score": None, "epoch": -1}
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

    def checkpoint_state() -> dict:
        """What DESIGN §7.2 means by "last and best plus full optimizer/scheduler state".

        `last.pt` carries the same state as `best.pt` rather than a bare `state_dict`:
        without the optimizer and scheduler it is not a resume point, and a resume point
        is the only reason a last checkpoint exists at all.

        The score here is the full DESIGN §7.2 vector, so `best.pt` can state what it
        won on without being read against a history file.
        """
        return {
            "run_id": ctx.run_id,
            "model": model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "step": step,
            "phase": phase,
            "score": score,
            "class_names": class_names,
            "config": asdict(config),
        }

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
                LOGGER.info("phase A -> B at step %d", step)

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
        LOGGER.info(
            "epoch %2d [%s] step %5d/%d  loss %.4f  bobcatAP cis %.4f trans %.4f  "
            "F2@0.5 cis %.4f trans %.4f  score %.4f",
            epoch,
            phase,
            step,
            max_steps,
            entry["train_loss"],
            entry["cis_val_clean"]["average_precision"],
            entry["trans_val"]["average_precision"],
            entry["cis_val_clean"]["frame_f2"],
            entry["trans_val"]["frame_f2"],
            score["primary"],
        )

        if phase == "B" and M.is_better_checkpoint(score, best["score"]):
            # Phase A checkpoints are never selected: the backbone has not moved, so a
            # head-only model that happens to score well early would be chosen over a
            # properly fine-tuned one and the run's whole phase B would be discarded.
            best = {"score": score, "epoch": epoch}
            runs.save_checkpoint(ctx.best_checkpoint_path, checkpoint_state())

        # Patience runs from the last epoch that won under the full rule, so an epoch
        # that ties on F2 and improves a tie-break counts as progress here exactly as it
        # does above. One comparator, one definition of "better".
        if epoch - best["epoch"] >= config.early_stopping_patience and phase == "B":
            LOGGER.info(
                "early stopping: no improvement for %d epochs",
                config.early_stopping_patience,
            )
            break
        if step >= max_steps:
            LOGGER.info("step budget reached: %d/%d", step, max_steps)
            break

    runs.save_checkpoint(ctx.checkpoint_path, checkpoint_state())

    summary = {
        "run_id": ctx.run_id,
        "run_name": config.run_name,
        "config": asdict(config),
        "class_names": class_names,
        "class_weights": weights.cpu().tolist(),
        "best_epoch": best["epoch"],
        "best_score": best["score"]["primary"] if best["score"] else None,
        # The vector the checkpoint actually won on, so the selection can be re-checked
        # against DESIGN §7.2 without re-deriving it from the history.
        "best_selection_score": best["score"],
        "selection_rule": {
            "order": list(M.SELECTION_ORDER),
            "primary_metric": M.PRIMARY_METRIC,
            "final_tiebreak": "earliest epoch",
            "phase_b_only": True,
        },
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

    # After the checkpoints exist, so the hashes are of the files that were actually
    # written. DESIGN §9.2's model hashes, and what C3's policy JSON binds itself to.
    ctx.record_hashes(
        {
            "checkpoint:best": (
                ctx.best_checkpoint_path if ctx.best_checkpoint_path.exists() else None
            ),
            "checkpoint:last": ctx.checkpoint_path,
        }
    )
    ctx.finish(
        status="completed",
        best_epoch=best["epoch"],
        best_score=summary["best_score"],
        best_selection_score=best["score"],
        selection_rule=summary["selection_rule"],
        budget=summary["budget"],
        device=str(device),
    )
    LOGGER.info(
        "best epoch %s score %s -> %s", best["epoch"], summary["best_score"], output
    )
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
