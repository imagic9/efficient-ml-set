"""What a training run leaves behind (issue #10, PLAN C2's third bullet).

`runs.py` claims that "every phase from B onward writes through here, so DESIGN §9.2's
requirement ... is satisfied by construction rather than by remembering to do it". That
sentence was false for a month: `train.py` never imported the module, and all three C1a
arms were produced without provenance or hashes while the docstring said otherwise.

A claim of the form "by construction" is exactly the kind that has to be asserted, not
read. So this drives the real `train.run()` end to end over a synthetic six-image corpus
and inspects the directory it produced. It is slow by this suite's standards (seconds,
not milliseconds), and it is the only test here that would have caught the gap.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wildlife_trigger import metrics as M
from wildlife_trigger import runs

CLASSES = ["bobcat", "coyote", "empty"]


def write_image(directory: Path, name: str, value: int) -> None:
    import cv2

    directory.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(directory / name), np.full((40, 60, 3), value, dtype=np.uint8))


def write_split(
    manifests: Path, images: Path, split: str, labels: list[str]
) -> None:
    records = []
    for index, label in enumerate(labels):
        image_id = f"{split}_{index}"
        write_image(images, f"{image_id}.jpg", 40 * index % 255)
        records.append(
            {
                "image_id": image_id,
                "file_name": f"{image_id}.jpg",
                "labels": [label],
                "primary_label": label,
                "multi_class": False,
                "location": 1,
                # One sequence per two frames: sequence-balanced recall needs sequences.
                "seq_id": f"{split}_seq{index // 2}",
            }
        )
    (manifests / f"{split}.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
    )


@pytest.fixture
def corpus(tmp_path: Path) -> dict:
    manifests = tmp_path / "manifests"
    images = tmp_path / "images"
    manifests.mkdir()

    labels = ["bobcat", "coyote", "empty", "bobcat", "empty", "coyote", "bobcat", "empty"]
    write_split(manifests, images, "train", labels)
    write_split(manifests, images, "cis_val_clean", ["bobcat", "empty", "coyote", "empty"])
    write_split(manifests, images, "trans_val", ["bobcat", "bobcat", "empty", "coyote"])

    classes = tmp_path / "classes.yaml"
    classes.write_text(
        "classes:\n"
        + "".join(f"  - {{name: {n}, index: {i}}}\n" for i, n in enumerate(CLASSES))
    )
    return {"manifests": manifests, "images": images, "classes": classes, "root": tmp_path}


@pytest.mark.slow
def test_a_run_records_everything_c2_requires(corpus: dict) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("cv2")

    from wildlife_trigger.train import TrainConfig, run

    config = TrainConfig(
        run_name="selftest",
        phase="C2",
        manifests_dir=str(corpus["manifests"]),
        images_dir=str(corpus["images"]),
        supplement_manifest=None,
        classes_config=str(corpus["classes"]),
        cache_dir=str(corpus["root"] / "cache"),  # absent: the run decodes its own JPEGs
        output_dir=str(corpus["root"] / "results"),
        width=64,
        height=48,
        batch_size=4,
        workers=0,
        amp=False,
        pretrained=False,  # the engine under test is the recording, not ImageNet
        max_steps=4,
        head_steps=2,
        early_stopping_patience=6,
    )
    summary = run(config)

    run_dir = corpus["root"] / "results" / "c2" / summary["run_id"]
    assert run_dir.is_dir(), "the run id must name the directory the run wrote"

    # -- PLAN C2, bullet 3: "full history, resolved config, environment, dataset/model
    #    hashes, and validation logits/predictions" (the last via dump_predictions).
    resolved = json.loads((run_dir / "resolved_config.json").read_text())
    assert resolved["config"]["seed"] == 42
    assert resolved["config"]["width"] == 64
    assert resolved["command_line"], "how it was invoked is part of reproducing it"

    provenance = json.loads((run_dir / "provenance.json").read_text())
    for key in ("cpu", "gpu", "python", "git", "toolchain"):
        assert key in provenance, f"environment is missing {key}"

    hashes = json.loads((run_dir / "hashes.json").read_text())
    assert hashes["manifest:train"]["sha256"] == runs.sha256_file(
        corpus["manifests"] / "train.jsonl"
    ), "the recorded dataset hash must be the dataset that was read"
    assert hashes["manifest:cis_val_clean"] and hashes["manifest:trans_val"]
    assert hashes["config:classes"], "the class map decides what the head means"
    assert hashes["manifest:empty_supplement"] is None, (
        "a run without the supplement records that, rather than omitting the key"
    )
    assert hashes["caches"]["train"] is None, "no cache: this run decoded its own pixels"

    assert hashes["checkpoint:best"]["sha256"] == runs.sha256_file(run_dir / "best.pt")
    assert hashes["checkpoint:last"]["sha256"] == runs.sha256_file(run_dir / "last.pt")

    assert (run_dir / "history.json").exists()
    assert (run_dir / "run.log").read_text().count("epoch") >= 1, (
        "the log must outlive the ssh session that started the run"
    )

    summary_json = json.loads((run_dir / "run_summary.json").read_text())
    assert summary_json["status"] == "completed"
    assert summary_json["best_selection_score"], "the winning vector belongs in the record"


@pytest.mark.slow
def test_both_checkpoints_carry_optimizer_state(corpus: dict) -> None:
    """DESIGN §7.2: "save last and best checkpoints plus full optimizer/scheduler state".

    `last.pt` used to hold `{"model", "step"}`. A last checkpoint with no optimizer is
    not a resume point, and resuming is the only reason it exists — a run killed at
    epoch 25 of 30 would have restarted from zero while looking like it could not.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("cv2")

    from wildlife_trigger.train import TrainConfig, run

    config = TrainConfig(
        run_name="ckpt",
        phase="C2",
        manifests_dir=str(corpus["manifests"]),
        images_dir=str(corpus["images"]),
        supplement_manifest=None,
        classes_config=str(corpus["classes"]),
        cache_dir=str(corpus["root"] / "cache"),
        output_dir=str(corpus["root"] / "results"),
        width=64,
        height=48,
        batch_size=4,
        workers=0,
        amp=False,
        pretrained=False,
        max_steps=4,
        head_steps=2,
    )
    summary = run(config)
    run_dir = corpus["root"] / "results" / "c2" / summary["run_id"]

    for name in ("best.pt", "last.pt"):
        state = torch.load(run_dir / name, map_location="cpu", weights_only=False)
        assert state["optimiser"]["state"] or state["optimiser"]["param_groups"], (
            f"{name} carries no optimizer state"
        )
        assert state["run_id"] == summary["run_id"], f"{name} must name its own run"
        assert state["class_names"] == CLASSES

    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    assert best["phase"] == "B", "phase A checkpoints are never selected"
    assert set(best["score"]) >= set(M.SELECTION_ORDER), (
        "best.pt must state the whole vector it won on (issue #12)"
    )
    assert best["epoch"] == summary["best_epoch"], (
        "the checkpoint on disk must be the epoch the summary claims"
    )
