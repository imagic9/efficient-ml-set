"""Tests for run directories, provenance, logging and checkpoint/resume (A2).

These assert the properties later phases actually depend on, not the shape of the
API: that a run records enough to be explained afterwards, that a checkpoint
survives a kill mid-write, and that a resumable run can be found again.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from wildlife_trigger import runs


@pytest.fixture(autouse=True)
def _isolate_logging():
    """RunContext attaches handlers to the root logger; do not leak them."""
    root = logging.getLogger()
    before = list(root.handlers)
    yield
    for handler in list(root.handlers):
        if handler not in before:
            handler.close()
            root.removeHandler(handler)


def test_run_id_is_sortable_and_safe() -> None:
    import datetime as dt

    when = dt.datetime(2026, 7, 15, 18, 30, 0, tzinfo=dt.timezone.utc)
    run_id = runs.make_run_id("C2", "m0/fp32 seed=42", when)

    assert run_id == "c2_m0_fp32_seed_42_20260715T183000Z"
    assert "/" not in run_id and " " not in run_id, "must be a safe directory name"

    earlier = runs.make_run_id("C2", "x", when - dt.timedelta(hours=1))
    assert earlier < run_id, "run ids must sort chronologically"


def test_create_writes_config_and_provenance(tmp_path: Path) -> None:
    ctx = runs.RunContext.create(
        phase="A2",
        name="selftest",
        config={"lr": 3e-4, "seed": 42},
        results_root=tmp_path,
    )

    resolved = json.loads((ctx.run_dir / "resolved_config.json").read_text())
    assert resolved["config"] == {"lr": 3e-4, "seed": 42}
    assert resolved["run_id"] == ctx.run_id
    assert resolved["command_line"], "must record how it was invoked"

    prov = json.loads((ctx.run_dir / "provenance.json").read_text())
    # DESIGN §9.2's list, spot-checked on the fields that make a result explainable.
    for key in ("cpu", "gpu", "python", "git", "toolchain", "other_workloads"):
        assert key in prov, f"provenance is missing {key}"

    assert ctx.log_path.exists(), "logging must persist beyond the terminal"


def test_provenance_never_captures_secrets(tmp_path: Path) -> None:
    """The safelist is the whole defence; a wholesale os.environ dump would leak."""
    import os

    os.environ["WILDLIFE_FAKE_TOKEN"] = "ghp_not_a_real_secret_but_shaped_like_one"
    try:
        ctx = runs.RunContext.create(
            phase="A2", name="secret", config={}, results_root=tmp_path
        )
        text = (ctx.run_dir / "provenance.json").read_text()
        assert "ghp_not_a_real_secret" not in text
        assert "WILDLIFE_FAKE_TOKEN" not in text
    finally:
        del os.environ["WILDLIFE_FAKE_TOKEN"]


def test_finish_records_status_and_elapsed(tmp_path: Path) -> None:
    ctx = runs.RunContext.create(
        phase="A2", name="finish", config={}, results_root=tmp_path
    )
    ctx.finish(status="completed", note="all good")

    summary = json.loads((ctx.run_dir / "run_summary.json").read_text())
    assert summary["status"] == "completed"
    assert summary["note"] == "all good"
    assert summary["elapsed_seconds"] >= 0


def test_atomic_write_leaves_no_partial_file(tmp_path: Path) -> None:
    target = tmp_path / "metrics.json"
    runs.atomic_write_json(target, {"recall": 0.91})

    assert json.loads(target.read_text()) == {"recall": 0.91}
    assert not list(tmp_path.glob("*.tmp")), "temp file must not survive"


def test_checkpoint_roundtrip_and_atomicity(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    path = tmp_path / "ckpt.pt"
    state = {"epoch": 7, "model": {"w": torch.zeros(3)}, "seed": 42}
    runs.save_checkpoint(path, state)

    assert not list(tmp_path.glob("*.tmp")), "no partial file after a clean save"

    loaded = runs.load_checkpoint(path)
    assert loaded is not None
    assert loaded["epoch"] == 7
    assert loaded["seed"] == 42

    assert runs.load_checkpoint(tmp_path / "absent.pt") is None, (
        "a missing checkpoint means 'start fresh', not an error"
    )


def test_find_resumable_picks_the_newest_with_a_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("torch")

    phase = tmp_path / "c2"
    # Older run that did checkpoint; newer run that died before its first save.
    older = phase / "c2_m0_20260101T000000Z"
    newer = phase / "c2_m0_20260202T000000Z"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    runs.save_checkpoint(older / "checkpoint_last.pt", {"epoch": 1})

    found = runs.find_resumable(tmp_path, "C2", "m0")
    assert found == older, "a run without a checkpoint is not resumable"

    runs.save_checkpoint(newer / "checkpoint_last.pt", {"epoch": 5})
    assert runs.find_resumable(tmp_path, "C2", "m0") == newer, "newest wins"

    assert runs.find_resumable(tmp_path, "C2", "nonexistent") is None
    assert runs.find_resumable(tmp_path, "ZZ", "m0") is None


def test_dirty_tree_is_warned_about(tmp_path: Path, caplog) -> None:
    """A result from uncommitted code cannot be reproduced from a commit."""
    import subprocess

    repo = tmp_path / "repo"
    (repo / "sub").mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "f.txt").write_text("one")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)
    (repo / "f.txt").write_text("modified but not committed")

    import os

    cwd = os.getcwd()
    os.chdir(repo)
    try:
        with caplog.at_level(logging.WARNING):
            runs.RunContext.create(
                phase="A2", name="dirty", config={}, results_root=tmp_path / "out"
            )
    finally:
        os.chdir(cwd)

    assert any("DIRTY" in r.message for r in caplog.records), (
        "an unreproducible run must be impossible to overlook"
    )
