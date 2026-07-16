"""Tests for run directories, provenance, logging and checkpoint/resume (A2).

These assert the properties later phases actually depend on, not the shape of the
API: that a run records enough to be explained afterwards, that a checkpoint
survives a kill mid-write, and that a resumable run can be found again.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pytest

from wildlife_trigger import runs

# The root-logger cleanup RunContext needs lives in conftest.py, autouse for every test.


def test_run_id_is_safe_as_a_directory_name() -> None:
    import datetime as dt

    when = dt.datetime(2026, 7, 15, 18, 30, 0, tzinfo=dt.timezone.utc)
    run_id = runs.make_run_id("C2", "m0/fp32 seed=42", when)

    assert run_id == "c2_m0_fp32_seed_42_20260715T183000Z"
    # A run name comes from a config and can contain anything; a '/' would silently
    # create a nested directory instead of the run.
    assert "/" not in run_id and " " not in run_id


def test_run_ids_sort_chronologically_within_a_name() -> None:
    """The ordering `find_resumable` depends on — and its exact limit.

    The id is `{phase}_{name}_{timestamp}`, so the name sorts before the time:
    ids for *different* names do not order by time at all. That is fine, because
    find_resumable filters to one name first, but it is worth pinning so nobody
    later assumes a global chronological sort that does not exist.
    """
    import datetime as dt

    when = dt.datetime(2026, 7, 15, 18, 30, 0, tzinfo=dt.timezone.utc)
    earlier = runs.make_run_id("C2", "m0", when - dt.timedelta(hours=1))
    later = runs.make_run_id("C2", "m0", when)
    assert earlier < later

    # Different names: ordering follows the name, not the clock.
    assert runs.make_run_id("C2", "zzz", when - dt.timedelta(days=9)) > later


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
    runs.save_checkpoint(older / "last.pt", {"epoch": 1})

    found = runs.find_resumable(tmp_path, "C2", "m0")
    assert found == older, "a run without a checkpoint is not resumable"

    runs.save_checkpoint(newer / "last.pt", {"epoch": 5})
    assert runs.find_resumable(tmp_path, "C2", "m0") == newer, "newest wins"

    assert runs.find_resumable(tmp_path, "C2", "nonexistent") is None
    assert runs.find_resumable(tmp_path, "ZZ", "m0") is None


def test_sha256_file_is_the_file_and_streams_it(tmp_path: Path) -> None:
    """Streamed, because a preprocessing cache is 8.5 GB and read_bytes() is not."""
    blob = tmp_path / "pixels.npy"
    payload = b"x" * ((1 << 20) + 7)  # crosses the 1 MiB block boundary, unevenly
    blob.write_bytes(payload)

    assert runs.sha256_file(blob) == hashlib.sha256(payload).hexdigest()


def test_record_hashes_merges_inputs_recorded_before_outputs_exist(tmp_path: Path) -> None:
    """A run hashes what it read at the start and what it wrote at the end.

    Overwriting on the second call would leave a run that died mid-training unable to
    say what it trained on, which is the case the record is most needed for.
    """
    ctx = runs.RunContext.create(
        phase="A2", name="hashes", config={}, results_root=tmp_path
    )
    manifest = tmp_path / "train.jsonl"
    manifest.write_text('{"image_id": "a"}\n')

    ctx.record_hashes({"manifest:train": manifest}, caches={"train": None})
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"weights")
    ctx.record_hashes({"checkpoint:best": checkpoint})

    payload = json.loads((ctx.run_dir / "hashes.json").read_text())
    assert payload["manifest:train"]["sha256"] == runs.sha256_file(manifest), (
        "the input hash must survive the output pass"
    )
    assert payload["manifest:train"]["bytes"] == manifest.stat().st_size
    assert payload["checkpoint:best"]["sha256"] == runs.sha256_file(checkpoint)
    assert payload["caches"] == {"train": None}


def test_record_hashes_writes_null_for_an_absent_input(tmp_path: Path) -> None:
    """DESIGN §5.2's no-empty arm has no supplement, and that is a recorded fact.

    Dropping the key would make "this run had no supplement" look identical to "nobody
    recorded whether it did".
    """
    ctx = runs.RunContext.create(
        phase="A2", name="null", config={}, results_root=tmp_path
    )
    ctx.record_hashes({"manifest:empty_supplement": None})

    payload = json.loads((ctx.run_dir / "hashes.json").read_text())
    assert "manifest:empty_supplement" in payload
    assert payload["manifest:empty_supplement"] is None


@pytest.fixture
def committed_repo(tmp_path: Path):
    """A git repo with one committed file and a results/ directory inside it."""
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "f.txt").write_text("one")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)
    return repo


def create_run_from(repo: Path, results_root: Path, name: str) -> runs.RunContext:
    """RunContext reads git from the cwd, so the cwd is part of the scenario."""
    import os

    cwd = os.getcwd()
    os.chdir(repo)
    try:
        return runs.RunContext.create(
            phase="A2", name=name, config={}, results_root=results_root
        )
    finally:
        os.chdir(cwd)


def test_dirty_tree_is_warned_about(committed_repo: Path, caplog) -> None:
    """A result from uncommitted code cannot be reproduced from a commit."""
    (committed_repo / "f.txt").write_text("modified but not committed")

    with caplog.at_level(logging.WARNING):
        ctx = create_run_from(committed_repo, committed_repo / "results", "dirty")

    assert any("DIRTY" in r.message for r in caplog.records), (
        "an unreproducible run must be impossible to overlook"
    )
    git = json.loads((ctx.run_dir / "provenance.json").read_text())["git"]
    assert git["reproducible_from_commit"] is False
    assert any("f.txt" in entry for entry in git["uncommitted_code"])


def test_a_run_does_not_call_itself_dirty(committed_repo: Path, caplog) -> None:
    """The bug that made the warning worthless the first time a real run used it.

    `results/` holds committed evidence rather than being ignored, so the run directory
    is an untracked change the moment it is created — and provenance was captured after
    creating it. Every run therefore reported "working tree is DIRTY", citing its own
    output. A warning that always fires is one nobody reads, so it must fire only on
    uncommitted *code*.
    """
    results = committed_repo / "results"
    with caplog.at_level(logging.WARNING):
        ctx = create_run_from(committed_repo, results, "clean")

    assert not any("DIRTY" in r.message for r in caplog.records), (
        "a run's own directory is not a reason to distrust the run"
    )
    git = json.loads((ctx.run_dir / "provenance.json").read_text())["git"]
    assert git["reproducible_from_commit"] is True

    # And the previous run's uncommitted output does not condemn the next one either.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        create_run_from(committed_repo, results, "second")
    assert not any("DIRTY" in r.message for r in caplog.records)


def test_the_console_handler_is_actually_attached(tmp_path: Path, capsys) -> None:
    """FileHandler IS a StreamHandler, and that swallowed every run's terminal output.

    The guard `if not any(isinstance(h, StreamHandler) ...)` was satisfied by the
    FileHandler added two lines above it, so the console handler was never added: a
    detached run's nohup log came out empty while training ran fine, and progress was
    visible only inside the run directory.

    Root's handlers are cleared first because pytest's own log-capture handler is also a
    StreamHandler subclass — under it the guard skips the console handler for a real
    reason, and the test would pass or fail on the harness rather than on the code.
    """
    root = logging.getLogger()
    saved = list(root.handlers)
    for handler in saved:
        root.removeHandler(handler)
    try:
        runs.RunContext.create(
            phase="A2", name="console", config={}, results_root=tmp_path
        )
        logging.getLogger("wildlife_trigger.selftest").info("epoch 0 done")
        assert "epoch 0 done" in capsys.readouterr().out, (
            "a long run's progress must reach the terminal watching it"
        )
    finally:
        for handler in saved:
            root.addHandler(handler)
