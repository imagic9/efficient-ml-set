"""Run directories, provenance, persistent logging and checkpoint/resume.

Every phase from B onward writes through here, so DESIGN §9.2's requirement --
that each run record its resolved config, git state, environment, seeds and
hashes -- is satisfied by construction rather than by remembering to do it.

That sentence was false for a month (issue #10): `train.py` produced all three
C1a arms without importing this module, so the three run directories that decided
the input contract hold no environment and no hashes at all. A guarantee only one
caller has to remember is not a guarantee, which is the whole argument for this
module existing. `train.py` now writes through here; the C1a directories stay as
they are, and their provenance gap is what it is.

Two properties matter more than convenience:

*Persistence.* Long jobs on gx10 outlive the ssh session that started them. Logs
go to a file inside the run directory, not only to a terminal that will close.

*Resumability.* A training run that dies at epoch 25 of 30 must not restart from
zero. Checkpoints are written atomically, because a process killed mid-write
leaves a truncated file that fails to load -- which is exactly when you need it.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wildlife_trigger import provenance

LOGGER = logging.getLogger(__name__)

_RUN_ID_TIME_FORMAT = "%Y%m%dT%H%M%SZ"

# One name per artifact, project-wide. `train.py` wrote these before this module was
# wired in, `dump_predictions` defaults to `best.pt`, and DESIGN/PLAN name them --
# so the run directory keeps them and this module stops proposing a second spelling.
BEST_CHECKPOINT = "best.pt"
LAST_CHECKPOINT = "last.pt"


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def make_run_id(phase: str, name: str, when: dt.datetime | None = None) -> str:
    """`c2_m0_fp32_seed42_20260715T183000Z` — sorts chronologically per phase."""
    stamp = (when or utc_now()).strftime(_RUN_ID_TIME_FORMAT)
    safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name)
    return f"{phase.lower()}_{safe}_{stamp}"


def sha256_file(path: Path) -> str:
    """Hash a file's bytes, streamed.

    Streamed because the files a run needs to fingerprint are not all small: a
    checkpoint is ~9 MB and a preprocessing cache is 8.5 GB, and `read_bytes()` on
    the latter would ask gx10 for the whole array at once.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON via a temp file and rename, so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def changes_outside(git: dict, ignored_root: Path) -> list[str]:
    """Porcelain entries that are not under `ignored_root`.

    The dirty warning is about *code*: a result produced from uncommitted code cannot be
    reproduced from a commit. But `results/` holds committed evidence rather than being
    ignored, so every run directory is an untracked change until someone commits it —
    starting with the run's own. Counting those makes every run dirty, and a warning that
    always fires is one the reader learns to skip.
    """
    repo_root = git.get("repo_root")
    if not repo_root:
        return list(git.get("dirty_files", []))

    root = ignored_root.resolve()
    outside = []
    for entry in git.get("dirty_files", []):
        # " M path", "?? path/", "R  old -> new"; a rename is judged by its destination.
        path = entry[3:].strip().split(" -> ")[-1].strip('"')
        if not (Path(repo_root) / path).resolve().is_relative_to(root):
            outside.append(entry)
    return outside


@dataclass
class RunContext:
    """One execution of one stage, with everything needed to explain it later."""

    run_id: str
    run_dir: Path
    phase: str
    config: dict[str, Any]
    results_root: Path | None = None
    started_at: dt.datetime = field(default_factory=utc_now)

    @classmethod
    def create(
        cls,
        phase: str,
        name: str,
        config: dict[str, Any],
        results_root: Path,
        capture_provenance: bool = True,
    ) -> RunContext:
        run_id = make_run_id(phase, name)
        run_dir = results_root / phase.lower() / run_id

        # Read the git state *before* the directory exists. Otherwise the first thing
        # every run sees is its own empty directory as an untracked change.
        git = provenance.collect_git(Path.cwd()) if capture_provenance else None

        run_dir.mkdir(parents=True, exist_ok=True)
        ctx = cls(
            run_id=run_id,
            run_dir=run_dir,
            phase=phase,
            config=dict(config),
            results_root=results_root,
        )
        ctx._setup_logging()
        ctx._write_resolved_config()
        if capture_provenance:
            ctx._write_provenance(git)

        LOGGER.info("run %s started in %s", run_id, run_dir)
        return ctx

    # -- outputs --------------------------------------------------------------

    @property
    def log_path(self) -> Path:
        return self.run_dir / "run.log"

    @property
    def checkpoint_path(self) -> Path:
        return self.run_dir / LAST_CHECKPOINT

    @property
    def best_checkpoint_path(self) -> Path:
        return self.run_dir / BEST_CHECKPOINT

    def artifact(self, relative: str) -> Path:
        path = self.run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def record_hashes(
        self, files: Mapping[str, Path | str | None], **extra: Any
    ) -> Path:
        """Fingerprint what this run read and wrote (DESIGN §9.2).

        Merged into `hashes.json` rather than overwriting it, because the inputs are
        known before the first step and the outputs only after the last: a run that
        dies at epoch 25 should still be able to say which manifests it trained on.

        A `None` path is recorded as null, not dropped. "This run had no empty
        supplement" is a fact about the run; an absent key is indistinguishable from
        nobody having looked.
        """
        path = self.run_dir / "hashes.json"
        payload = json.loads(path.read_text()) if path.exists() else {}

        for label, target in files.items():
            if target is None:
                payload[label] = None
                continue
            target = Path(target)
            payload[label] = {
                "path": str(target),
                "sha256": sha256_file(target),
                "bytes": target.stat().st_size,
            }
        payload.update(extra)

        atomic_write_json(path, payload)
        return path

    # -- setup ----------------------------------------------------------------

    def _setup_logging(self) -> None:
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )
        fmt.converter = __import__("time").gmtime  # UTC everywhere, no local time

        file_handler = logging.FileHandler(self.log_path)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

        # `not isinstance(h, FileHandler)` is load-bearing: FileHandler *is* a
        # StreamHandler, so the plain check was satisfied by the handler added on the
        # line above and the console one was never attached. Every run through here
        # printed nothing to its terminal -- and nohup captured an empty file.
        if not any(
            isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
            for h in root.handlers
        ):
            stream = logging.StreamHandler(sys.stdout)
            stream.setFormatter(fmt)
            root.addHandler(stream)

    def _write_resolved_config(self) -> None:
        atomic_write_json(
            self.run_dir / "resolved_config.json",
            {
                "run_id": self.run_id,
                "phase": self.phase,
                "started_at_utc": self.started_at.isoformat(),
                "command_line": sys.argv,
                "config": self.config,
            },
        )

    def _write_provenance(self, git: dict) -> None:
        """Reuse the A0 capture rather than re-implementing a partial version.

        `git` is collected by `create` before the run directory exists, so it describes
        the tree this run started from rather than the tree this run just modified.
        """
        # Uncommitted results are outputs; uncommitted code is the thing that makes a
        # result unreproducible. Both are recorded; only the second is warned about.
        code_changes = (
            changes_outside(git, self.results_root)
            if self.results_root is not None
            else list(git.get("dirty_files", []))
        )

        snapshot = {
            "label": f"run {self.run_id}",
            "captured_at_utc": utc_now().isoformat(),
            "host": {"hostname": __import__("socket").gethostname()},
            "cpu": provenance.collect_cpu(),
            "memory": provenance.collect_memory(),
            "gpu": provenance.collect_gpu(),
            "toolchain": provenance.collect_toolchain(),
            "python": provenance.collect_python(),
            "git": {
                **git,
                "uncommitted_code": code_changes,
                "reproducible_from_commit": not code_changes,
            },
            "other_workloads": provenance.collect_other_workloads(),
        }
        atomic_write_json(self.run_dir / "provenance.json", snapshot)

        if code_changes:
            # Not fatal: a dirty tree is normal while developing. But a result
            # produced from uncommitted code cannot be reproduced from a commit,
            # so it must be impossible to overlook later.
            LOGGER.warning(
                "working tree is DIRTY at %s; this run is not reproducible from a "
                "commit alone (%d uncommitted file(s) outside %s): %s",
                git.get("commit_short"),
                len(code_changes),
                self.results_root,
                ", ".join(code_changes[:5]),
            )

    # -- results --------------------------------------------------------------

    def write_metrics(self, metrics: dict[str, Any]) -> Path:
        path = self.run_dir / "metrics.json"
        atomic_write_json(path, metrics)
        return path

    def finish(self, status: str = "completed", **extra: Any) -> None:
        elapsed = (utc_now() - self.started_at).total_seconds()
        atomic_write_json(
            self.run_dir / "run_summary.json",
            {
                "run_id": self.run_id,
                "status": status,
                "started_at_utc": self.started_at.isoformat(),
                "finished_at_utc": utc_now().isoformat(),
                "elapsed_seconds": round(elapsed, 3),
                **extra,
            },
        )
        LOGGER.info("run %s %s in %.1fs", self.run_id, status, elapsed)


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Atomically persist arbitrary training state.

    Atomic because the interesting failure is a process killed mid-write: the
    partial file loads as corrupt precisely when it is the thing you were
    counting on. Write beside the target, then rename.
    """
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path) -> dict[str, Any] | None:
    """Return the checkpoint, or None if absent. A corrupt file raises."""
    import torch

    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def find_resumable(results_root: Path, phase: str, name: str) -> Path | None:
    """Newest run directory for this phase/name that holds a last checkpoint."""
    phase_dir = results_root / phase.lower()
    if not phase_dir.is_dir():
        return None
    candidates = sorted(
        (d for d in phase_dir.iterdir() if d.is_dir() and name in d.name),
        reverse=True,
    )
    for candidate in candidates:
        if (candidate / LAST_CHECKPOINT).exists():
            return candidate
    return None


def run_detached(command: list[str], log_path: Path, cwd: Path | None = None) -> int:
    """Start a long job that survives this ssh session; return its pid.

    gx10 work is driven over ssh, and a dropped connection would otherwise take
    the training with it. setsid detaches from the controlling terminal so the
    job outlives the shell.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("ab")
    process = subprocess.Popen(
        command,
        stdout=handle,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        start_new_session=True,
        close_fds=True,
    )
    LOGGER.info("detached pid %d -> %s", process.pid, log_path)
    return process.pid
