#!/usr/bin/env python3
"""Capture the execution environment as machine-readable provenance.

DESIGN §9.2 requires every run to record where it happened and with what. This
script is the single implementation of that capture, used by PLAN A0 for the
project start snapshot and reused by later phases for per-run provenance.

Never widen the environment capture to os.environ wholesale: it carries tokens
and dataset credentials. ENV_SAFELIST below is the only environment exposed.

Usage:
    python scripts/capture_provenance.py --output results/provenance/project_start.json \
        --label "A0 project start"
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path

# Only these environment variables are recorded. Anything else may hold secrets.
ENV_SAFELIST = (
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "PYTHONHASHSEED",
    "VIRTUAL_ENV",
)

# Packages whose versions matter to reproducibility. Absent is a valid answer and
# is recorded as null rather than omitted, so a later diff shows what appeared.
TRACKED_PACKAGES = (
    "torch",
    "torchvision",
    "numpy",
    "onnx",
    "onnxruntime",
    "onnxruntime-gpu",
    "torch-pruning",
    "opencv-python",
    "opencv-python-headless",
    "torchao",
    "pytorch-quantization",
    "pyyaml",
    "pillow",
)


def run(cmd: list[str], timeout: int = 20) -> str | None:
    """Return stripped stdout, or None if the command is unavailable or fails."""
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    text = out.stdout.strip()
    return text or None


def first_line(text: str | None) -> str | None:
    return text.splitlines()[0].strip() if text else None


def collect_cpu() -> dict:
    features: list[str] = []
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("Features"):
                features = line.split(":", 1)[1].split()
                break
    except OSError:
        pass

    model = None
    lscpu = run(["lscpu"])
    if lscpu:
        models = [
            l.split(":", 1)[1].strip()
            for l in lscpu.splitlines()
            if l.startswith("Model name")
        ]
        model = " + ".join(dict.fromkeys(models)) or None

    # These decide whether an INT8 kernel path exists at all, so they are called
    # out by name rather than left for a human to find in the flag soup.
    notable = {k: (k in features) for k in ("asimddp", "i8mm", "sve", "sve2", "bf16")}

    return {
        "model_name": model,
        "logical_cores": os.cpu_count(),
        "features": features or None,
        "notable_features": notable,
    }


def collect_memory() -> dict:
    info: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            if key in ("MemTotal", "MemAvailable", "MemFree"):
                info[key] = int(rest.split()[0])
    except (OSError, ValueError, IndexError):
        return {}
    return {k.lower() + "_gib": round(v / 1024 / 1024, 1) for k, v in info.items()}


def collect_gpu() -> dict:
    query = "name,driver_version,memory.total,memory.used,compute_cap,utilization.gpu"
    csv = run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"])
    gpus = []
    if csv:
        for row in csv.splitlines():
            parts = [p.strip() for p in row.split(",")]
            if len(parts) == 6:
                gpus.append(dict(zip(query.split(","), parts)))

    apps = run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ]
    )
    return {
        "gpus": gpus or None,
        "compute_apps": apps.splitlines() if apps else [],
        "cuda_toolkit": first_line(
            run(["/usr/local/cuda/bin/nvcc", "--version"]) or run(["nvcc", "--version"])
        ),
        "cuda_home": (
            str(Path("/usr/local/cuda").resolve())
            if Path("/usr/local/cuda").exists()
            else None
        ),
    }


def collect_toolchain() -> dict:
    return {
        "gcc": first_line(run(["gcc", "--version"])),
        "gpp": first_line(run(["g++", "--version"])),
        "cmake": first_line(run(["cmake", "--version"])),
        "glibc": first_line(run(["ldd", "--version"])),
        "git": first_line(run(["git", "--version"])),
        "docker": first_line(run(["docker", "--version"])),
        "persistent_jobs": {
            tool: shutil.which(tool) is not None
            for tool in ("tmux", "screen", "systemd-run", "nohup")
        },
    }


def collect_python() -> dict:
    import importlib.metadata as md

    packages: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = md.version(name)
        except md.PackageNotFoundError:
            packages[name] = None

    torch_info = None
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "device_capability": (
                list(torch.cuda.get_device_capability(0))
                if torch.cuda.is_available()
                else None
            ),
        }
    except Exception as exc:  # torch absent or broken; both are worth recording
        torch_info = {"error": f"{type(exc).__name__}: {exc}"}

    return {
        "executable": sys.executable,
        "version": platform.python_version(),
        "packages": packages,
        "torch": torch_info,
    }


def collect_git(repo: Path) -> dict:
    def git(*args: str) -> str | None:
        return run(["git", "-C", str(repo), *args])

    porcelain = git("status", "--porcelain")
    return {
        "repo_root": str(repo),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": git("rev-parse", "HEAD"),
        "commit_short": git("rev-parse", "--short", "HEAD"),
        "remote": git("remote", "get-url", "origin"),
        "dirty": bool(porcelain),
        "dirty_files": porcelain.splitlines() if porcelain else [],
    }


def collect_disk(path: Path) -> dict:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_gib": round(usage.total / 1024**3, 1),
        "used_gib": round(usage.used / 1024**3, 1),
        "free_gib": round(usage.free / 1024**3, 1),
    }


def collect_other_workloads() -> dict:
    """Record co-tenant load. DESIGN §12.4 forbids benchmarking beside one."""
    containers = run(
        ["docker", "ps", "--format", "{{.Names}}|{{.Image}}|{{.Status}}"]
    )
    load = None
    try:
        load = list(os.getloadavg())
    except OSError:
        pass
    return {
        "load_average": load,
        "docker_containers": containers.splitlines() if containers else [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--label", required=True, help="Human tag for this snapshot")
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root (default: infer from this file)",
    )
    args = parser.parse_args()

    snapshot = {
        "label": args.label,
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command_line": ["python"] + sys.argv,
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "os_release": dict(
                line.split("=", 1)
                for line in Path("/etc/os-release").read_text().splitlines()
                if "=" in line
            )
            if Path("/etc/os-release").exists()
            else None,
            "kernel": first_line(run(["uname", "-a"])),
        },
        "cpu": collect_cpu(),
        "memory": collect_memory(),
        "gpu": collect_gpu(),
        "toolchain": collect_toolchain(),
        "python": collect_python(),
        "git": collect_git(args.repo),
        "disk": collect_disk(args.repo),
        "other_workloads": collect_other_workloads(),
        "environment": {k: os.environ.get(k) for k in ENV_SAFELIST},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, indent=2, sort_keys=False) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
