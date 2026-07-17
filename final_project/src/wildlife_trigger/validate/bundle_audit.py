#!/usr/bin/env python3
"""Audit the staged deployment bundle: completeness, checksums, and target glibc.

The bundle is what actually reaches the Pi, and the Pi trip is one-shot and
unscheduled. Every check here answers a question that is free to ask now and
expensive to discover in the field:

  - is every file the launcher needs actually staged, including the soname symlinks
    the loader follows (a dangling link stages fine and fails at exec);
  - does MANIFEST.sha256 verify, so what arrives can be proved to be what left;
  - does every ELF object need at most GLIBC 2.36? gx10 is 2.39, and a binary built
    natively there requests symbols Pi OS Bookworm's loader simply does not have.

Runs the ELF checks inside the target container, because that is where the objects'
own toolchain lives — `objdump` on the host would be a different binutils inspecting
a cross-target object.

Usage:
    python -m wildlife_trigger.validate.bundle_audit --bundle B --project-root R ...
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

# Pi OS Bookworm ships glibc 2.36 (measured, A2). Any versioned symbol above this is
# a binary the Pi's loader refuses outright — not a warning, a hard failure at exec.
MAX_TARGET_GLIBC = (2, 36)

# What a working bundle must contain. The symlinks matter as much as the payload: the
# loader looks for libonnxruntime.so.1, not for the versioned filename.
REQUIRED_PATHS = (
    "bin/wildlife_trigger",
    "bin/run.sh",
    "lib/libonnxruntime.so",
    "lib/libonnxruntime.so.1",
    "models/M0.onnx",
    "models/M2.onnx",
    "models/M4.onnx",
    "policies/class_map.json",
    "policies/M0.json",
    "policies/M2.json",
    "policies/M4.json",
    "data/manifest.jsonl",
    "preflight.sh",
    "install.sh",
    "run_demo.sh",
    "run_benchmark.sh",
    "README.md",
    "BUNDLE.json",
    "MANIFEST.sha256",
)

GLIBC_SYMBOL = re.compile(r"GLIBC_(\d+)\.(\d+)")


def in_container(project_root: Path, image_tag: str, command: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "run", "--rm", "-v", f"{project_root}:/work", "-w", "/work",
         image_tag, "bash", "-lc", command],
        capture_output=True,
        text=True,
        check=False,
    )


def max_glibc_requirement(
    project_root: Path, image_tag: str, rel_bundle: str
) -> tuple[tuple[int, int], dict]:
    """Highest GLIBC_x.y any staged ELF object asks for."""
    result = in_container(
        project_root,
        image_tag,
        f"cd /work/{rel_bundle} && find . -type f "
        f"\\( -name '*.so*' -o -perm -u+x \\) -exec sh -c "
        f"'file -b \"$1\" | grep -q ELF && echo \"== $1\" && "
        f"objdump -T \"$1\" 2>/dev/null | grep -o \"GLIBC_[0-9.]*\"' _ {{}} \\;",
    )

    highest = (0, 0)
    per_object: dict[str, str] = {}
    current = "<unknown>"
    for line in result.stdout.splitlines():
        if line.startswith("== "):
            current = line[3:]
            continue
        match = GLIBC_SYMBOL.fullmatch(line.strip())
        if not match:
            continue
        version = (int(match.group(1)), int(match.group(2)))
        highest = max(highest, version)
        existing = per_object.get(current, "0.0")
        if version > tuple(int(p) for p in existing.split(".")):
            per_object[current] = f"{version[0]}.{version[1]}"

    return highest, per_object


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--image-tag", required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    rel_bundle = str(args.bundle.relative_to(args.project_root))

    missing = [p for p in REQUIRED_PATHS if not (args.bundle / p).exists()]

    checksums = in_container(
        args.project_root,
        args.image_tag,
        f"cd /work/{rel_bundle} && sha256sum -c MANIFEST.sha256",
    )
    manifest = args.bundle / "MANIFEST.sha256"
    file_count = len(manifest.read_text().splitlines()) if manifest.exists() else 0

    highest, per_object = max_glibc_requirement(args.project_root, args.image_tag, rel_bundle)
    glibc_ok = highest <= MAX_TARGET_GLIBC

    report = {
        "bundle": str(args.bundle),
        "required_paths": list(REQUIRED_PATHS),
        "missing_paths": missing,
        "checksums_verified": checksums.returncode == 0,
        "file_count": file_count,
        "max_glibc": f"{highest[0]}.{highest[1]}",
        "max_glibc_allowed": f"{MAX_TARGET_GLIBC[0]}.{MAX_TARGET_GLIBC[1]}",
        "glibc_per_object": per_object,
        "passed": glibc_ok and not missing and checksums.returncode == 0,
        "note": (
            "The session-optimized graph is deliberately NOT bundled: ORT warns a "
            "graph serialized above ORT_ENABLE_EXTENDED is only valid in the "
            "environment that optimized it. The Pi optimizes the ordinary model "
            "itself. OpenCV is not bundled either — install.sh apt-installs the "
            "matching 4.6.0 runtime (E7 decision; Debian's imgcodecs closure is "
            "impractical to carry)."
        ),
    }

    print(f"    files staged      : {file_count}")
    print(f"    checksums verified: {report['checksums_verified']}")
    print(
        f"    max GLIBC needed  : {report['max_glibc']} "
        f"(target allows {report['max_glibc_allowed']}) -> "
        f"{'OK' if glibc_ok else 'TOO NEW FOR THE PI'}"
    )
    if missing:
        print(f"    MISSING: {missing}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
