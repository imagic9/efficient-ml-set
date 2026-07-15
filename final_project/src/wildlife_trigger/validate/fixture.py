#!/usr/bin/env python3
"""Write the raw float32 blob both call sites read.

P0 compares what Python ORT and C++ ORT produce. That comparison only means
anything if both were handed the same bytes: two independently generated "random"
inputs would differ, the outputs would differ, and the difference would look like
a runtime discrepancy instead of what it is.

Raw little-endian float32, C order, no header — the format `numpy.tofile` writes
and `std::ifstream::read` consumes, chosen because it needs no library on either
side. The C++ probe checks the file's byte count against the model's expected
element count and refuses a mismatch rather than reading garbage.

Usage:
    python -m wildlife_trigger.validate.fixture --output x.bin --shape 1 3 224 224
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np


def write_fixture(output: Path, shape: tuple[int, ...], seed: int = 0) -> dict:
    """Write a deterministic normal(0,1) blob and report its hash.

    normal(0,1) approximates the range ImageNet-normalised pixels occupy, so the
    activations are plausible and every kernel is exercised. A fixed seed makes the
    file reproducible; the SHA-256 is recorded so a later run can prove it read the
    same fixture rather than assuming the seed still means the same thing.
    """
    rng = np.random.default_rng(seed)
    data = rng.standard_normal(shape, dtype=np.float32)

    output.parent.mkdir(parents=True, exist_ok=True)
    data.tofile(output)

    return {
        "path": str(output),
        "shape": list(shape),
        "dtype": "float32",
        "seed": seed,
        "elements": int(data.size),
        "bytes": output.stat().st_size,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shape", required=True, type=int, nargs="+")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    info = write_fixture(args.output, tuple(args.shape), args.seed)
    print(
        f"    fixture {info['path']} "
        f"({info['elements']} float32, sha256 {info['sha256'][:12]}...)",
        file=sys.stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
