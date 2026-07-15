"""Smoke tests for the package skeleton (PLAN A1).

These are deliberately about structure rather than behaviour: A1's gate is that a
clean checkout has an understandable structure and its test suite runs. Real
behaviour arrives with the modules in phases B-D.

The import test is not a tautology. DESIGN §14 puts the package at
`src/wildlife_trigger/`, and the repository `.gitignore` previously matched a bare
`data/` at any depth, which silently excluded `src/wildlife_trigger/data/`. A test
that imports every subpackage fails loudly if that regresses.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

SUBPACKAGES = [
    "wildlife_trigger",
    "wildlife_trigger.data",
    "wildlife_trigger.models",
    "wildlife_trigger.optimize",
    "wildlife_trigger.reporting",
    "wildlife_trigger.validate",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    module = importlib.import_module(name)
    assert module.__doc__, f"{name} has no docstring explaining what belongs in it"


def test_version_is_exposed() -> None:
    import wildlife_trigger

    assert wildlife_trigger.__version__ == "0.1.0"


@pytest.mark.parametrize(
    "relative_path",
    [
        "DESIGN.md",
        "PLAN.md",
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "SUBMISSION.md",
        "pyproject.toml",
        "configs/data",
        "configs/train",
        "configs/optimize",
        "configs/runtime",
        "cpp/CMakeLists.txt",
        "data/README.md",
        "artifacts/README.md",
        "deploy/pi/README.md",
        "results/provenance",
        "scripts/capture_provenance.py",
        "scripts/isa_probe.c",
        "tests/fixtures",
        "notebooks",
    ],
)
def test_design_section_14_path_exists(relative_path: str) -> None:
    assert (PROJECT_ROOT / relative_path).exists(), (
        f"DESIGN §14 requires {relative_path}"
    )


def test_no_dataset_or_key_material_is_tracked() -> None:
    """Guard the two mistakes that are only noticed once they are public.

    Large archives bloat the repository irreversibly, and key material must never
    reach a public remote. Both are cheap to assert and expensive to undo.
    """
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()

    forbidden_suffixes = (".tar.gz", ".zip", ".pt", ".pth", ".pub", ".pem")
    offenders = [
        f
        for f in tracked
        if f.endswith(forbidden_suffixes) or "id_ed25519" in f or "id_rsa" in f
    ]
    assert not offenders, f"must not be tracked: {offenders}"
