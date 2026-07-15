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
    "wildlife_trigger.provenance",
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
        "src/wildlife_trigger/provenance.py",
        "scripts/isa_probe.c",
        "tests/fixtures",
        "notebooks",
    ],
)
def test_design_section_14_path_exists(relative_path: str) -> None:
    assert (PROJECT_ROOT / relative_path).exists(), (
        f"DESIGN §14 requires {relative_path}"
    )


def test_no_authored_file_is_silently_ignored() -> None:
    """Catch the bug class where a bare directory rule swallows real source.

    This has now bitten twice. A bare `data/` matched `src/wildlife_trigger/data/`
    and `configs/data/`; a bare `env/` matched `configs/env/`, dropping the pinned
    version manifests that the Dockerfile and setup scripts read. Both were silent:
    `git add` simply skipped the files, and a clean clone would have failed far
    later, in a place that pointed nowhere near the cause.

    Rather than enumerate the paths we happen to remember, walk everything we
    authored and assert git tracks it. Generated and vendored trees are excluded
    by the prune list, so anything left is ours by construction.
    """
    import subprocess

    prune = {
        ".git", "build", "__pycache__", ".pytest_cache", ".ruff_cache",
        "node_modules", ".venv", "venv", "Docker_VSCode",
    }
    authored_suffixes = {
        ".py", ".cpp", ".hpp", ".c", ".h", ".txt", ".toml", ".yaml", ".yml",
        ".json", ".sh", ".env", ".cff", ".md",
    }

    def is_generated(parts: tuple[str, ...]) -> bool:
        # `pip install -e` drops *.egg-info full of .txt files that look authored
        # but are build metadata. Matched by suffix because the directory name
        # carries the package name and cannot be listed literally.
        return bool(prune & set(parts)) or any(p.endswith(".egg-info") for p in parts)

    candidates: list[Path] = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if is_generated(path.relative_to(PROJECT_ROOT).parts):
            continue
        if path.suffix not in authored_suffixes:
            continue
        candidates.append(path)

    assert candidates, "found no authored files; the walk itself is broken"

    ignored = subprocess.run(
        ["git", "check-ignore", "--stdin"],
        cwd=PROJECT_ROOT,
        input="\n".join(str(p) for p in candidates),
        capture_output=True,
        text=True,
    ).stdout.split()

    relative = sorted(str(Path(p).relative_to(PROJECT_ROOT)) for p in ignored)
    assert not relative, (
        "these authored files are ignored by .gitignore and would vanish from a "
        f"clean clone: {relative}"
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


def test_no_credential_shaped_string_is_tracked() -> None:
    """Filenames are the easy half; the content is what actually leaks.

    The repository is public, so a committed token is compromised the moment it
    is pushed and stays in history after any 'fix'. This is cheap enough to run
    on every commit and catastrophic enough to be worth it.
    """
    import subprocess

    patterns = [
        r"(ghp_|gho_|ghs_|github_pat_)[A-Za-z0-9]{20,}",  # GitHub tokens
        r"AKIA[0-9A-Z]{16}",  # AWS access key id
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",  # any private key
        r"xox[baprs]-[A-Za-z0-9-]{10,}",  # Slack
    ]

    hits: list[str] = []
    for pattern in patterns:
        found = subprocess.run(
            ["git", "grep", "-nIE", pattern],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        # git grep exits 1 when there are no matches, which is the good case.
        if found.returncode == 0 and found.stdout.strip():
            hits.extend(found.stdout.strip().splitlines())

    assert not hits, f"credential-shaped strings are tracked: {hits}"
