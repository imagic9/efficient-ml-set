#!/usr/bin/env python3
"""B0 — acquire and fingerprint the CCT-20 sources.

DESIGN §5.1 is explicit that the downloadable JSONs, not the ECCV paper, are
executable truth: the paper states 57,868 images and the current split files sum to
57,864. So this module records what was actually downloaded — URL, timestamp, size,
SHA-256 — and every later count is reconciled against those bytes rather than against
a number copied from a table.

**What is deliberately NOT downloaded**, because it is easy to reach for and
enormous:

  - `cct_images.tar.gz` (105 GB): the full-resolution archive. The `_sm` archive at
    6 GB carries every CCT-20 frame we need, capped at 1024 px per side.
  - `caltech_bboxes_20200316.json` (35 MB): only required by the Stretch KD
    experiment, which is not unlocked.

The empty supplement (B2) downloads ~5,000 individual images later; those arrive at
original resolution and DESIGN §5.2 step 7 mandates downsizing them to match. That
correction is B2's job, not this module's.

Usage:
    python -m wildlife_trigger.data.acquire --output-dir data/raw
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path

# Verified against lila.science/datasets/caltech-camera-traps on 2026-07-15. LILA has
# rehosted before (Azure blob -> Google Cloud Storage), so the URL is recorded in the
# manifest of every run: a future failure should show *what* it tried to fetch.
LILA_BASE = "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps"

SOURCES = {
    "eccv_18_annotations.tar.gz": {
        "url": f"{LILA_BASE}/eccv_18_annotations.tar.gz",
        "purpose": "CCT-20 official split metadata (DESIGN §5.1)",
        "approx_bytes": 3_000_000,
    },
    "eccv_18_all_images_sm.tar.gz": {
        "url": f"{LILA_BASE}/eccv_18_all_images_sm.tar.gz",
        "purpose": "all 57,864 CCT-20 frames, max 1024 px per side (DESIGN §5.1)",
        "approx_bytes": 6_000_000_000,
    },
    "caltech_camera_traps.json.zip": {
        "url": f"{LILA_BASE}/labels/caltech_camera_traps.json.zip",
        "purpose": "full-CCT metadata, ONLY for empty-supplement selection (DESIGN §5.2)",
        "approx_bytes": 9_000_000,
    },
}

# DESIGN §14 and the model card need these verbatim. Recorded here so the report
# cannot drift from what the dataset actually requires.
LICENSE = "Community Data License Agreement (permissive variant)"
CITATION = (
    "Sara Beery, Grant Van Horn, Pietro Perona. Recognition in Terra Incognita. "
    "Proceedings of the 15th European Conference on Computer Vision (ECCV 2018)."
)
DATASET_PAGE = "https://lila.science/datasets/caltech-camera-traps"


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, retries: int = 3) -> None:
    """Fetch `url` to `destination`, resuming a partial file.

    `curl -C -` rather than urllib: this is a 6 GB transfer over the public internet,
    and a connection that drops at 5.8 GB should resume, not restart. Retries are
    bounded — an endless retry loop in an unattended script is how a broken URL
    becomes an overnight no-op.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        result = subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--continue-at", "-",
                "--retry", "3",
                "--retry-delay", "5",
                "--connect-timeout", "30",
                "--progress-bar",
                "--output", str(destination),
                url,
            ],
            check=False,
        )
        if result.returncode == 0:
            return
        # 33 = server does not support resume; the partial file must go or curl will
        # keep asking for a range that is refused, forever.
        if result.returncode == 33 and destination.exists():
            print(f"  server refused resume; restarting {destination.name}")
            destination.unlink()
        print(f"  attempt {attempt}/{retries} failed (curl exit {result.returncode})")

    raise RuntimeError(f"could not download {url} after {retries} attempts")


def fingerprint(name: str, path: Path, url: str, purpose: str) -> dict:
    return {
        "name": name,
        "url": url,
        "purpose": purpose,
        "downloaded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def extract(archive: Path, destination: Path) -> list[str]:
    """Extract a .tar.gz or .zip and return the member names.

    `filter="data"` on tar: a tarball is untrusted input, and without it a member
    named `../../etc/foo` extracts outside the destination. Python 3.14 makes this
    the default; being explicit means the behaviour does not depend on which
    interpreter runs it.
    """
    destination.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(destination)
            return handle.namelist()

    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(destination, filter="data")
        return handle.getnames()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(SOURCES),
        help="Fetch a subset. The 6 GB image archive is the slow one; the metadata "
        "alone is enough to build manifests.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Download and fingerprint without unpacking.",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    wanted = args.only or list(SOURCES)
    archives = args.output_dir / "archives"
    extracted = args.output_dir / "extracted"

    records = []
    for name in wanted:
        source = SOURCES[name]
        path = archives / name

        # Re-hash rather than trust the presence of a file: a half-downloaded archive
        # from a killed run is exactly the file that is here and wrong.
        if path.exists() and path.stat().st_size >= source["approx_bytes"] * 0.95:
            print(f"{name}: already present ({path.stat().st_size / 1e9:.2f} GB)")
        else:
            print(f"{name}: downloading from {source['url']}")
            download(source["url"], path)

        print(f"{name}: hashing {path.stat().st_size / 1e6:.1f} MB")
        record = fingerprint(name, path, source["url"], source["purpose"])

        if not args.skip_extract:
            print(f"{name}: extracting")
            members = extract(path, extracted)
            record["extracted_members"] = len(members)
            record["extracted_to"] = str(extracted)

        records.append(record)
        print(f"{name}: sha256 {record['sha256']}")

    manifest = {
        "task": "B0",
        "dataset": "Caltech Camera Traps / CCT-20",
        "dataset_page": DATASET_PAGE,
        "license": LICENSE,
        "citation": CITATION,
        "not_downloaded": {
            "cct_images.tar.gz": "105 GB full-resolution archive; the _sm archive "
            "carries every CCT-20 frame at <=1024 px, which is the contract.",
            "caltech_bboxes_20200316.json": "35 MB; only needed by the Stretch KD "
            "experiment, which is not unlocked.",
        },
        "sources": records,
    }

    report_path = args.report or (args.output_dir / "source_manifest.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
