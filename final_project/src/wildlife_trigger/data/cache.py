#!/usr/bin/env python3
"""B3 — the offline preprocessing cache (DESIGN §5.5).

Steps 1-4 of the preprocessing contract are deterministic and depend only on the source
pixels and the configured geometry, so they are computed once into a memmapped array of
uint8 letterboxes. Training reads that instead of decoding 57,864 JPEGs on every epoch of
every run — and DESIGN needs at least three training runs for the input-shape control
alone, before M0-M4.

## The condition this rests on

**The augmentation list contains no random crop and no random resize** (DESIGN §5.5:
"no crop that can exclude the labelled animal"). Flip, jitter, grayscale and blur all
operate on the final-size tensor, so caching the final letterbox destroys no
augmentation entropy. If a `RandomResizedCrop` is ever added, this cache silently
freezes the crop and must be removed. That is the deal.

## Why the fingerprint is the whole design

A cache that outlives the config that produced it trains on stale pixels. Nothing
downstream can detect that: it is not a crash, it is a quietly wrong number. So the
cache stores the preprocessing fingerprint and the manifest's own hash beside the
pixels, and `open_cache` refuses to return an array whose fingerprint does not match
what the caller asked for. Rebuilding an hour of CPU is cheap; a silently wrong M0 is
not.

## Layout

  `<cache>/<manifest>-<width>x<height>/pixels.npy`  uint8 [N, H, W, 3], memmapped
  `<cache>/<manifest>-<width>x<height>/meta.json`   fingerprint, manifest hash, order

8.5 GB for 57,864 frames at 256x192. gx10 has 117 GB of RAM, so after the first epoch it
lives in page cache and dataloading effectively disappears.

Usage:
    python -m wildlife_trigger.data.cache --manifest data/manifests/train.jsonl \
        --images-dir data/raw/extracted/eccv_18_all_images_sm --cache-dir data/cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from wildlife_trigger.data.preprocess import PreprocessConfig, decode, letterbox_bgr
from wildlife_trigger.runs import sha256_file

CACHE_FORMAT_VERSION = 2

__all__ = ["build", "cache_dir_for", "fingerprint", "open_cache", "sha256_file"]


def fingerprint(config: PreprocessConfig, manifest_sha256: str) -> dict:
    return {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "preprocess": config.fingerprint(),
        "manifest_sha256": manifest_sha256,
    }


def cache_dir_for(cache_root: Path, manifest: Path, config: PreprocessConfig) -> Path:
    # Shape is in the directory name because the 224x224-versus-256x192 control needs
    # both caches to coexist (DESIGN §5.5).
    return cache_root / f"{manifest.stem}-{config.width}x{config.height}"


def _letterbox_one(args: tuple[str, int, int, int]) -> np.ndarray:
    """Worker: decode + letterbox one file. Top-level so it can be pickled."""
    path, width, height, pad_value = args
    config = PreprocessConfig(width=width, height=height, pad_value=pad_value)
    letterbox, _ = letterbox_bgr(decode(path), config)
    return letterbox


def build(
    manifest: Path,
    images_dir: Path,
    cache_root: Path,
    config: PreprocessConfig,
    workers: int = 8,
    force: bool = False,
) -> dict:
    records = [json.loads(l) for l in manifest.read_text().splitlines()]
    manifest_hash = sha256_file(manifest)
    target = cache_dir_for(cache_root, manifest, config)
    meta_path = target / "meta.json"
    pixels_path = target / "pixels.npy"

    expected = fingerprint(config, manifest_hash)
    if meta_path.exists() and pixels_path.exists() and not force:
        existing = json.loads(meta_path.read_text())
        if existing.get("fingerprint") == expected:
            print(f"cache is current: {target} ({existing['images']} images)")
            return existing
        print("cache fingerprint differs from the requested config; rebuilding")

    target.mkdir(parents=True, exist_ok=True)
    array = np.lib.format.open_memmap(
        pixels_path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(records), config.height, config.width, 3),
    )

    tasks = [
        (str(images_dir / r["file_name"]), config.width, config.height, config.pad_value)
        for r in records
    ]

    print(f"building {target}: {len(records)} images, {workers} workers")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        # chunksize amortises IPC: 57,864 individual dispatches cost more than the
        # decode itself.
        for index, letterbox in enumerate(pool.map(_letterbox_one, tasks, chunksize=64)):
            array[index] = letterbox
            if (index + 1) % 5000 == 0:
                print(f"  {index + 1}/{len(records)}", flush=True)
    array.flush()

    meta = {
        "fingerprint": expected,
        "images": len(records),
        "manifest": str(manifest),
        "shape": [len(records), config.height, config.width, 3],
        "dtype": "uint8",
        "bytes": pixels_path.stat().st_size,
        # The row order IS the manifest order. Stored so a reader can assert it rather
        # than assume it: a cache row silently paired with the wrong label is the worst
        # bug available here, and it would look like a model that just does not learn.
        "image_id_order_sha256": hashlib.sha256(
            "\n".join(r["image_id"] for r in records).encode()
        ).hexdigest(),
        "contains": "steps 1-4 only (decode, BGR->RGB, aspect-preserving resize, pad). "
        "Normalisation and augmentation run per batch on top of this.",
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {pixels_path} ({meta['bytes'] / 1e9:.2f} GB)")
    return meta


def open_cache(
    manifest: Path, cache_root: Path, config: PreprocessConfig
) -> tuple[np.ndarray, dict]:
    """Return the memmapped pixels, or refuse.

    Refusing is the feature. A stale cache produces a plausible training curve against
    pixels that no longer match the config, and no metric can see it.
    """
    target = cache_dir_for(cache_root, manifest, config)
    meta_path = target / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"no cache at {target}. Build it with wildlife_trigger.data.cache."
        )

    meta = json.loads(meta_path.read_text())
    expected = fingerprint(config, sha256_file(manifest))
    if meta.get("fingerprint") != expected:
        raise RuntimeError(
            f"cache at {target} is stale.\n"
            f"  cached : {json.dumps(meta.get('fingerprint'), sort_keys=True)}\n"
            f"  wanted : {json.dumps(expected, sort_keys=True)}\n"
            "The pixels were produced by a different preprocessing config or a "
            "different manifest. Rebuild it — training on these would be silently "
            "wrong rather than visibly broken."
        )

    pixels = np.load(target / "pixels.npy", mmap_mode="r")
    records = [json.loads(l) for l in manifest.read_text().splitlines()]
    if len(pixels) != len(records):
        raise RuntimeError(
            f"cache holds {len(pixels)} rows but the manifest lists {len(records)}"
        )
    order = hashlib.sha256("\n".join(r["image_id"] for r in records).encode()).hexdigest()
    if meta.get("image_id_order_sha256") != order:
        raise RuntimeError(
            "the manifest's image order does not match the cache's. Every row would be "
            "paired with the wrong label."
        )
    return pixels, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = PreprocessConfig(width=args.width, height=args.height)
    meta = build(
        args.manifest, args.images_dir, args.cache_dir, config, args.workers, args.force
    )
    print(json.dumps({k: v for k, v in meta.items() if k != "fingerprint"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
