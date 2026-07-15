#!/usr/bin/env python3
"""B2 steps 6-8 — fetch the selected empty frames and downsize them to match `_sm`.

**Step 7 is the whole point of this module.** LILA serves per-image downloads at
original resolution (~2048x1494); every CCT-20 split is capped at 1024 px. Without the
downsize, `empty` becomes the only training class carrying double resolution and a
distinct JPEG generation — a feature perfectly correlated with the label, absent from
validation and test, and therefore invisible exactly where it does its damage
(DESIGN §5.2).

The correction is applied here, once, and both checksums and both geometries are
recorded so the correction itself is auditable. `shortcut_probe` then tries to tell the
pools apart; near-chance is the evidence that this worked.

Downloads run concurrently because 5,000 sequential HTTPS round-trips are dominated by
latency, not bandwidth. Everything else about the operation is deliberately boring:
each image is verified, downsized deterministically, and recorded.

Usage:
    python -m wildlife_trigger.data.fetch_supplement \
        --manifest data/manifests/cct_empty_train_v1.jsonl \
        --output-dir data/images/empty_supplement
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from wildlife_trigger.data.empty_supplement import (
    JPEG_QUALITY,
    MAX_LONG_SIDE,
    RESAMPLE_FILTER,
)

# Politeness and self-defence: LILA is a public good, and an unbounded thread pool
# against it is both rude and a good way to get rate-limited mid-run.
MAX_WORKERS = 12
TIMEOUT_SECONDS = 60
RETRIES = 3


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def fetch(url: str) -> bytes:
    last: Exception | None = None
    for _ in range(RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
    raise RuntimeError(f"failed to fetch {url}: {last}")


def downsize(payload: bytes, destination: Path) -> dict:
    """Downsize to `MAX_LONG_SIDE` and write JPEG. Returns both geometries.

    An image already within the cap is re-encoded anyway rather than copied. That looks
    wasteful and is deliberate: copying would give those frames one JPEG generation
    while their neighbours get two, reintroducing at the file level the very encoding
    difference this step exists to remove. Uniform treatment beats minimal treatment.
    """
    from io import BytesIO

    with Image.open(BytesIO(payload)) as image:
        image = image.convert("RGB")
        original_size = image.size  # (width, height)

        scale = min(
            1.0,
            MAX_LONG_SIDE / max(original_size),
        )
        target = (
            max(1, round(original_size[0] * scale)),
            max(1, round(original_size[1] * scale)),
        )
        resized = (
            image.resize(target, getattr(Image.Resampling, RESAMPLE_FILTER))
            if scale < 1.0
            else image
        )

        destination.parent.mkdir(parents=True, exist_ok=True)
        resized.save(destination, format="JPEG", quality=JPEG_QUALITY)

    return {
        "original_width": original_size[0],
        "original_height": original_size[1],
        "downsized_width": target[0],
        "downsized_height": target[1],
        "was_downsized": scale < 1.0,
    }


def process(record: dict, output_dir: Path) -> dict:
    destination = output_dir / record["file_name"]
    result = dict(record)

    if destination.exists():
        # Resume: an interrupted 5,000-image run should not restart. The file is
        # re-measured rather than trusted, because a half-written JPEG from a killed
        # run is exactly the file that exists and is wrong.
        try:
            with Image.open(destination) as image:
                image.verify()
            with Image.open(destination) as image:
                size = image.size
            result.update(
                {
                    "downsized_width": size[0],
                    "downsized_height": size[1],
                    "downsized_sha256": sha256_file(destination),
                    "relative_path": str(destination.relative_to(output_dir.parent)),
                    "status": "already_present",
                }
            )
            return result
        except Exception:
            destination.unlink(missing_ok=True)

    payload = fetch(record["source_url"])
    geometry = downsize(payload, destination)

    result.update(geometry)
    result.update(
        {
            # Both checksums, per DESIGN §5.2 step 8: the original proves what LILA
            # served, the downsized proves what entered training.
            "original_sha256": sha256_bytes(payload),
            "original_bytes": len(payload),
            "downsized_sha256": sha256_file(destination),
            "downsized_bytes": destination.stat().st_size,
            "relative_path": str(destination.relative_to(output_dir.parent)),
            "resample_filter": RESAMPLE_FILTER,
            "jpeg_quality": JPEG_QUALITY,
            "status": "fetched",
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--limit", type=int, help="Fetch only the first N (debugging).")
    args = parser.parse_args()

    records = [json.loads(l) for l in args.manifest.read_text().splitlines()]
    if args.limit:
        records = records[: args.limit]
    print(f"fetching {len(records)} empty frames with {args.workers} workers")

    done: list[dict] = []
    failures: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process, record, args.output_dir): record for record in records
        }
        for index, future in enumerate(as_completed(futures), start=1):
            record = futures[future]
            try:
                done.append(future.result())
            except Exception as exc:
                failures.append(
                    {"image_id": record["image_id"], "error": f"{type(exc).__name__}: {exc}"}
                )
            if index % 250 == 0 or index == len(records):
                print(f"  {index}/{len(records)}  failures={len(failures)}", flush=True)

    done.sort(key=lambda r: r["image_id"])

    output_manifest = args.output_manifest or args.manifest
    output_manifest.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in done)
    )

    downsized = sum(1 for r in done if r.get("was_downsized"))
    long_sides = [max(r["downsized_width"], r["downsized_height"]) for r in done]
    report = {
        "task": "B2-fetch",
        "requested": len(records),
        "fetched": len(done),
        "failures": failures[:20],
        "failure_count": len(failures),
        "downsized": downsized,
        "already_within_cap": len(done) - downsized,
        "resample_filter": RESAMPLE_FILTER,
        "jpeg_quality": JPEG_QUALITY,
        "max_long_side_cap": MAX_LONG_SIDE,
        "observed_max_long_side": max(long_sides) if long_sides else 0,
        "cap_holds": max(long_sides) <= MAX_LONG_SIDE if long_sides else False,
        "total_downsized_bytes": sum(r.get("downsized_bytes", 0) for r in done),
        "manifest": str(output_manifest),
        "manifest_sha256": hashlib.sha256(output_manifest.read_bytes()).hexdigest(),
        "why_step_7": (
            "LILA serves per-image downloads at original resolution while CCT-20 is "
            "capped at 1024 px. Without this, `empty` would be the only training class "
            "carrying double resolution -- a label-correlated feature that is absent "
            "from val/test and therefore silently inflates the bobcat false-fire rate "
            "exactly where it is measured (DESIGN §5.2)."
        ),
    }

    print(
        f"\nfetched {report['fetched']}/{report['requested']}  "
        f"downsized {downsized}  failures {report['failure_count']}"
    )
    print(
        f"max long side after downsize: {report['observed_max_long_side']} "
        f"(cap {MAX_LONG_SIDE}) -> {'OK' if report['cap_holds'] else 'CAP VIOLATED'}"
    )
    print(f"total on disk: {report['total_downsized_bytes'] / 1e9:.2f} GB")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.report}")

    ok = report["cap_holds"] and report["failure_count"] == 0
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
