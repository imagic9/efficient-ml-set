# Vendored third-party sources

DESIGN §11 requires the release bundle to build with **no network fetch and no
system JSON/YAML development package**. Anything the runtime needs is vendored
here as source, pinned, and hashed.

Each vendored dependency must carry:

- the exact upstream version and release URL;
- its licence file, unmodified;
- its SHA-256, recorded in the dependency manifest below;
- a note of why it is vendored rather than found via `find_package`.

## nlohmann/json 3.12.0

| | |
|---|---|
| File | `nlohmann/json.hpp` (single header, 953,436 bytes) |
| SHA-256 | `aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63` |
| Upstream | `https://github.com/nlohmann/json/releases/download/v3.12.0/json.hpp` |
| Licence | MIT, `nlohmann/LICENSE.MIT`, unmodified |
| Vendored | 2026-07-15, A4 |

**Why vendored rather than `find_package`.** The Pi bundle must build and run
without apt-installing a JSON development package (DESIGN §11), and the version
that would be found on the build host is not the version that would be found on the
Pi. A single header removes the question.

**Why now.** This directory previously said "expected first entry … added in E1 …
vendoring it now, before anything reads JSON, would commit ~900 KB in support of no
caller." A4 is that caller: it implements the policy loader, which reads the
`SHUTTER_TRIGGER` policy JSON described in DESIGN §4, and emits the structured
inference result. The condition the note set has been met.

**Hash provenance.** GitHub publishes no digest for this release asset (the API
returns `digest: null`), so the SHA-256 could not be cross-checked against an
upstream-published value the way ORT's was. Instead it was verified by fetching the
same header over an independent path — the tagged repository tree at
`raw.githubusercontent.com/nlohmann/json/v3.12.0/single_include/nlohmann/json.hpp`
— and confirming both are byte-identical. Two paths agreeing is weaker evidence than
a signed digest; it is what upstream makes available.

`scripts/verify_vendored.sh` re-checks the hash recorded here against the file on
disk, so a silent edit to a vendored header fails a gate rather than shipping.
