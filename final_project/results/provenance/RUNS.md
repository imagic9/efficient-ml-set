# Run log

Append-only. One entry per executed run, newest last. `Handoff/` is gitignored and
therefore local-only; this file is the public record.

Every entry names the machine-readable evidence it produced, so a claim in the
report can be traced back to a file without reading prose.

| UTC | Phase/task | Host | Commit | What ran | Evidence |
|---|---|---|---|---|---|
| 2026-07-15T16:52Z | A0 | gx10 | `45117a9` | Environment capture, gx10 as found: boreal LLM stack running, 34.7 GiB RAM available, 68,078 MiB GPU held | `project_start.json` @ `05dccde` |
| 2026-07-15T16:58Z | A0 | gx10 | `8ca765a` | Stopped boreal LLM stack (`docker compose -f /data/v01/infra/docker-compose.llm.yml stop`); all 5 containers exited 0. RAM available 34.7 -> 117.8 GiB, GPU held 68,078 -> 176 MiB | this file |
| 2026-07-15T16:59Z | A0 | gx10 | `8ca765a` | Re-captured environment, gx10 dedicated | `results/provenance/project_start.json` |
| 2026-07-15T17:18Z | A0 | gx10 | `84fe66e` | ISA probe: native vs `qemu -cpu cortex-a76` vs `-cpu cortex-a72`. Confirms A76 = `asimd`+`asimddp` only, A72 additionally lacks `asimddp` | `scripts/isa_probe.c`, DESIGN §4 |
| 2026-07-15T17:32Z | A1 | gx10 | `036bf4f` | Python suite: 28 passed. C++ configure/build/ctest: 1/1 passed. Same test binary under `qemu -cpu cortex-a76` reports the Pi 5 feature set; `-mcpu=native` guard verified to fire | this file |
| 2026-07-15T17:50Z | A2 | gx10 | `fce3505` | Measured ORT 1.27.1 aarch64 needs only GLIBC_2.27 / GLIBCXX_3.4.21 — far below bookworm's 2.36 — so one identical ORT binary serves gx10 and Pi. SHA-256 cross-checked against GitHub's published digest | `configs/env/pins.env` |
| 2026-07-15T17:58Z | A2 | gx10 | `fce3505` | Built `wildlife-trigger-target:bookworm` from `debian:bookworm-slim@sha256:7b140f37…`: glibc 2.36, gcc 12.2, cmake 3.25.1, OpenCV 4.6.0, qemu 7.2.22, ORT 1.27.1 | `docker/Dockerfile.target` |
| 2026-07-15T18:05Z | A2 | gx10 | `HEAD` | Target env verified end to end: our binary needs GLIBC ≤ 2.34 and ORT ≤ 2.27 against a 2.36 target; ORT links, reports 1.27.1 and constructs a session **under `qemu -cpu cortex-a76`** | `scripts/verify_target_env.sh` |
