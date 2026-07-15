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
