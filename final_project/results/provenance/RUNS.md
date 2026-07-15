# Run log

Append-only. One entry per executed run, newest last. `Handoff/` is gitignored and
therefore local-only; this file is the public record.

Every entry names the machine-readable evidence it produced, so a claim in the
report can be traced back to a file without reading prose.

| UTC | Phase/task | Host | Commit | What ran | Evidence |
|---|---|---|---|---|---|
| 2026-07-15T16:52Z | A0 | gx10 | `45117a9` | Project-start environment capture | `results/provenance/project_start.json` |
