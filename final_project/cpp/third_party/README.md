# Vendored third-party sources

DESIGN §11 requires the release bundle to build with **no network fetch and no
system JSON/YAML development package**. Anything the runtime needs is vendored
here as source, pinned, and hashed.

Each vendored dependency must carry:

- the exact upstream version and release URL;
- its licence file, unmodified;
- its SHA-256, recorded in the dependency manifest;
- a note of why it is vendored rather than found via `find_package`.

Nothing is vendored yet. Expected first entry is `nlohmann/json.hpp`, the
single-header parser for the runtime policy and threshold-catalog JSON, added in
E1 when the policy loader is implemented. Vendoring it now, before anything reads
JSON, would commit ~900 KB in support of no caller.
