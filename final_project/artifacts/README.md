# Artifacts

Models, policies and model cards. Large binaries are **not** committed: they are
published through GitHub Releases with hashes and download links recorded here.

- `policies/` — `bobcat_v1.json` (the primary evaluated policy),
  `bobcat_coyote_v1.json` (multi-target example), `threshold_catalog.json`
  (14 status entries; 11 selectable targets, and `null` thresholds for `badger`,
  `deer` and `fox`, which lack the validation support to define an operating
  point). JSON rather than YAML so the C++ bundle vendors one pinned header-only
  parser instead of depending on a system library.
- `manifests/` — bundle and release manifests.
- `model_cards/` — one per deployed model: data, intended use, limitations,
  preprocessing, metrics, policy, licence, hashes.
- `checksums.sha256` — hashes for every published artifact.

Populated from Phase C onward. Download links and hashes are filled in at G5.
