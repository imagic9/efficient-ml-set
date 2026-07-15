# Data

**No images or archives are committed here.** Only manifests, hashes and audit
output — the things small enough to version and important enough to pin.

## What gets downloaded

Roughly 8.1 GB total; see DESIGN §5.1. The 105 GB `cct_images.tar.gz` is never
downloaded.

| Source | Size | Purpose |
|---|---:|---|
| `eccv_18_all_images_sm.tar.gz` | 6 GB | CCT-20 frames, **max 1024 px per side** |
| `eccv_18_annotations.tar.gz` | 3 MB | official split metadata |
| `caltech_camera_traps.json.zip` | 9 MB | full-CCT metadata, empty-supplement selection only |
| per-image CCT paths | ~2.1 GB | the 5,000 empty supplement frames, **original resolution** |

The two image sources arrive at **different resolutions**, and that is the trap
DESIGN §5.2 step 7 exists to close: the supplement is downsized to max 1024 px
before training, or resolution becomes a feature that predicts `empty` perfectly
in training and is absent at evaluation.

## Contents

- `manifests/` — frozen JSONL manifests: the five official splits, the derived
  `cis_val_clean.jsonl`, and `cct_empty_train_v1.jsonl`. Committed.
- Raw images live outside version control under ignored paths (`data/raw/`,
  `data/images/`).

## Licence and citation

CCT is released under the Community Data License Agreement (Permissive). Cite
Beery, Van Horn and Perona, *Recognition in Terra Incognita*, ECCV 2018. Source
URLs and SHA-256 hashes are recorded by B0 and never taken on trust from this
file.
