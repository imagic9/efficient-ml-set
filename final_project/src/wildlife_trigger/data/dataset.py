#!/usr/bin/env python3
"""B3 — dataset readers, augmentation, and class weighting.

Three decisions here are DESIGN's, not conveniences, and each is easy to get wrong in a
way that produces a plausible number:

**The seven multi-class train images are excluded from cross-entropy, not deleted.**
A frame holding a bobcat *and* a coyote has no single correct label, so it cannot train a
softmax — but it is still a true bobcat frame, and target-presence evaluation must count
it. So `labels` (the complete set) survives on every record and `primary_label` is None
where the frame is multi-class. Deleting them would quietly shrink the positive count
that recall is measured against.

**Augmentation contains no crop and no resize.** DESIGN §5.5: "no crop that can exclude
the labelled animal". A `RandomResizedCrop` would be the obvious default and would
regularly delete the animal from a camera-trap frame where it occupies a few percent of
the pixels — teaching the model that those frames are empty. It is also the condition the
offline cache rests on.

**Class weights come from the frozen training manifest**, by effective number of samples,
and never from test frequencies (DESIGN §6.2).

Validation and test are fully deterministic: no augmentation, no shuffling of the
transform, nothing that makes the same image score differently on two runs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from wildlife_trigger.data.cache import open_cache
from wildlife_trigger.data.preprocess import PreprocessConfig, decode, letterbox_bgr, normalise

# DESIGN §6.2. beta -> 1 weights by inverse frequency; 0.999 is the standard
# effective-number setting and is capped below so a class with two examples cannot
# dominate the gradient.
EFFECTIVE_NUMBER_BETA = 0.999
MAX_CLASS_WEIGHT_RATIO = 20.0


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def load_class_names(classes_config: Path) -> list[str]:
    import yaml

    document = yaml.safe_load(classes_config.read_text())
    entries = sorted(document["classes"], key=lambda c: c["index"])
    if [c["index"] for c in entries] != list(range(len(entries))):
        raise RuntimeError(f"{classes_config}: class indices are not 0..N-1")
    return [c["name"] for c in entries]


def class_weights(records: list[dict], class_names: list[str]) -> torch.Tensor:
    """Effective-number-of-samples weights from the training manifest (DESIGN §6.2).

    Counts `primary_label`, because that is what cross-entropy actually sees: weighting
    by the full label set would inflate classes that co-occur.
    """
    counts = np.zeros(len(class_names), dtype=np.float64)
    index_of = {name: i for i, name in enumerate(class_names)}
    for record in records:
        if record["primary_label"] is not None:
            counts[index_of[record["primary_label"]]] += 1

    # A class absent from train gets the weight of a single-sample class rather than
    # inf. CCT-20's train split has no `empty` at all before the supplement, and 1/0
    # would poison the whole vector.
    effective = 1.0 - np.power(EFFECTIVE_NUMBER_BETA, np.maximum(counts, 1.0))
    weights = (1.0 - EFFECTIVE_NUMBER_BETA) / effective

    weights = weights / weights.min()
    weights = np.minimum(weights, MAX_CLASS_WEIGHT_RATIO)
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


class Augmentation:
    """Training-only photometric augmentation (DESIGN §5.5).

    Operates on the uint8 HWC letterbox, before normalisation, so it composes with the
    cache. Every transform is photometric or a flip: nothing here can move the animal
    out of frame.
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)

        if self.rng.random() < 0.5:
            image = image[:, ::-1, :]

        # Mild brightness/contrast/saturation jitter.
        image *= self.rng.uniform(0.85, 1.15)  # brightness
        mean = image.mean()
        image = (image - mean) * self.rng.uniform(0.85, 1.15) + mean  # contrast
        grey = image.mean(axis=2, keepdims=True)
        image = grey + (image - grey) * self.rng.uniform(0.8, 1.2)  # saturation

        # Random grayscale, p=0.15: camera traps switch to IR at night, and this is what
        # that looks like to the network.
        if self.rng.random() < 0.15:
            image = np.repeat(image.mean(axis=2, keepdims=True), 3, axis=2)

        # Mild Gaussian blur, p=0.10. A 3x3 separable kernel; cv2 would be faster but
        # this keeps the augmentation free of the OpenCV version question entirely.
        if self.rng.random() < 0.10:
            kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
            for axis in (0, 1):
                image = np.apply_along_axis(
                    lambda row: np.convolve(row, kernel, mode="same"), axis, image
                )

        return np.clip(image, 0, 255).astype(np.uint8)


class WildlifeDataset(Dataset):
    """Reads the cache when available, decodes on demand when not.

    `images_dir` is required even with a cache: the cache is a derived artifact and the
    manifest's file_name is the source of truth, so a cache miss must be able to fall
    back rather than fail.
    """

    def __init__(
        self,
        manifest: Path,
        class_names: list[str],
        config: PreprocessConfig,
        images_dir: Path,
        cache_root: Path | None = None,
        train: bool = False,
        seed: int = 0,
        image_root_overrides: dict[str, Path] | None = None,
    ):
        self.records = load_manifest(manifest)
        self.class_names = class_names
        self.index_of = {name: i for i, name in enumerate(class_names)}
        self.config = config
        self.images_dir = Path(images_dir)
        self.train = train
        self.augment = Augmentation(seed) if train else None
        self.overrides = image_root_overrides or {}

        self.manifest = Path(manifest)
        self.pixels = None
        # Always defined, so "this dataset decoded its own JPEGs" is an answerable
        # question rather than an AttributeError. A run has to record which cache it
        # read (DESIGN §9.2), including when the answer is "none".
        self.cache_meta = None
        if cache_root is not None:
            try:
                self.pixels, self.cache_meta = open_cache(manifest, cache_root, config)
            except FileNotFoundError:
                self.pixels = None

        # Cross-entropy needs a single target. DESIGN B3: exclude the multi-class frames
        # from CE while retaining their full label sets for target-presence evaluation.
        self.ce_indices = [
            i for i, r in enumerate(self.records) if r["primary_label"] is not None
        ]

        # Labels this head does not model. DESIGN §5.2's no-empty arm is a 15-output
        # model, and validation is full of `empty` frames — for that model an empty
        # frame is simply "no animal present", which is exactly the negative the
        # false-fire rate is measured on. So unmodelled labels are tolerated rather
        # than fatal.
        #
        # Recorded rather than swallowed: a typo'd class name would otherwise vanish
        # into the same code path and quietly drop real positives. A run summary that
        # says `unmodelled_labels: ['empty']` is intended; anything else is a bug.
        self.unmodelled_labels = sorted(
            {label for r in self.records for label in r["labels"]} - set(class_names)
        )

    def __len__(self) -> int:
        return len(self.records)

    def image_path(self, record: dict) -> Path:
        root = self.overrides.get(record.get("source", "cct20"), self.images_dir)
        if record.get("relative_path"):
            return Path(root).parent / record["relative_path"]
        return Path(root) / record["file_name"]

    def letterbox(self, index: int) -> np.ndarray:
        if self.pixels is not None:
            return np.asarray(self.pixels[index])
        image, _ = letterbox_bgr(decode(self.image_path(self.records[index])), self.config)
        return image

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image = self.letterbox(index)

        if self.augment is not None:
            image = self.augment(image)

        tensor = torch.from_numpy(normalise(image, self.config))

        # -1 for a multi-class frame, and for a frame whose class this head does not
        # model: CrossEntropyLoss(ignore_index=-1) skips both, and no caller can mistake
        # -1 for class 0.
        target = self.index_of.get(record["primary_label"], -1)

        # A label the head does not model contributes no presence — which is the correct
        # semantics, not a workaround: for the 15-output arm an `empty` frame genuinely
        # has no animal present, so it is a negative for every target and the false-fire
        # rate counts it.
        present = torch.zeros(len(self.class_names), dtype=torch.float32)
        for label in record["labels"]:
            # `class_index`, not `index`: rebinding `index` here shadows the dataset
            # index this method was called with, and the returned "index" then becomes
            # the last label's class index. Evaluation uses it to look up seq_id, so
            # sequence-balanced recall would have been computed against the wrong
            # sequences — silently, for every frame.
            class_index = self.index_of.get(label)
            if class_index is not None:
                present[class_index] = 1.0

        return {
            "image": tensor,
            "target": target,
            "present": present,
            "index": index,
        }


class ConcatManifestDataset(Dataset):
    """Train split plus the empty supplement, as one dataset.

    The supplement lives in its own directory and its own manifest — it is not CCT-20 and
    is never allowed to look like it. Keeping them as separate datasets joined here means
    the `A-empty-5k` ablation is a constructor argument rather than a data edit.
    """

    def __init__(self, parts: list[Dataset]):
        self.parts = parts
        self.offsets = []
        total = 0
        for part in parts:
            self.offsets.append(total)
            total += len(part)
        self.total = total

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, index: int) -> dict:
        for part_index in range(len(self.parts) - 1, -1, -1):
            if index >= self.offsets[part_index]:
                return self.parts[part_index][index - self.offsets[part_index]]
        raise IndexError(index)
