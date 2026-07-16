#!/usr/bin/env python3
"""The versioned generic policy schema (DESIGN §4), on the side that generates it.

The C++ loader (`cpp/src/policy.cpp`) is the authority on what a policy *is*: it
rejects empty target lists, duplicate/unknown/non-animal classes, null thresholds,
values outside [0, 1], unsupported modes and schema versions, and hash mismatches.
This module is the same contract on the writing side, so that an invalid policy is
refused at generation — where the calibration numbers are still in scope and the
error message can say why — rather than at 6 a.m. on the Pi.

Two facts live here and nowhere else in Python:

- **The class catalog.** DESIGN §4 fixes 14 animal classes plus `car`/`empty`,
  which are model outputs but never selectable wildlife targets. `badger`, `deer`
  and `fox` carry no calibrated threshold at any confidence level (one validation
  positive between the three of them), so a policy naming them must be rejected,
  never generated with an invented number.
- **Canonical JSON bytes.** The class map's hash is bound into every policy, and
  the C++ loader hashes the *file's exact bytes*. `sort_keys` plus a fixed
  separator means the same content always serialises to the same bytes, so
  re-running a generator cannot invalidate a policy that did not change.

The integer order of the classes is NOT fixed here: it comes from the CCT-20
annotations, frozen by B1 and recorded in every run's `class_names`. This module
checks the *set*; the order is the caller's evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

SCHEMA_VERSION = 1

# DESIGN §4: 14 animals + car + empty = the 16-way single-label task.
ANIMAL_CLASSES = (
    "badger",
    "bird",
    "bobcat",
    "cat",
    "coyote",
    "deer",
    "dog",
    "fox",
    "opossum",
    "rabbit",
    "raccoon",
    "rodent",
    "skunk",
    "squirrel",
)
NON_ANIMAL_CLASSES = ("car", "empty")

# DESIGN §4 catalog: no defensible operating point exists for these (badger has one
# validation image; deer and fox have none). The loader rejects them; so do we.
NO_THRESHOLD_CLASSES = ("badger", "deer", "fox")

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def canonical_json(payload: dict) -> str:
    """The one serialisation whose bytes are stable enough to hash."""
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def write_canonical_json(path: Path, payload: dict) -> str:
    """Write canonical JSON and return the SHA-256 of the bytes on disk."""
    text = canonical_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def build_class_map(class_names: list[str]) -> dict:
    """The real class map, from a run's frozen `class_names` order.

    The names are DESIGN §4's; the order is B1's, carried by the training run. A
    mismatch in either direction is an error, not a warning: a policy threshold
    binds to an *index*, and an index against the wrong order fires on the wrong
    animal.
    """
    expected = set(ANIMAL_CLASSES) | set(NON_ANIMAL_CLASSES)
    got = set(class_names)
    if len(class_names) != len(got):
        duplicates = sorted({c for c in class_names if class_names.count(c) > 1})
        raise ValueError(f"class_names contains duplicates: {duplicates}")
    if got != expected:
        raise ValueError(
            f"class_names disagrees with DESIGN §4's 16 classes: "
            f"unknown={sorted(got - expected)}, missing={sorted(expected - got)}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        # Every list keeps the training order, so there is exactly one order in
        # the artifact and nothing for a reader to reconcile.
        "classes": list(class_names),
        "animal_classes": [c for c in class_names if c in ANIMAL_CLASSES],
        "non_selectable_classes": [c for c in class_names if c in NON_ANIMAL_CLASSES],
    }


def validate_policy(
    policy: dict,
    class_map: dict,
    *,
    model_sha256: str = "",
    class_map_sha256: str = "",
) -> None:
    """Raise ValueError on anything the C++ loader would refuse.

    Kept as one function over a parsed dict so tests can round-trip: everything
    `build_policy` produces must pass, and every REJECTION_CASE in
    `validate.policy_rejections` must fail.

    `model_sha256`/`class_map_sha256` are the *actual* hashes of the artifacts the
    policy is about to be paired with; when given, a policy bound to different
    ones is refused, exactly as the loader refuses it at startup. Left empty, the
    binding is only format-checked — generation time, where the pairing does not
    exist yet.
    """
    if policy.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version {policy.get('schema_version')!r} is not {SCHEMA_VERSION}"
        )
    if policy.get("mode") != "any":
        raise ValueError(
            f"mode {policy.get('mode')!r} is unsupported; DESIGN §4 defines 'any' "
            "as the only Core combination mode"
        )

    for key, actual in (
        ("model_sha256", model_sha256),
        ("class_map_sha256", class_map_sha256),
    ):
        value = policy.get(key, "")
        if value and not _SHA256_HEX.fullmatch(value):
            raise ValueError(f"{key} {value!r} is not a SHA-256 hex digest")
        if value and actual and value != actual:
            raise ValueError(
                f"{key} mismatch: policy was calibrated for {value[:16]}... but "
                f"the artifact at hand is {actual[:16]}.... Thresholds are "
                "model-specific (DESIGN §6.3), and class indices may not even agree"
            )

    targets = policy.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError(
            "target list is empty. A policy that selects nothing can never fire; "
            "that is a configuration mistake, not a valid 'off' switch"
        )

    classes = class_map["classes"]
    animals = set(class_map["animal_classes"])
    seen: set[str] = set()
    for target in targets:
        name = target.get("class")
        if name in seen:
            raise ValueError(f"target {name!r} is listed twice")
        seen.add(name)
        if name not in classes:
            raise ValueError(f"target {name!r} is not a class this model knows")
        if name not in animals:
            raise ValueError(
                f"target {name!r} is a model class but not a selectable wildlife "
                "target (DESIGN §4 excludes car and empty)"
            )
        if name in NO_THRESHOLD_CLASSES:
            raise ValueError(
                f"{name!r} has no calibrated threshold in the DESIGN §4 catalog "
                "(insufficient validation support). A policy naming it must be "
                "rejected, not generated."
            )
        threshold = target.get("threshold")
        # `isinstance` before the comparison: None and strings must be named, not
        # crash; bool is an int subclass and would slip through a bare int check.
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError(
                f"target {name!r} has no numeric threshold. Classes without a "
                "calibrated operating point must be rejected, never given an "
                "invented number"
            )
        # >= and <= so NaN fails, same as the C++ loader: every comparison against
        # a NaN threshold is false, i.e. a trigger that never fires and never errors.
        if math.isnan(threshold) or not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold for {name!r} is outside [0, 1]")


def build_policy(
    policy_id: str,
    targets: list[dict],
    class_map: dict,
    class_map_sha256: str,
    model_sha256: str,
    metadata: dict | None = None,
) -> dict:
    """A schema-1 policy that the C++ loader will accept, or a ValueError now.

    `metadata` merges extra top-level keys (calibration record, provisional
    notes); the loader ignores keys it does not read, and the extra context is
    what makes the artifact auditable. Reserved schema keys cannot be overridden
    through it — a metadata block that silently replaced `targets` would be this
    module arguing with itself.
    """
    policy = {
        "schema_version": SCHEMA_VERSION,
        "policy_id": policy_id,
        "model_sha256": model_sha256,
        "class_map_sha256": class_map_sha256,
        "mode": "any",
        "targets": [
            {"class": t["class"], "threshold": t["threshold"]} for t in targets
        ],
    }
    if metadata:
        collisions = set(metadata) & set(policy)
        if collisions:
            raise ValueError(f"metadata may not override schema keys: {sorted(collisions)}")
        policy.update(metadata)

    validate_policy(policy, class_map)
    return policy
