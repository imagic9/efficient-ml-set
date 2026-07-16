"""The generic policy schema, on the side that writes it (PLAN C3).

The C++ loader is the contract's authority; `wildlife_trigger.policy` is the same
contract at generation time. The tests that matter most here are the coupling ones:
every policy the Python side builds must load in C++, and every policy the C++ test
suite rejects must already be refused in Python — otherwise the two sides drift and
the drift is only discovered on the Pi.
"""

from __future__ import annotations

import math

import pytest

from wildlife_trigger import policy as P
from wildlife_trigger.validate.policy_rejections import REJECTION_CASES

# The frozen B1 training order (results/training/c2/.../hashes.json), which is
# deliberately NOT alphabetical: the smoke map's order was a placeholder, and these
# tests must prove the real map preserves the run's order rather than "fixing" it.
TRAINING_ORDER = [
    "opossum", "raccoon", "squirrel", "bobcat", "skunk", "dog", "coyote", "rabbit",
    "bird", "cat", "badger", "empty", "car", "deer", "fox", "rodent",
]

A_HASH = "a" * 64
B_HASH = "b" * 64


def class_map() -> dict:
    return P.build_class_map(TRAINING_ORDER)


def valid_policy(**overrides) -> dict:
    built = P.build_policy(
        policy_id="bobcat_v1",
        targets=[{"class": "bobcat", "threshold": 0.5381}],
        class_map=class_map(),
        class_map_sha256=A_HASH,
        model_sha256=B_HASH,
    )
    built.update(overrides)
    return built


class TestClassMap:
    def test_training_order_is_preserved_not_sorted(self) -> None:
        """A threshold binds to an index. 'Tidying' the order re-aims every one."""
        result = class_map()
        assert result["classes"] == TRAINING_ORDER
        assert result["classes"].index("bobcat") == 3

    def test_animal_and_non_selectable_lists_follow_the_same_order(self) -> None:
        """One order in the artifact; nothing for a reader to reconcile."""
        result = class_map()
        assert result["animal_classes"] == [
            c for c in TRAINING_ORDER if c not in ("car", "empty")
        ]
        assert result["non_selectable_classes"] == ["empty", "car"]

    def test_unknown_or_missing_classes_are_refused(self) -> None:
        with pytest.raises(ValueError, match="unicorn"):
            P.build_class_map(TRAINING_ORDER[:-1] + ["unicorn"])
        with pytest.raises(ValueError, match="missing"):
            P.build_class_map(TRAINING_ORDER[:-1])

    def test_duplicates_are_refused(self) -> None:
        """C++ `index_of` returns the first match, so the second copy of a name is a
        class whose threshold silently never fires."""
        with pytest.raises(ValueError, match="duplicates"):
            P.build_class_map(TRAINING_ORDER[:-1] + ["bobcat"])


class TestCanonicalBytes:
    def test_same_content_same_hash(self, tmp_path) -> None:
        """The class map's hash is bound into the policy: re-writing unchanged
        content must not invalidate a policy that did not change."""
        first = P.write_canonical_json(tmp_path / "a.json", class_map())
        second = P.write_canonical_json(tmp_path / "b.json", class_map())
        assert first == second
        assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()


class TestBuildPolicy:
    def test_round_trip(self) -> None:
        """Everything build_policy produces must pass its own validator."""
        built = valid_policy()
        P.validate_policy(
            built, class_map(), model_sha256=B_HASH, class_map_sha256=A_HASH
        )
        assert built["schema_version"] == P.SCHEMA_VERSION
        assert built["mode"] == "any"

    def test_no_threshold_catalog_classes_are_refused_at_generation(self) -> None:
        """DESIGN §4: badger/deer/fox have no defensible operating point. The
        loader rejects them at startup; generation must not get that far."""
        for name in P.NO_THRESHOLD_CLASSES:
            with pytest.raises(ValueError, match=name):
                P.build_policy(
                    policy_id="bad",
                    targets=[{"class": name, "threshold": 0.5}],
                    class_map=class_map(),
                    class_map_sha256=A_HASH,
                    model_sha256=B_HASH,
                )

    def test_metadata_may_not_override_schema_keys(self) -> None:
        """A metadata block that replaced `targets` would be the module arguing
        with itself; the extra context must stay extra."""
        with pytest.raises(ValueError, match="targets"):
            P.build_policy(
                policy_id="bad",
                targets=[{"class": "bobcat", "threshold": 0.5}],
                class_map=class_map(),
                class_map_sha256=A_HASH,
                model_sha256=B_HASH,
                metadata={"targets": []},
            )

    def test_metadata_rides_along(self) -> None:
        built = P.build_policy(
            policy_id="bobcat_v1",
            targets=[{"class": "bobcat", "threshold": 0.5}],
            class_map=class_map(),
            class_map_sha256=A_HASH,
            model_sha256=B_HASH,
            metadata={"calibration": {"status": "recall_floor_infeasible"}},
        )
        assert built["calibration"]["status"] == "recall_floor_infeasible"


class TestValidatePolicy:
    @pytest.mark.parametrize(
        "name", [n for n in REJECTION_CASES if n != "malformed_json"]
    )
    def test_every_cpp_rejection_case_is_refused_in_python(self, name: str) -> None:
        """The coupling test: the list of policies the deployed CLI must refuse
        (`validate.policy_rejections`, driven against the real binary by A4) is
        exactly the list the Python validator must refuse. One list, two enforcers.

        `malformed_json` is excluded because it is not a dict — it exercises the
        JSON parser, which Python delegates to `json.loads` the same way C++
        delegates to nlohmann.
        """
        case = dict(REJECTION_CASES[name])
        # The actual hashes: a case that carries no binding skips the comparison
        # (the loader's semantics), and the two wrong-hash cases collide with these.
        with pytest.raises(ValueError):
            P.validate_policy(
                case, class_map(), model_sha256=A_HASH, class_map_sha256=A_HASH
            )

    def test_hash_mismatch_is_refused_when_actuals_are_known(self) -> None:
        """The binding that stops a policy calibrated for one model being applied
        to another, where the same index may denote a different animal."""
        with pytest.raises(ValueError, match="model_sha256 mismatch"):
            P.validate_policy(
                valid_policy(), class_map(), model_sha256=A_HASH
            )
        with pytest.raises(ValueError, match="class_map_sha256 mismatch"):
            P.validate_policy(
                valid_policy(), class_map(), class_map_sha256=B_HASH
            )

    def test_empty_binding_is_format_checked_only(self) -> None:
        """At generation the pairing may not exist yet; an empty actual skips the
        comparison, exactly as the C++ loader treats an empty expected hash."""
        P.validate_policy(valid_policy(), class_map())

    def test_non_hex_hash_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not a SHA-256"):
            P.validate_policy(valid_policy(model_sha256="not-a-hash"), class_map())

    def test_nan_threshold_is_refused(self) -> None:
        """NaN compares false against everything: a trigger that never fires and
        never errors. Same guard, same reason, as the C++ loader's `>= && <=`."""
        broken = valid_policy(targets=[{"class": "bobcat", "threshold": math.nan}])
        with pytest.raises(ValueError, match="outside"):
            P.validate_policy(broken, class_map())

    def test_boolean_threshold_is_refused(self) -> None:
        """bool is an int subclass; `True` must not pass as threshold 1.0."""
        broken = valid_policy(targets=[{"class": "bobcat", "threshold": True}])
        with pytest.raises(ValueError, match="numeric"):
            P.validate_policy(broken, class_map())

    def test_boundary_thresholds_are_accepted(self) -> None:
        """0.0 and 1.0 are legal operating points, and the calibration searches
        observed scores — the boundary is ordinary, not an edge case."""
        for threshold in (0.0, 1.0):
            P.validate_policy(
                valid_policy(targets=[{"class": "bobcat", "threshold": threshold}]),
                class_map(),
            )


class TestSmokeStillHonoursTheContract:
    """models.smoke now delegates to this module; its artifacts must not change."""

    def test_smoke_policy_passes_the_generic_validator(self) -> None:
        smoke = pytest.importorskip("wildlife_trigger.models.smoke")
        built = smoke.build_policy(A_HASH, B_HASH, ["bobcat", "coyote"])
        P.validate_policy(built, smoke.build_class_map())
        assert built["policy_id"] == "smoke_bobcat_coyote_v0"
        assert "provisional" in built

    def test_smoke_still_refuses_no_threshold_classes(self) -> None:
        smoke = pytest.importorskip("wildlife_trigger.models.smoke")
        with pytest.raises(ValueError, match="badger"):
            smoke.build_policy(A_HASH, B_HASH, ["badger"])
