"""D4 machinery: allocation, architecture reload, selection, and the row.

The allocation tests run against a synthetic sensitivity report whose curves
are chosen so the correct greedy behaviour is arithmetically obvious; the
reload tests run against the real frozen architecture because shape surgery
is exactly where a synthetic stand-in would prove nothing.
"""

from __future__ import annotations

import json

import pytest
import torch

from wildlife_trigger.models.mobilenet import build_mobilenet_v2
from wildlife_trigger.optimize import prune as P
from wildlife_trigger.optimize import select_m3 as S
from wildlife_trigger.optimize.m3_finetune import M3Config, load_config

NUM_CLASSES = 16


def synthetic_report() -> dict:
    """Two groups: 'robust' (zero damage) and 'fragile' (steep damage).

    Each contributes 10% MAC reduction at the 0.5 cap. Correct greedy
    behaviour: eat the robust group entirely before touching the fragile one.
    """

    def curve(width, damage_at_cap, mac_at_cap):
        points = []
        for ratio in (0.125, 0.25, 0.375, 0.5):
            removed = int(width * ratio)
            points.append(
                {
                    "requested_ratio": ratio,
                    "width_before": width,
                    "width_after": width - removed,
                    "realized_ratio": ratio,
                    "delta_primary": damage_at_cap * (ratio / 0.5),
                    "mac_reduction": mac_at_cap * (ratio / 0.5),
                }
            )
        return points

    return {
        "baseline": {"profile": {"macs": 1_000_000, "params": 1}},
        "groups": [
            {"block": 2, "conv": "features.2.conv.0.0", "width": 96,
             "curve": curve(96, 0.0, 0.10)},
            {"block": 3, "conv": "features.3.conv.0.0", "width": 144,
             "curve": curve(144, 0.4, 0.10)},
        ],
    }


class TestAllocateGreedy:
    def test_prefers_the_robust_group(self):
        # 5% target: half the robust group's budget, fragile untouched.
        allocation = P.allocate_greedy(synthetic_report(), 0.05)
        assert allocation["removals"].get("features.3") is None
        assert allocation["removals"]["features.2"] > 0
        assert allocation["predicted_reduction"] >= 0.05

    def test_spills_into_the_fragile_group_only_after_the_cap(self):
        # 15% target > robust group's 10% cap contribution.
        allocation = P.allocate_greedy(synthetic_report(), 0.15)
        assert allocation["removals"]["features.2"] == 48  # the whole cap
        assert allocation["removals"]["features.3"] > 0
        assert allocation["predicted_reduction"] >= 0.15

    def test_envelope_exhaustion_is_recorded_not_hidden(self):
        # 25% target > the 20% envelope: everything at cap, flag raised.
        allocation = P.allocate_greedy(synthetic_report(), 0.25)
        assert allocation["envelope_exhausted"] is True
        assert allocation["removals"]["features.2"] == 48
        assert allocation["removals"]["features.3"] == 72
        assert allocation["predicted_reduction"] == pytest.approx(0.20)

    def test_deterministic(self):
        first = P.allocate_greedy(synthetic_report(), 0.15)
        second = P.allocate_greedy(synthetic_report(), 0.15)
        assert first == second

    def test_widths_stay_aligned(self):
        allocation = P.allocate_greedy(synthetic_report(), 0.15)
        for width in allocation["widths"].values():
            assert width >= 8 and width % 8 == 0

    def test_refuses_a_report_not_measured_to_the_cap(self):
        report = synthetic_report()
        for group in report["groups"]:
            group["curve"] = group["curve"][:-1]  # drop the 0.5 point
        with pytest.raises(ValueError, match="cannot exceed measurement"):
            P.allocate_greedy(report, 0.05)

    def test_envelope_helper_matches_the_curves(self):
        assert P.allocation_envelope(synthetic_report()) == pytest.approx(0.20)


class TestDamageInterpolation:
    def test_interpolates_between_measured_points(self):
        points = [(0, 0.0), (10, 0.1), (20, 0.4)]
        assert P._damage_at(points, 5) == pytest.approx(0.05)
        assert P._damage_at(points, 15) == pytest.approx(0.25)
        assert P._damage_at(points, 20) == pytest.approx(0.4)

    def test_refuses_extrapolation(self):
        with pytest.raises(ValueError, match="beyond the measured curve"):
            P._damage_at([(0, 0.0), (10, 0.1)], 11)


class TestApplyWidths:
    def test_reproduces_a_pruned_architecture_exactly(self):
        torch.manual_seed(3)
        pruned = build_mobilenet_v2(num_classes=NUM_CLASSES, pretrained=False)
        report = P.prune_expansion(pruned, {3: 0.25, 17: 0.5}, export_check=False)
        widths = report["invariants"]["expansion_widths"]

        fresh = build_mobilenet_v2(num_classes=NUM_CLASSES, pretrained=False)
        P.apply_widths(fresh, widths)
        fresh.load_state_dict(pruned.state_dict())

        example = torch.zeros(1, 3, 192, 256)
        pruned.eval(), fresh.eval()
        with torch.inference_mode():
            assert torch.equal(pruned(example), fresh(example))

    def test_refuses_unaligned_or_grown_widths(self):
        model = build_mobilenet_v2(num_classes=NUM_CLASSES, pretrained=False)
        with pytest.raises(ValueError, match="multiple of 8"):
            P.apply_widths(model, {"features.3": 100})
        model = build_mobilenet_v2(num_classes=NUM_CLASSES, pretrained=False)
        with pytest.raises(ValueError, match="multiple of 8"):
            P.apply_widths(model, {"features.3": 200})  # > 144: cannot grow

    def test_refuses_non_expansion_blocks(self):
        model = build_mobilenet_v2(num_classes=NUM_CLASSES, pretrained=False)
        with pytest.raises(ValueError, match="not an expansion block"):
            P.apply_widths(model, {"features.1": 16})


def selection_row(label, primary, macs, reduction, **overrides):
    row = {
        "label": label,
        "candidate_id": f"d4_m3_{label}",
        "primary": primary,
        "cis_f2": primary * 1.7,
        "trans_f2": primary * 0.3,
        "macs_ladder": macs,
        "realized_mac_reduction_tp": reduction,
        "params": 2_000_000,
        "bytes": 8_000_000,
        "pre_finetune_primary": primary / 2,
        "threshold": 0.5,
        "calibration_status": "recall_floor_infeasible",
        "policy_path": f"artifacts/policies/bobcat_m3_prune_{label}_v1.json",
    }
    row.update(overrides)
    return row


class TestSelectM3:
    REFERENCE = 0.3667

    def test_selects_largest_reduction_above_the_line(self):
        rows = [
            selection_row("c15", 0.370, 250_000_000, 0.15),
            selection_row("c30", 0.360, 210_000_000, 0.30),  # above 0.3484
            selection_row("c45", 0.300, 180_000_000, 0.43),  # below the line
        ]
        verdict = S.select(rows, self.REFERENCE)
        assert verdict["selected"] == "c30"
        assert verdict["recovery_line_met"] is True

    def test_dominated_candidates_cannot_win(self):
        rows = [
            selection_row("c15", 0.370, 250_000_000, 0.15),
            # c30 dominates c15's twin: same primary, fewer MACs
            selection_row("c30", 0.370, 210_000_000, 0.30),
            selection_row("c45", 0.200, 180_000_000, 0.43),
        ]
        verdict = S.select(rows, self.REFERENCE)
        assert "c15" in verdict["dominated"]
        assert verdict["selected"] == "c30"

    def test_line_failure_falls_back_to_highest_primary(self):
        rows = [
            selection_row("c15", 0.320, 250_000_000, 0.15),
            selection_row("c30", 0.300, 210_000_000, 0.30),
            selection_row("c45", 0.250, 180_000_000, 0.43),
        ]
        verdict = S.select(rows, self.REFERENCE)
        assert verdict["recovery_line_met"] is False
        assert verdict["selected"] == "c15"
        assert "no non-dominated candidate reached" in verdict["rule"]


class TestM3Config:
    def test_refuses_unknown_keys(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("source_run_id: x\nnot_a_field: 1\n")
        with pytest.raises(ValueError, match="unknown keys"):
            load_config(path)

    def test_refuses_missing_pins(self, tmp_path):
        path = tmp_path / "unpinned.yaml"
        path.write_text(
            "source_run_id: x\nsource_checkpoint: y\n"
            "source_checkpoint_sha256: z\n"  # sensitivity sha missing
        )
        with pytest.raises(ValueError, match="sensitivity_report_sha256"):
            load_config(path)

    def test_registered_defaults(self):
        config = M3Config()
        assert config.lr == 3e-4
        assert config.max_epochs == 15
        assert config.early_stopping_patience == 4
        assert config.targets == {"c15": 0.15, "c30": 0.30, "c45": 0.45}


class TestLadderMacs:
    def test_macs_of_model_matches_the_ladder_reference(self):
        from wildlife_trigger.validate.input_cost import macs_at, macs_of_model

        model = build_mobilenet_v2(num_classes=NUM_CLASSES, pretrained=False)
        assert macs_of_model(model, 256, 192) == 293_402_624
        assert macs_at(256, 192, NUM_CLASSES) == 293_402_624

    def test_pruning_lowers_the_ladder_counter_too(self):
        from wildlife_trigger.validate.input_cost import macs_of_model

        model = build_mobilenet_v2(num_classes=NUM_CLASSES, pretrained=False)
        P.prune_expansion(model, {17: 0.5}, export_check=False)
        assert macs_of_model(model, 256, 192) < 293_402_624


class TestP3Fp32Check:
    def test_passes_on_matching_shapes_and_fails_on_tampered(self, tmp_path):
        import torch.nn as nn

        from wildlife_trigger.models.export import export_onnx
        from wildlife_trigger.validate.p3_quantized import check_graph_fp32_pruned

        torch.manual_seed(0)
        model = build_mobilenet_v2(num_classes=NUM_CLASSES, pretrained=False)
        P.prune_expansion(model, {3: 0.5}, export_check=False)
        model.eval()
        path = tmp_path / "m3.onnx"
        export_onnx(model, path, torch.zeros(1, 3, 192, 256))

        shapes = sorted(
            list(m.weight.shape) for m in model.modules() if isinstance(m, nn.Conv2d)
        )
        candidate = {"pruning": {"exported_conv_shapes": shapes}}
        result = check_graph_fp32_pruned(path, tmp_path / "cov", "t", candidate)
        assert result["passed"] is True
        assert result["integer_execution"] is False

        tampered = {"pruning": {"exported_conv_shapes": shapes[:-1]}}
        result = check_graph_fp32_pruned(path, tmp_path / "cov2", "t", tampered)
        assert result["passed"] is False
        assert "conv-shape multiset" in result["failures"][0]


class TestComparisonPrunedRow:
    @pytest.fixture(scope="class")
    def world(self, tmp_path_factory):
        """A pruned candidate directory + policy + parity that agree."""
        from wildlife_trigger.runs import sha256_file

        root = tmp_path_factory.mktemp("m3row")
        run_dir = root / "runs" / "d4_m3_c30_x"
        run_dir.mkdir(parents=True)
        candidate_dir = root / "c30"
        candidate_dir.mkdir()

        torch.manual_seed(0)
        model = build_mobilenet_v2(num_classes=NUM_CLASSES, pretrained=False)
        report = P.prune_expansion(model, {17: 0.5}, export_check=False)
        widths = report["invariants"]["expansion_widths"]
        class_names = [f"c{i}" for i in range(NUM_CLASSES)]
        torch.save(
            {"model": model.state_dict(), "epoch": 3, "widths": widths,
             "class_names": class_names},
            run_dir / "best.pt",
        )

        artifact = candidate_dir / "model.onnx"
        artifact.write_bytes(b"pruned-graph-stand-in; only its hash is compared")
        onnx_sha = sha256_file(artifact)

        candidate = {
            "candidate_id": "d4_m3_c30_x",
            "kind": "pruned_fp32",
            "method": "c30",
            "seed": 42,
            "source_run_id": "c2_m0",
            "source_checkpoint": {"sha256": "s"},
            "finetune_run_id": "d4_m3_c30_x",
            "finetune_run_dir": str(run_dir),
            "best_epoch": 3,
            "best_checkpoint_sha256": sha256_file(run_dir / "best.pt"),
            "input": {"width": 256, "height": 192},
            "pruning": {
                "target_fraction": 0.30,
                "realized_mac_reduction_tp": 0.28,
                "param_reduction": 0.05,
                "widths": widths,
                "pre_finetune_primary": 0.1,
                "macs_ladder_convention": 0,  # measured at row time, not copied
            },
            "model": {"sha256": onnx_sha},
        }
        (candidate_dir / "candidate.json").write_text(json.dumps(candidate))

        domain = {
            "target": {
                "frame_f2": 0.55, "average_precision": 0.5, "frame_recall": 0.5,
                "frame_precision": 0.5, "sequence_balanced_recall": 0.5,
                "false_fire_rate": 0.03, "fire_rate": 0.05,
                "event_capture_rate": 0.5, "threshold": 0.5,
            }
        }
        evaluation = {
            "label": "d4_m3_c30_x",
            "model": {"path": str(artifact), "sha256": onnx_sha},
            "class_names": class_names,
            "regime": {"input": "256x192", "intra_op_threads": 1},
            "selection_score": {"primary": 0.35},
            "domains": {"cis_val_clean": domain, "trans_val": domain},
        }
        (candidate_dir / "evaluation.json").write_text(json.dumps(evaluation))

        parity = root / "p3.json"
        parity.write_text(json.dumps(
            {"onnx": {"sha256": onnx_sha}, "verdict": {"passed": True}}
        ))

        policy = {
            "policy_id": "bobcat_m3_prune_c30_v1",
            "model_sha256": onnx_sha,
            "model": {"artifact": str(artifact), "parity": str(parity)},
            "targets": [{"class": "bobcat", "threshold": 0.62}],
            "calibration": {
                "run_id": "d4_m3_c30_x",
                "status": "recall_floor_infeasible",
                "primary_rule_met": False,
                "per_domain": {},
            },
        }
        policy_path = root / "policy.json"
        policy_path.write_text(json.dumps(policy))
        return {
            "root": root, "candidate_dir": candidate_dir,
            "policy_path": policy_path, "run_dir": run_dir, "onnx_sha": onnx_sha,
        }

    def test_row_measures_its_own_params_and_macs(self, world):
        from wildlife_trigger import comparison as C

        candidate, evaluation, policy = C.load_candidate_row_inputs(
            world["candidate_dir"], world["policy_path"]
        )
        params, macs = C.pruned_params_and_macs(candidate)
        assert params < 2_244_368  # fewer than M0
        assert macs < 293_402_624

        artifact, onnx_sha = C.verify_artifact(policy)
        row = C.build_pruned_row(
            "M3", "pruned_fp32", candidate, evaluation, policy,
            world["policy_path"], artifact, onnx_sha,
            world["root"] / "p3.json", params, macs,
        )
        assert row["params"] == params
        assert row["macs"] == macs
        assert row["pruning"]["method"] == "c30"
        assert row["validation_at_0p5"]["cis_f2"] == 0.55

    def test_refuses_a_checkpoint_that_moved(self, world):
        from wildlife_trigger import comparison as C

        candidate, _, _ = C.load_candidate_row_inputs(
            world["candidate_dir"], world["policy_path"]
        )
        tampered = dict(candidate)
        tampered["best_checkpoint_sha256"] = "0" * 64
        with pytest.raises(RuntimeError, match="unknown file"):
            C.pruned_params_and_macs(tampered)
