"""D3 pruning machinery: the contract is the test surface.

DESIGN §8.3 fixes what may be pruned (16 expansion groups), what never moves
(stem, t=1 block, projections/residuals, final conv, classifier), the SIMD
alignment (round_to=8), and the post-mutation invariants. Each is asserted
against the real frozen architecture on CPU — pruning is cheap; it is the
64 validation evaluations that need gx10, and those are not tested here.

The refusal tests matter most: a plan computed for a subtly different
architecture, a group that reaches outside its block, or an invariant that
passes on a broken model would each corrupt every M3/M4 number downstream.
"""

from __future__ import annotations

import pytest
import torch

from wildlife_trigger.models.mobilenet import build_mobilenet_v2
from wildlife_trigger.optimize import prune as P

NUM_CLASSES = 16


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_mobilenet_v2(num_classes=NUM_CLASSES, pretrained=False)


@pytest.fixture()
def fresh_model():
    torch.manual_seed(0)
    return build_mobilenet_v2(num_classes=NUM_CLASSES, pretrained=False)


class TestClassify:
    def test_finds_exactly_the_16_expansion_convs(self, model):
        plan = P.classify(model)
        assert sorted(plan.expansion) == list(range(2, 18))
        for block, conv in plan.expansion.items():
            assert conv is model.features[block].conv[0][0]
            assert conv.kernel_size == (1, 1) and conv.groups == 1

    def test_ignored_covers_every_fixed_root(self, model):
        plan = P.classify(model)
        ignored = {id(m) for m in plan.ignored}
        assert id(model.features[0][0]) in ignored  # stem
        assert id(model.features[1].conv[0][0]) in ignored  # t=1 depthwise
        assert id(model.features[18][0]) in ignored  # final 1280
        assert id(model.classifier[-1]) in ignored  # task head
        for block in range(2, 18):  # every projection
            assert id(model.features[block].conv[2]) in ignored

    def test_depthwise_of_t6_blocks_must_not_be_ignored(self, model):
        """The load-bearing torch-pruning fact: an ignored out-channel member
        disqualifies the whole group, so ignoring depthwise convs would turn
        every expansion group into a silent no-op."""
        plan = P.classify(model)
        ignored = {id(m) for m in plan.ignored}
        for block in range(2, 18):
            assert id(model.features[block].conv[1][0]) not in ignored

    def test_refuses_a_different_architecture(self):
        torch.manual_seed(0)
        other = build_mobilenet_v2(
            num_classes=NUM_CLASSES, pretrained=False, width_mult=0.5
        )
        # width_mult=0.5 keeps the block layout but changes every width; the
        # frozen-width invariants are what must refuse it downstream. classify
        # itself refuses structural differences:
        other.features[5].conv = other.features[5].conv[:3]
        with pytest.raises(ValueError, match="not the frozen MobileNetV2"):
            P.classify(other)


class TestVerifyGroup:
    def test_group_is_exactly_the_design_coupling(self, model):
        plan = P.classify(model)
        evidence = P.verify_group(model, plan, 5)
        assert evidence["verified"] is True

    def test_detects_a_group_reaching_outside_the_contract(self, model):
        """If the solver's group for a root ever includes an unexpected
        weighted module, verify_group must raise rather than shrug."""
        plan = P.classify(model)
        # Lie about which block we are verifying: block 5's group against
        # block 6's modules is exactly "the wrong members".
        plan_wrong = P.PruningPlan(
            expansion={6: plan.expansion[5]}, ignored=plan.ignored
        )
        with pytest.raises(RuntimeError, match="does not match"):
            P.verify_group(model, plan_wrong, 6)


class TestPruneExpansion:
    def test_single_group_prunes_to_a_multiple_of_8(self, fresh_model):
        report = P.prune_expansion(
            fresh_model, {3: 0.25}, export_check=False
        )
        realized = report["realized"]["features.3"]
        assert realized["width_before"] == 144
        assert realized["width_after"] == 104
        assert realized["width_after"] % 8 == 0
        conv = fresh_model.features[3].conv
        assert (
            conv[0][0].out_channels
            == conv[1][0].in_channels
            == conv[1][0].out_channels
            == conv[1][0].groups
            == conv[2].in_channels
            == 104
        )

    def test_requested_and_realized_are_recorded_separately(self, fresh_model):
        report = P.prune_expansion(fresh_model, {3: 0.25}, export_check=False)
        realized = report["realized"]["features.3"]
        assert realized["requested_ratio"] == 0.25
        assert realized["realized_ratio"] != 0.25  # 144 -> 104 is 27.8%
        assert report["mac_reduction"] > 0

    def test_everything_fixed_stays_fixed(self, fresh_model):
        widths_before = {
            b: fresh_model.features[b].conv[0][0].out_channels for b in range(2, 18)
        }
        P.prune_expansion(fresh_model, {10: 0.5}, export_check=False)
        assert fresh_model.features[0][0].out_channels == 32
        assert fresh_model.features[18][0].out_channels == 1280
        assert fresh_model.classifier[-1].out_features == NUM_CLASSES
        for block in range(2, 18):
            conv = fresh_model.features[block].conv
            expected = P.PROJECT_OUT[block - 2]
            assert conv[2].out_channels == expected
            if block != 10:
                assert conv[0][0].out_channels == widths_before[block]

    def test_pruning_is_deterministic_given_the_weights(self):
        outputs = []
        for _ in range(2):
            torch.manual_seed(7)
            model = build_mobilenet_v2(num_classes=NUM_CLASSES, pretrained=False)
            P.prune_expansion(model, {4: 0.375}, export_check=False)
            outputs.append(
                {k: v.clone() for k, v in model.state_dict().items()}
            )
        assert outputs[0].keys() == outputs[1].keys()
        for key in outputs[0]:
            assert torch.equal(outputs[0][key], outputs[1][key]), key

    def test_refuses_non_expansion_blocks(self, fresh_model):
        with pytest.raises(ValueError, match="not expansion blocks"):
            P.prune_expansion(fresh_model, {1: 0.25}, export_check=False)
        with pytest.raises(ValueError, match="not expansion blocks"):
            P.prune_expansion(fresh_model, {18: 0.25}, export_check=False)

    def test_multi_group_request_prunes_each_group(self, fresh_model):
        report = P.prune_expansion(
            fresh_model, {2: 0.25, 17: 0.5}, export_check=False
        )
        assert report["realized"]["features.2"]["width_after"] == 72  # 96*0.75
        assert report["realized"]["features.17"]["width_after"] == 480  # 960/2
        assert report["mac_reduction"] > 0


class TestInvariants:
    def test_pass_on_the_frozen_model(self, model):
        result = P.check_invariants(model, num_classes=NUM_CLASSES)
        assert result["forward_backward"] == "ok"
        assert result["expansion_widths"]["features.17"] == 960

    def test_catch_a_broken_depthwise_coupling(self, fresh_model):
        # Sabotage: shrink the expansion conv without its dependents.
        conv = fresh_model.features[5].conv[0][0]
        conv.out_channels = 64
        conv.weight = torch.nn.Parameter(conv.weight[:64].clone())
        with pytest.raises(RuntimeError, match="coupling broken"):
            P.check_invariants(fresh_model, num_classes=NUM_CLASSES)

    def test_catch_an_unaligned_width(self, fresh_model):
        report = P.prune_expansion(fresh_model, {3: 0.25}, export_check=False)
        assert report  # sanity: the aligned prune passed
        # Now sabotage the whole coupled group to width 100 (not %8).
        conv = fresh_model.features[3].conv
        for module, attr in (
            (conv[0][0], "out_channels"),
            (conv[1][0], "in_channels"),
            (conv[1][0], "out_channels"),
            (conv[2], "in_channels"),
        ):
            setattr(module, attr, 100)
        conv[1][0].groups = 100
        with pytest.raises(RuntimeError, match="multiple of 8"):
            P.check_invariants(fresh_model, num_classes=NUM_CLASSES)

    def test_catch_a_moved_projection_width(self, fresh_model):
        conv = fresh_model.features[7].conv[2]
        conv.out_channels = 56
        conv.weight = torch.nn.Parameter(conv.weight[:56].clone())
        with pytest.raises(RuntimeError, match="residual widths"):
            P.check_invariants(fresh_model, num_classes=NUM_CLASSES)


class TestOnnxExport:
    def test_export_carries_the_pruned_widths(self, fresh_model):
        report = P.prune_expansion(fresh_model, {3: 0.5}, export_check=True)
        assert report["invariants"]["onnx_export"] == "ok"
        assert report["realized"]["features.3"]["width_after"] == 72

    def test_export_check_catches_stale_shapes(self, model):
        # Ask the checker to confirm a width the model does not have.
        with pytest.raises(RuntimeError, match="does not carry"):
            P.check_onnx_export(model, {"features.3": 104})


class TestProfile:
    def test_m0_profile_matches_the_probe_measurement(self, model):
        measured = P.profile(model)
        assert measured["params"] == 2_244_368  # the ladder's param count
        assert measured["macs"] == 312_467_472  # tp counter at 192x256

    def test_pruning_reduces_the_same_counter(self, fresh_model):
        before = P.profile(fresh_model)
        P.prune_expansion(fresh_model, {17: 0.5}, export_check=False)
        after = P.profile(fresh_model)
        assert after["macs"] < before["macs"]
        assert after["params"] < before["params"]


class TestEvaluateAtYardstick:
    def test_metrics_from_a_fabricated_loader(self):
        """The eval path is exercised with a synthetic two-class world so the
        metric wiring (bobcat column, seq_ids, primary) is proven without gx10."""

        class TinyDataset:
            class_names = ["empty", "bobcat"]
            records = [{"seq_id": f"s{i//2}"} for i in range(8)]

        class TinyLoader:
            dataset = TinyDataset()

            def __iter__(self):
                images = torch.zeros(8, 3, 192, 256)
                present = torch.tensor(
                    [[0, 1], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0]],
                    dtype=torch.float32,
                )
                yield {
                    "image": images,
                    "present": present,
                    "index": torch.arange(8),
                }

        class Oracle(torch.nn.Module):
            """Fires bobcat on the first half, empty on the second."""

            def forward(self, x):
                logits = torch.full((x.shape[0], 2), -10.0)
                logits[:4, 1] = 10.0
                logits[4:, 0] = 10.0
                return logits

        result = P.evaluate_at_yardstick(
            Oracle(), {"cis_val_clean": TinyLoader(), "trans_val": TinyLoader()},
            torch.device("cpu"),
        )
        assert result["per_domain"]["cis_val_clean"]["frame_f2"] == 1.0
        assert result["primary"] == 1.0
