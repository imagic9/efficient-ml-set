"""D5: the QAT trainer generalized to a pruned source (M4 = c30 + QAT).

The training run itself is gx10's; these tests prove the shape-plumbing that
would otherwise only fail three hours into a run: that a pruned base accepts
the QAT structure, that the width pins are enforced, and that M4's candidate
identity is carried into the evidence distinct from M2's.
"""

from __future__ import annotations

import pytest
import torch

from wildlife_trigger.models.mobilenet import build_mobilenet_v2
from wildlife_trigger.optimize import prune as P
from wildlife_trigger.optimize.qat import build_qat_model
from wildlife_trigger.optimize.qat_train import (
    QatConfig,
    build_source_architecture,
    load_config,
)

NUM_CLASSES = 16
# A small but real pruned width set (features.2 halved), enough to prove shape.
PRUNED = {"features.2": 48, "features.17": 480}


class TestBuildSourceArchitecture:
    def test_unpruned_when_no_widths(self):
        config = QatConfig()
        model = build_source_architecture(config, [f"c{i}" for i in range(NUM_CLASSES)])
        assert model.features[2].conv[0][0].out_channels == 96  # torchvision default

    def test_pruned_when_widths_given(self):
        config = QatConfig(pruned_widths=PRUNED)
        model = build_source_architecture(config, [f"c{i}" for i in range(NUM_CLASSES)])
        assert model.features[2].conv[0][0].out_channels == 48
        assert model.features[17].conv[0][0].out_channels == 480
        # coupling held
        dw = model.features[2].conv[1][0]
        assert dw.in_channels == dw.out_channels == dw.groups == 48


class TestQatOnPrunedBase:
    def test_qat_structure_accepts_a_pruned_base(self):
        """build_qat_model scans children, so it must not care about widths."""
        config = QatConfig(pruned_widths=PRUNED)
        base = build_source_architecture(config, [f"c{i}" for i in range(NUM_CLASSES)])
        model, structure = build_qat_model(base=base)
        assert structure["convolutions_quantized"] > 0
        # a forward pass on the frozen input shape must run end to end
        model.eval()
        with torch.inference_mode():
            out = model(torch.zeros(1, 3, 192, 256))
        assert out.shape == (1, NUM_CLASSES)

    def test_pruned_qat_state_dict_round_trips(self):
        """A QAT model built on a pruned base can save and reload its own
        state — the D5 export path rebuilds this exact structure."""
        config = QatConfig(pruned_widths=PRUNED)
        names = [f"c{i}" for i in range(NUM_CLASSES)]
        base = build_source_architecture(config, names)
        model, _ = build_qat_model(base=base)
        state = model.state_dict()

        base2 = build_source_architecture(config, names)
        model2, _ = build_qat_model(base=base2)
        model2.load_state_dict(state)  # must not shape-mismatch
        model.eval(), model2.eval()
        with torch.inference_mode():
            x = torch.zeros(1, 3, 192, 256)
            assert torch.equal(model(x), model2(x))


class TestWidthPinEnforcement:
    def test_checkpoint_widths_must_match_config(self, tmp_path):
        """load_m0_base refuses a checkpoint whose widths differ from the pins."""
        from wildlife_trigger.optimize.qat_train import load_m0_base
        from wildlife_trigger.runs import sha256_file

        names = [f"c{i}" for i in range(NUM_CLASSES)]
        model = build_mobilenet_v2(num_classes=NUM_CLASSES, pretrained=False)
        P.prune_expansion(model, {2: 0.5}, export_check=False)  # features.2 -> 48
        ckpt = tmp_path / "src.pt"
        torch.save(
            {"model": model.state_dict(), "epoch": 2,
             "widths": {"features.2": 48}, "class_names": names},
            ckpt,
        )
        sha = sha256_file(ckpt)

        # config claims a different width than the checkpoint records
        config = QatConfig(
            source_checkpoint=str(ckpt), source_checkpoint_sha256=sha,
            pruned_widths={"features.2": 56},
        )
        with pytest.raises(RuntimeError, match="differ from the checkpoint"):
            load_m0_base(config, names)

    def test_matching_widths_load(self, tmp_path):
        from wildlife_trigger.optimize.qat_train import load_m0_base
        from wildlife_trigger.runs import sha256_file

        names = [f"c{i}" for i in range(NUM_CLASSES)]
        model = build_mobilenet_v2(num_classes=NUM_CLASSES, pretrained=False)
        P.prune_expansion(model, {2: 0.5}, export_check=False)
        ckpt = tmp_path / "src.pt"
        torch.save(
            {"model": model.state_dict(), "epoch": 2,
             "widths": {"features.2": 48}, "class_names": names},
            ckpt,
        )
        sha = sha256_file(ckpt)
        config = QatConfig(
            source_run_id="d4_m3_c30", source_checkpoint=str(ckpt),
            source_checkpoint_sha256=sha, pruned_widths={"features.2": 48},
        )
        base, info = load_m0_base(config, names)
        assert base.features[2].conv[0][0].out_channels == 48
        assert info["pruned_widths"] == {"features.2": 48}


class TestConfigIdentity:
    def test_m4_config_parses_with_pruned_widths(self):
        config = load_config(
            __import__("pathlib").Path("configs/optimize/m4_qat.yaml")
        )
        assert config.candidate_prefix == "d5_m4_qat"
        assert config.candidate_kind == "pruned_qat"
        assert config.candidate_design == "8.4"
        assert config.pruned_widths["features.2"] == 48
        assert config.pruned_widths["features.17"] == 544
        assert config.source_run_id == "d4_m3_c30_20260717T052416Z"

    def test_m2_defaults_are_byte_stable(self):
        """M2's identity must not have shifted under the parameterization."""
        config = QatConfig()
        assert config.candidate_prefix == "d2_m2_qat"
        assert config.candidate_kind == "int8_qat"
        assert config.candidate_model_id == "M2-candidate"
        assert config.candidate_design == "8.2"
        assert config.pruned_widths == {}
