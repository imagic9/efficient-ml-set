"""The C4 export tool, end to end on fabricated run directories (PLAN C4).

Export is part of correctness (DESIGN §10): these tests are mostly about the
refusals — the tool must not produce an artifact whose lineage it cannot prove.
The happy path exports a real (randomly initialised) MobileNetV2 at a small input
geometry so the whole graph pipeline runs in seconds.
"""

from __future__ import annotations

import json

import onnx
import pytest
import torch

from wildlife_trigger import export as E
from wildlife_trigger.models.mobilenet import build_mobilenet_v2
from wildlife_trigger.runs import sha256_file

CLASS_NAMES = [
    "opossum", "raccoon", "squirrel", "bobcat", "skunk", "dog", "coyote", "rabbit",
    "bird", "cat", "badger", "empty", "car", "deer", "fox", "rodent",
]

# Small but real: 64x64 keeps MobileNetV2's five stride-2 stages valid (2x2 final
# feature map) while making export fast enough for a unit test.
WIDTH, HEIGHT = 64, 64


@pytest.fixture(scope="module")
def run_dir_factory(tmp_path_factory):
    """One trained-looking run directory per test, sharing a single state dict.

    Building MobileNetV2 once per module keeps the suite fast; each test still
    gets its own directory so mutations cannot leak between tests.
    """
    torch.manual_seed(0)
    state = build_mobilenet_v2(num_classes=len(CLASS_NAMES), pretrained=False).state_dict()

    def make(best_epoch: int = 11, recorded_epoch: int | None = None):
        run_dir = tmp_path_factory.mktemp("run")
        torch.save({"model": state, "epoch": recorded_epoch or best_epoch},
                   run_dir / "best.pt")
        (run_dir / "history.json").write_text(json.dumps({
            "run_name": "m0_test",
            "best_epoch": best_epoch,
            "class_names": CLASS_NAMES,
            "config": {"width": WIDTH, "height": HEIGHT},
        }))
        (run_dir / "hashes.json").write_text(json.dumps({
            "checkpoint:best": {
                "path": str(run_dir / "best.pt"),
                "sha256": sha256_file(run_dir / "best.pt"),
            },
        }))
        (run_dir / "run_summary.json").write_text(
            json.dumps({"run_id": "c4_m0_test_20260716T000000Z"})
        )
        return run_dir

    return make


def export(tmp_path, run_dir, **kwargs):
    return E.export_run(run_dir, evidence_root=tmp_path / "parity", **kwargs)


class TestHappyPath:
    def test_exports_verified_opset17_with_contracted_io(self, tmp_path, run_dir_factory) -> None:
        run_dir = run_dir_factory()
        result = export(tmp_path, run_dir)

        assert result["onnx"]["opset_import"][""] == 17
        (input_spec,) = result["onnx"]["inputs"]
        assert input_spec["name"] == "input"
        assert input_spec["shape"] == [1, 3, HEIGHT, WIDTH], (
            "batch stays static 1 and geometry comes from the run's own config"
        )
        (output_spec,) = result["onnx"]["outputs"]
        assert output_spec["name"] == "logits"
        assert output_spec["shape"] == [1, len(CLASS_NAMES)]

    def test_provenance_is_inside_the_graph(self, tmp_path, run_dir_factory) -> None:
        """An ONNX file found loose on a Pi must still say what it is."""
        run_dir = run_dir_factory()
        result = export(tmp_path, run_dir)

        model = onnx.load(result["onnx"]["path"], load_external_data=False)
        metadata = {p.key: p.value for p in model.metadata_props}
        assert metadata["wildlife_trigger.run_id"] == "c4_m0_test_20260716T000000Z"
        assert metadata["wildlife_trigger.best_epoch"] == "11"
        assert metadata["wildlife_trigger.checkpoint_sha256"] == result["checkpoint"]["sha256"]
        assert "NCHW RGB" in metadata["wildlife_trigger.input_contract"]

    def test_export_is_byte_reproducible(self, tmp_path, run_dir_factory) -> None:
        """Same weights, same versions, same bytes: no timestamp or commit may
        leak into the graph, or the hash the policy re-binds to breaks on every
        re-export of an unchanged model."""
        run_dir = run_dir_factory()
        first = export(tmp_path / "a", run_dir, output=tmp_path / "a" / "m.onnx")
        second = export(tmp_path / "b", run_dir, output=tmp_path / "b" / "m.onnx")
        assert first["onnx"]["sha256"] == second["onnx"]["sha256"]

    def test_evidence_is_written_under_the_run_id(self, tmp_path, run_dir_factory) -> None:
        run_dir = run_dir_factory()
        result = export(tmp_path, run_dir)
        evidence = json.loads(
            (tmp_path / "parity" / "c4_m0_test_20260716T000000Z" / "export.json").read_text()
        )
        assert evidence["onnx"]["sha256"] == result["onnx"]["sha256"]
        assert evidence["versions"]["torch"] == torch.__version__

    def test_policy_binding_check_passes_when_bound_to_this_checkpoint(
        self, tmp_path, run_dir_factory
    ) -> None:
        run_dir = run_dir_factory()
        checkpoint_sha = json.loads((run_dir / "hashes.json").read_text())[
            "checkpoint:best"
        ]["sha256"]
        policy_path = tmp_path / "bobcat_v1.json"
        policy_path.write_text(json.dumps({"model_sha256": checkpoint_sha}))
        result = export(tmp_path, run_dir, policy_path=policy_path)
        assert result["policy_check"]["model_sha256_matches_checkpoint"] is True


class TestRefusals:
    def test_overwritten_checkpoint_is_refused(self, tmp_path, run_dir_factory) -> None:
        """A best.pt that no longer hashes to the run's record is someone else's
        weights wearing this run's name."""
        run_dir = run_dir_factory()
        blob = (run_dir / "best.pt").read_bytes()
        (run_dir / "best.pt").write_bytes(blob + b"tamper")
        with pytest.raises(RuntimeError, match="overwritten"):
            export(tmp_path, run_dir)

    def test_epoch_mismatch_is_refused(self, tmp_path, run_dir_factory) -> None:
        run_dir = run_dir_factory(best_epoch=11, recorded_epoch=12)
        # hashes.json was written for this exact file, so the hash check passes
        # and the epoch check is what must catch it.
        with pytest.raises(RuntimeError, match="epoch 12"):
            export(tmp_path, run_dir)

    def test_policy_bound_to_other_weights_is_refused(
        self, tmp_path, run_dir_factory
    ) -> None:
        """The C3->C4 chain: exporting a graph does not entitle anyone to re-bind
        a policy that was calibrated on different weights."""
        run_dir = run_dir_factory()
        policy_path = tmp_path / "bobcat_v1.json"
        policy_path.write_text(json.dumps({"model_sha256": "0" * 64}))
        with pytest.raises(RuntimeError, match="calibrated on different weights"):
            export(tmp_path, run_dir, policy_path=policy_path)

    def test_missing_checkpoint_hash_is_refused(self, tmp_path, run_dir_factory) -> None:
        run_dir = run_dir_factory()
        (run_dir / "hashes.json").write_text(json.dumps({}))
        with pytest.raises(RuntimeError, match="checkpoint:best"):
            export(tmp_path, run_dir)
