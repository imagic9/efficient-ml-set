"""The deployment-regime rule for validation score dumps (DESIGN §6.3, issue #30).

`dump_predictions` feeds every calibration, and the device it calibrates for computes
true FP32 — so the dump must too. These tests pin the two halves of the amendment:
the regime is actually enforced, and the npz says which regime it was written under.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wildlife_trigger.validate import dump_predictions as D
from wildlife_trigger.validate import parity as P


@pytest.fixture()
def _restore_tf32():
    """TF32 flags are process-global; a test that flips them must put them back,
    or every later test in the process runs in an arithmetic it did not choose."""
    saved = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    yield
    torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = saved


class TestDeploymentRegime:
    def test_tf32_is_disabled_everywhere(self, _restore_tf32) -> None:
        """Both knobs, not just cuDNN: convolutions and the classifier's matmul
        must compute what the exported FP32 graph computes."""
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        D.enforce_deployment_regime()
        assert torch.backends.cudnn.allow_tf32 is False
        assert torch.backends.cuda.matmul.allow_tf32 is False


class TestNpzRegimeKey:
    def test_legacy_npz_reads_as_tf32_on(self, tmp_path) -> None:
        """The seed-42 npz predates the key and ran under torch 2.11's default.
        Reading it as anything but TF32-on would reproduce it under a regime it
        was never written with — the exact mis-specification the P2 guard was
        corrected for."""
        np.savez_compressed(tmp_path / "predictions.npz", run_name=np.array("legacy"))
        assert P.npz_cudnn_tf32(tmp_path) is True

    def test_amended_npz_reads_as_tf32_off(self, tmp_path) -> None:
        np.savez_compressed(
            tmp_path / "predictions.npz",
            run_name=np.array("post_amendment"),
            cudnn_tf32=np.array(False),
        )
        assert P.npz_cudnn_tf32(tmp_path) is False
