"""Unit tests for the NAS search space and the standalone network builder:

  * the space enumerates exactly 3^4 x 4 x 5 = 1620 architectures;
  * a sampled architecture is well-formed and StandaloneNet produces [B, 10] logits;
  * count_arch_params equals the built module's real parameter count (the number we
    report for "parameter count" must be the honest one);
  * width multiplier and block operation both move the parameter count the expected
    direction (wider => more params; dwsep/mbconv => fewer than conv3x3).

Run with pytest, or directly:  python tests/test_search_space.py
"""
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.search_space import (StandaloneNet, count_arch_params, sample_arch,  # noqa: E402
                              space_size, OPS, WIDTHS, ACTS, NUM_STAGES)


def test_space_size():
    assert space_size() == len(OPS) ** NUM_STAGES * len(WIDTHS) * len(ACTS) == 1620


def test_sampled_arch_forward():
    rng = random.Random(0)
    for _ in range(10):
        arch = sample_arch(rng)
        assert len(arch["ops"]) == NUM_STAGES
        assert all(o in OPS for o in arch["ops"])
        assert arch["width"] in WIDTHS and arch["act"] in ACTS
        net = StandaloneNet(arch)
        assert net(torch.randn(4, 3, 32, 32)).shape == (4, 10)


def test_param_count_matches_module():
    rng = random.Random(1)
    for _ in range(10):
        arch = sample_arch(rng)
        net = StandaloneNet(arch)
        real = sum(p.numel() for p in net.parameters())
        assert count_arch_params(arch) == real


def test_width_monotonic_params():
    base = {"ops": ["conv3x3"] * NUM_STAGES, "act": "relu"}
    counts = [count_arch_params({**base, "width": w}) for w in sorted(WIDTHS)]
    assert counts == sorted(counts) and counts[0] < counts[-1]


def test_efficient_ops_are_smaller():
    conv = count_arch_params({"ops": ["conv3x3"] * NUM_STAGES, "width": 1.0, "act": "relu"})
    dwsep = count_arch_params({"ops": ["dwsep"] * NUM_STAGES, "width": 1.0, "act": "relu"})
    mbconv = count_arch_params({"ops": ["mbconv"] * NUM_STAGES, "width": 1.0, "act": "relu"})
    assert dwsep < conv                      # depthwise separable is far leaner
    assert mbconv < conv                     # inverted residual (expand=3) still < full conv


if __name__ == "__main__":
    for fn in [test_space_size, test_sampled_arch_forward,
               test_param_count_matches_module, test_width_monotonic_params,
               test_efficient_ops_are_smaller]:
        fn()
        print(f"ok  {fn.__name__}")
    print("all search-space tests passed")
