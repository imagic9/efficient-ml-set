"""Unit tests for the weight-sharing supernet:

  * every block operation forwards to valid [B, 10] logits for random archs;
  * the sub-path the supernet evaluates is *structurally identical* to the
    StandaloneNet we retrain -- each sliced shared weight has exactly the same
    shape as the corresponding standalone conv weight (so "search the subnet, then
    retrain that architecture" is really the same architecture);
  * weight sharing works: a sub-network's backward reaches only the active channel
    slice of the shared weight and leaves the rest untouched (the overlap is what
    lets every sampled path train the same parameters);
  * BN recalibration actually refreshes the active path's running statistics.

Run with pytest, or directly:  python tests/test_supernet.py
"""
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.supernet import SuperNet, recalibrate_bn                      # noqa: E402
from src.search_space import (StandaloneNet, sample_arch, stage_channels,  # noqa: E402
                              NUM_STAGES, OPS, MB_EXPAND)


def _stage_io(arch):
    """(in_c, out_c) per stage for an architecture."""
    chs = stage_channels(arch["width"])
    io, in_c = [], 3
    for i in range(NUM_STAGES):
        io.append((in_c, chs[i]))
        in_c = chs[i]
    return io


def test_all_ops_forward():
    torch.manual_seed(0)
    net = SuperNet()
    x = torch.randn(4, 3, 32, 32)
    for op in OPS:                                   # a uniform-op arch for each op type
        arch = {"ops": [op] * NUM_STAGES, "width": 1.0, "act": "relu"}
        assert net(x, arch, training=False).shape == (4, 10)
    # a fully mixed arch also runs
    rng = random.Random(3)
    assert net(x, sample_arch(rng), training=False).shape == (4, 10)


def test_subnet_shapes_match_standalone():
    """The sliced shared weights must match the standalone conv shapes exactly."""
    net = SuperNet()
    rng = random.Random(1)
    for _ in range(8):
        arch = sample_arch(rng)
        std = StandaloneNet(arch)
        io = _stage_io(arch)
        for i in range(NUM_STAGES):
            op = arch["ops"][i]
            in_c, out_c = io[i]
            shared = net.stages[i][op]
            block = std.blocks[i]
            if op == "conv3x3":
                assert shared.weight[:out_c, :in_c].shape == block.conv.weight.shape
            elif op == "dwsep":
                assert shared.dw[:in_c].shape == block.dw.weight.shape
                assert shared.pw[:out_c, :in_c].shape == block.pw.weight.shape
            else:  # mbconv
                hid = in_c * MB_EXPAND
                assert shared.expand_w[:hid, :in_c].shape == block.expand.weight.shape
                assert shared.dw[:hid].shape == block.dw.weight.shape
                assert shared.project_w[:out_c, :hid].shape == block.project.weight.shape


def test_weight_sharing_gradient_slice():
    """A narrow-width sub-net's backward touches only the active slice of the
    shared conv weight; a wider sub-net of the same op touches a strictly larger
    region -- the overlap is the shared signal."""
    torch.manual_seed(0)
    net = SuperNet()
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))

    narrow = {"ops": ["conv3x3"] * NUM_STAGES, "width": 0.5, "act": "relu"}
    wide = {"ops": ["conv3x3"] * NUM_STAGES, "width": 1.25, "act": "relu"}
    w0 = net.stages[0]["conv3x3"].weight                # first-stage shared conv weight

    net.zero_grad()
    F.cross_entropy(net(x, narrow, training=True), y).backward()
    out_n = stage_channels(narrow["width"])[0]
    g_narrow = w0.grad.clone()
    assert g_narrow[:out_n, :3].abs().sum() > 0         # active slice gets gradient
    assert g_narrow[out_n:].abs().sum() == 0            # the rest is untouched

    net.zero_grad()
    F.cross_entropy(net(x, wide, training=True), y).backward()
    out_w = stage_channels(wide["width"])[0]
    g_wide = w0.grad.clone()
    assert out_w > out_n
    assert g_wide[:out_w, :3].abs().sum() > g_wide[:out_n, :3].abs().sum()  # touches more


def test_bn_recalibration_updates_stats():
    torch.manual_seed(0)
    net = SuperNet()
    arch = {"ops": ["conv3x3"] * NUM_STAGES, "width": 1.0, "act": "relu"}
    loader = [(torch.randn(16, 3, 32, 32), torch.randint(0, 10, (16,))) for _ in range(6)]
    bn = net.stages[0]["conv3x3"].bn
    before = bn.running_mean.clone()
    recalibrate_bn(net, arch, loader, device="cpu", num_batches=4)
    # the active slice's running mean has moved away from the reset value
    assert not torch.allclose(bn.running_mean, before)


if __name__ == "__main__":
    for fn in [test_all_ops_forward, test_subnet_shapes_match_standalone,
               test_weight_sharing_gradient_slice, test_bn_recalibration_updates_stats]:
        fn()
        print(f"ok  {fn.__name__}")
    print("all supernet tests passed")
