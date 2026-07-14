"""Unit tests for the elastic VGG11 used by NetAug:

  * base and augmented forward both produce valid [B, num_classes] logits;
  * the base net is a genuine sub-network -- its conv/head use a *slice* of the
    shared augmented weights, so a gradient from the augmented forward reaches the
    same parameters the base net uses (that shared signal is the whole point);
  * the base sub-network really is smaller than the augmented one.

Run with pytest, or directly:  python tests/test_netaug.py
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.netaug import ElasticVGG11  # noqa: E402


def test_both_modes_produce_logits():
    torch.manual_seed(0)
    m = ElasticVGG11(base_mult=0.25, aug_mult=1.0)
    x = torch.randn(4, 3, 32, 32)
    assert m(x, mode="base").shape == (4, 10)
    assert m(x, mode="aug").shape == (4, 10)


def test_base_is_smaller_than_aug():
    m = ElasticVGG11(base_mult=0.25, aug_mult=1.0)
    # base widths strictly below aug widths, and the reported base param count is
    # far below the shared (augmented) parameter store
    assert all(b < a for b, a in zip(m.base, m.aug))
    shared = sum(p.numel() for p in m.conv_w) + m.head_w.numel() + m.head_b.numel()
    assert m.base_param_count() < shared


def test_shared_weights_get_gradient_from_both_forwards():
    """The overlapping (base) region of the first conv weight must receive gradient
    contributions from BOTH the base and the augmented forward -- that shared
    training signal is exactly what NetAug relies on."""
    torch.manual_seed(1)
    m = ElasticVGG11(base_mult=0.25, aug_mult=1.0)
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))

    # gradient from the base forward alone
    m.zero_grad()
    F.cross_entropy(m(x, mode="base"), y).backward()
    g_base = m.conv_w[0].grad.clone()

    # gradient from the augmented forward alone
    m.zero_grad()
    F.cross_entropy(m(x, mode="aug"), y).backward()
    g_aug = m.conv_w[0].grad.clone()

    b_out, b_in = m.base[0], 3
    # base forward only touches the top-left slice; aug forward touches everything
    assert g_base[:b_out, :b_in].abs().sum() > 0
    assert g_base[b_out:].abs().sum() == 0                     # base leaves the rest untouched
    assert g_aug.abs().sum() > g_aug[:b_out, :b_in].abs().sum()  # aug touches more than the slice
    assert g_aug[:b_out, :b_in].abs().sum() > 0                # ...including the shared slice


if __name__ == "__main__":
    for fn in [test_both_modes_produce_logits, test_base_is_smaller_than_aug,
               test_shared_weights_get_gradient_from_both_forwards]:
        fn()
        print(f"ok  {fn.__name__}")
    print("all netaug tests passed")
