"""Unit tests for the KD loss -- the easy-to-get-wrong distillation piece:

  * alpha=1 collapses to plain cross-entropy (our CE-only baseline shares the path);
  * a perfectly-matched student (student logits == teacher logits) has ~zero KL,
    so at alpha=0 the loss is ~0 -- the temperature scaling is wired correctly;
  * gradients flow into the student but never into the frozen teacher.

Run with pytest, or directly:  python tests/test_distill.py
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.distill import DistillLoss  # noqa: E402


def test_alpha_one_equals_cross_entropy():
    """alpha=1 must ignore the teacher and equal nn.CrossEntropyLoss exactly."""
    torch.manual_seed(0)
    student = torch.randn(16, 10, requires_grad=True)
    teacher = torch.randn(16, 10)
    targets = torch.randint(0, 10, (16,))
    kd = DistillLoss(temperature=4.0, alpha=1.0)
    got = kd(student, teacher, targets)
    expected = F.cross_entropy(student, targets)
    assert torch.allclose(got, expected, atol=1e-6)
    # teacher can even be None at alpha=1
    assert torch.allclose(kd(student, None, targets), expected, atol=1e-6)


def test_matched_logits_zero_kl():
    """If the student already matches the teacher, the pure-KD loss (alpha=0) is ~0."""
    torch.manual_seed(1)
    logits = torch.randn(16, 10)
    targets = torch.randint(0, 10, (16,))
    kd = DistillLoss(temperature=3.0, alpha=0.0)
    loss = kd(logits.clone().requires_grad_(True), logits.clone(), targets)
    assert loss.abs().item() < 1e-5


def test_gradient_reaches_student_not_teacher():
    """Backprop must update the student logits and leave the teacher untouched.

    We hand the loss a grad-enabled teacher *without* detaching it ourselves, so
    this genuinely checks that DistillLoss detaches the teacher targets internally
    (they are a constant reference distribution -- no gradient should flow back).
    """
    torch.manual_seed(2)
    student = torch.randn(8, 10, requires_grad=True)
    teacher = torch.randn(8, 10, requires_grad=True)   # deliberately grad-enabled
    targets = torch.randint(0, 10, (8,))
    kd = DistillLoss(temperature=4.0, alpha=0.5)
    kd(student, teacher, targets).backward()
    assert student.grad is not None and student.grad.abs().sum() > 0
    assert teacher.grad is None                        # loss detached the teacher


def test_temperature_scaling_keeps_soft_gradient_order():
    """The T^2 factor keeps the soft-loss magnitude on the same order across T,
    so a single learning rate works for any temperature (sanity, not exact)."""
    torch.manual_seed(3)
    student = torch.randn(32, 10)
    teacher = torch.randn(32, 10)
    targets = torch.randint(0, 10, (32,))
    vals = []
    for T in (1.0, 4.0, 8.0):
        kd = DistillLoss(temperature=T, alpha=0.0)
        vals.append(kd(student.clone().requires_grad_(True), teacher, targets).item())
    # without the T^2 rescale these would shrink ~1/T^2 (64x from T=1 to T=8);
    # with it they stay within a small factor of each other
    assert max(vals) / min(vals) < 4.0


if __name__ == "__main__":
    for fn in [test_alpha_one_equals_cross_entropy, test_matched_logits_zero_kl,
               test_gradient_reaches_student_not_teacher,
               test_temperature_scaling_keeps_soft_gradient_order]:
        fn()
        print(f"ok  {fn.__name__}")
    print("all distill tests passed")
