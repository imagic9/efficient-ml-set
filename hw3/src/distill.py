"""Knowledge Distillation: KD loss + a KD-aware training loop.

Standard Hinton (2015) distillation. The student is trained against two signals:

  * hard labels -- ordinary cross-entropy against the ground-truth class;
  * soft labels -- the teacher's softened class distribution, matched with a
    KL-divergence at temperature T.

    L = alpha * CE(student, y)  +  (1 - alpha) * T^2 * KL(teacher_T || student_T)

Temperature T > 1 flattens both distributions so the student also learns the
teacher's "dark knowledge" (the relative probabilities of the wrong classes,
e.g. a cat image being a bit dog-like and not at all truck-like). The T^2 factor
rescales the soft-loss gradient, which otherwise shrinks like 1/T^2, back to the
same order as the CE gradient so a single learning rate works for any T.

alpha trades the two terms: alpha=1 is pure CE (no teacher -- our recovery
baseline), alpha=0 is pure distillation. We keep both terms by default.

The teacher here is our own uncompressed fp32 VGG11 -- this is *self*-distillation
(same architecture, teacher just isn't compressed), which is exactly the HW3 setup.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .engine import evaluate
from .utils import AverageMeter


class DistillLoss(nn.Module):
    """Combined hard-label + soft-label distillation loss.

    forward(student_logits, teacher_logits, targets) -> scalar loss.

    When alpha == 1 the KL term is skipped entirely, so this same object also
    expresses the pure cross-entropy recovery baseline (teacher_logits may be
    None in that case). That lets the KD run and the CE-only run share one code
    path -- the only thing that changes is alpha -- which is what makes the
    KD-vs-CE comparison a fair one.
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        super().__init__()
        self.T = float(temperature)
        self.alpha = float(alpha)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, targets):
        ce = self.ce(student_logits, targets)
        if self.alpha >= 1.0 or teacher_logits is None:
            return ce
        T = self.T
        # Standard KD direction: KL(teacher || student) at temperature T. F.kl_div
        # computes sum(target * (log target - input)) = D_KL(target || input), so
        # passing input=log-probs(student), target=probs(teacher) yields exactly
        # D_KL(teacher || student) -- teacher is the reference distribution the
        # student is pulled towards. batchmean gives the correct per-sample average.
        # The teacher targets are a constant, so detach() before softmax: the loss
        # is correct even if the caller hands us grad-enabled teacher logits.
        soft_student = F.log_softmax(student_logits / T, dim=1)
        soft_teacher = F.softmax(teacher_logits.detach() / T, dim=1)
        kd = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T * T)
        return self.alpha * ce + (1.0 - self.alpha) * kd


@torch.no_grad()
def _teacher_logits(teacher, images):
    # no_grad (not inference_mode): the resulting logits are used as a constant
    # target inside the student's autograd graph, and inference-mode tensors
    # cannot participate in autograd-tracked ops.
    return teacher(images)


def kd_train_one_epoch(student, teacher, loader, optimizer, device, distill,
                       pruner=None, progress=False):
    """One epoch of KD training for a dense or pruned student.

    teacher runs in eval/no-grad and provides soft targets. If a pruner is given
    its masks are re-applied after every step so pruned weights stay at zero --
    this is how the pruned student is fine-tuned without losing its sparsity.
    """
    from tqdm import tqdm
    student.train()
    if teacher is not None:
        teacher.eval()
    loss_m, acc_m = AverageMeter(), AverageMeter()
    iterator = tqdm(loader, leave=False) if progress else loader
    for images, targets in iterator:
        images, targets = images.to(device), targets.to(device)
        t_logits = _teacher_logits(teacher, images) if teacher is not None else None
        optimizer.zero_grad()
        s_logits = student(images)
        loss = distill(s_logits, t_logits, targets)
        loss.backward()
        optimizer.step()
        if pruner is not None:
            pruner.apply(student)
        acc_m.update((s_logits.argmax(1) == targets).float().mean().item(), images.size(0))
        loss_m.update(loss.item(), images.size(0))
    return loss_m.avg, acc_m.avg


def kd_train(student, teacher, train_loader, val_loader, device, epochs,
             distill, lr=0.01, momentum=0.9, weight_decay=5e-4, pruner=None,
             log_prefix="", progress=False):
    """SGD + cosine KD fine-tuning of a (possibly pruned) student.

    Mirrors engine.train exactly -- same optimiser, schedule, model-selection --
    so a CE-only run (DistillLoss(alpha=1)) and a KD run differ only in the loss.
    Returns (history, best_val_acc, best_state).
    """
    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=momentum,
                                weight_decay=weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    val_criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, best_state = 0.0, None
    for epoch in range(epochs):
        tr_loss, tr_acc = kd_train_one_epoch(student, teacher, train_loader,
                                             optimizer, device, distill, pruner, progress)
        va_loss, va_acc = evaluate(student, val_loader, device, val_criterion)
        scheduler.step()
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
        print(f"{log_prefix}epoch {epoch + 1:3d}/{epochs} "
              f"train_acc={tr_acc:.4f} val_acc={va_acc:.4f} best={best_acc:.4f}")
    return history, best_acc, best_state
