"""Centroid fine-tuning loop (QAT) for K-Means weight sharing -- slide 36.

The optimiser holds ONLY the centroids (plus the small unquantized params: biases
and BatchNorm affine terms). The conv/linear weights are never stepped directly --
they are re-materialised from the codebooks each iteration. This is the literal
"update the centroids" box from the diagram.
"""
import torch
import torch.nn as nn

from .engine import evaluate
from .kmeans_quant import quantizable_layers
from .utils import AverageMeter


@torch.no_grad()
def recalibrate_bn(model, quantizer, loader, device, num_batches=50):
    """Refresh BatchNorm running stats to match the current (quantized) weights.

    During QAT the weights shift every step, so the running mean/var that eval()
    relies on lag behind. We reset them and re-accumulate over a few hundred
    training images with the final weights in place -- a standard, cheap fix that
    closes the train/eval gap without touching the learned parameters.
    """
    quantizer.reconstruct(model)
    saved = {}
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            saved[m] = m.momentum
            m.reset_running_stats()
            m.momentum = None            # cumulative moving average over the pass
    model.train()
    seen = 0
    for images, _ in loader:
        model(images.to(device))
        seen += 1
        if seen >= num_batches:
            break
    model.eval()
    for m, mom in saved.items():
        m.momentum = mom                 # restore for subsequent training epochs


def _extra_trainable_params(model, quantizer):
    """Everything trainable that is NOT a quantized weight.

    Biases and BatchNorm weight/bias are tiny and stay fp32; letting them adapt
    a little helps the centroids recover accuracy. The quantized conv/linear
    *weights* are excluded -- they only change via reconstruct().
    """
    q_weight_ids = {id(dict(quantizable_layers(model))[n].weight)
                    for n in quantizer.bits}
    extras = []
    for p in model.parameters():
        if p.requires_grad and id(p) not in q_weight_ids:
            extras.append(p)
    return extras


def qat_finetune(model, quantizer, train_loader, val_loader, device, epochs,
                 lr=1e-3, weight_decay=0.0, adapt_extras=True, pool_mode="mean",
                 optim_name="adam", sgd_lr=0.05, clip_norm=1.0, log_prefix="",
                 teacher=None, distill=None):
    """Fine-tune centroids via gradient pooling. Returns (history, best_val, best).

    Knowledge distillation (HW3): pass a frozen `teacher` and a `distill`
    (DistillLoss) to fine-tune the centroids against the teacher's soft targets
    instead of plain cross-entropy. With teacher=None this is the exact HW2 QAT
    loop, so the CE-only and KD centroid-recovery runs share one code path.

    best is a snapshot to restore the best-val state: (codebook_state, model_state).

    What moves during QAT: the QUANTIZED weights (Conv/Linear) change ONLY through
    their centroids -- individual weights are never stepped, they are re-materialised
    as codebook[index] each batch. The small un-quantized parameters (biases and
    BatchNorm affine gamma/beta) stay fp32; adapt_extras=True (default) lets them
    adapt too, which recovers a bit more accuracy (quantified in the ablation),
    while adapt_extras=False keeps everything trainable frozen except the centroids.

    Optimiser choice matters here. The centroids are tied weights (millions of
    weights share one value), so their loss surface is sharp and the pooled-gradient
    scale varies wildly between layers -- plain SGD+momentum overshoots and diverges
    within a couple of epochs. Adam normalises each centroid's step by its own
    gradient statistics, which tames both problems; clip_norm is an extra safety net.
    """
    criterion = nn.CrossEntropyLoss()
    if teacher is not None:
        teacher.eval()
    params = quantizer.centroid_parameters()
    if adapt_extras:
        params = params + _extra_trainable_params(model, quantizer)
    if optim_name == "sgd":                       # only used by the ablation
        optimizer = torch.optim.SGD(params, lr=sgd_lr, momentum=0.9, nesterov=True)
    else:
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, best = 0.0, None
    for epoch in range(epochs):
        model.train()
        loss_m, acc_m = AverageMeter(), AverageMeter()
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            quantizer.reconstruct(model)          # weight = codebook[index]
            # zero the WHOLE model's grads, not just the optimiser's: the conv/linear
            # weights are leaves outside the optimiser, so their .grad would otherwise
            # accumulate across batches and blow the pooled centroid gradient up.
            model.zero_grad(set_to_none=True)
            optimizer.zero_grad()
            logits = model(images)
            if teacher is not None:
                with torch.no_grad():             # constant target; not inference_mode
                    t_logits = teacher(images)
                loss = distill(logits, t_logits, targets)
            else:
                loss = criterion(logits, targets)
            loss.backward()                       # fills weight.grad (fresh each batch)
            quantizer.pool_gradients(model, mode=pool_mode)  # weight.grad -> codebook.grad
            if clip_norm:
                torch.nn.utils.clip_grad_norm_(params, clip_norm)
            optimizer.step()                      # steps centroids (+ extras)
            loss_m.update(loss.item(), images.size(0))
            acc_m.update((logits.argmax(1) == targets).float().mean().item(),
                         images.size(0))
        quantizer.reconstruct(model)              # sync weights to updated codebook
        scheduler.step()

        # refresh BN stats to the shifted weights so val (and model selection) is trustworthy
        recalibrate_bn(model, quantizer, train_loader, device)
        va_loss, va_acc = evaluate(model, val_loader, device, criterion)
        history["train_loss"].append(loss_m.avg)
        history["train_acc"].append(acc_m.avg)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        if va_acc > best_acc:
            best_acc = va_acc
            best = (
                {n: cb.detach().cpu().clone() for n, cb in quantizer.codebooks.items()},
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            )
        print(f"{log_prefix}epoch {epoch + 1:2d}/{epochs} "
              f"train_acc={acc_m.avg:.4f} val_acc={va_acc:.4f} best={best_acc:.4f}")
    return history, best_acc, best


def restore_best(model, quantizer, best):
    """Load the best-val snapshot back into model + quantizer codebooks."""
    codebook_state, model_state = best
    model.load_state_dict({k: v.to(next(model.parameters()).device)
                           for k, v in model_state.items()})
    for n, cb in codebook_state.items():
        quantizer.codebooks[n].data = cb.to(quantizer.codebooks[n].device)
