"""Training / evaluation loops with optional pruning-mask re-application."""
import torch
import torch.nn as nn
from tqdm import tqdm

from .utils import AverageMeter


@torch.inference_mode()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        preds = logits.argmax(dim=1)
        acc = (preds == targets).float().mean().item()
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))
    return loss_meter.avg, acc_meter.avg


def train_one_epoch(model, loader, optimizer, device, criterion=None,
                    pruner=None, progress=False):
    """One pass over the training set.

    If a pruner is given, its masks are re-applied after every optimizer step so
    that weights zeroed by pruning stay at zero while the rest keep training.
    """
    model.train()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    iterator = tqdm(loader, leave=False) if progress else loader
    for images, targets in iterator:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        if pruner is not None:
            pruner.apply(model)
        acc = (logits.argmax(dim=1) == targets).float().mean().item()
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))
    return loss_meter.avg, acc_meter.avg


def train(model, train_loader, val_loader, device, epochs, lr=0.05,
          momentum=0.9, weight_decay=5e-4, pruner=None, log_prefix="",
          progress=False):
    """Full SGD training with cosine LR decay. Returns history + best val acc.

    Keeps the checkpoint (state_dict) with the highest validation accuracy.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                weight_decay=weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_acc, best_state = 0.0, None
    for epoch in range(epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device,
                                          criterion, pruner, progress)
        va_loss, va_acc = evaluate(model, val_loader, device, criterion)
        scheduler.step()
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"{log_prefix}epoch {epoch + 1:3d}/{epochs} "
              f"train_acc={tr_acc:.4f} val_acc={va_acc:.4f} best={best_acc:.4f}")
    return history, best_acc, best_state
