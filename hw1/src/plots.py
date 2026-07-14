"""Matplotlib helpers used by both the driver script and the notebook."""
import matplotlib.pyplot as plt


def plot_history(history, title="Training"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history["train_acc"]) + 1)
    ax1.plot(epochs, history["train_acc"], label="train")
    ax1.plot(epochs, history["val_acc"], label="val")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("accuracy"); ax1.set_title(f"{title}: accuracy")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, history["train_loss"], label="train")
    ax2.plot(epochs, history["val_loss"], label="val")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("loss"); ax2.set_title(f"{title}: loss")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_sparsity_vs_acc(series, title="Accuracy vs sparsity"):
    """series: dict label -> list of (sparsity, accuracy)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, points in series.items():
        xs = [p[0] * 100 for p in points]
        ys = [p[1] * 100 for p in points]
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_xlabel("sparsity, %"); ax.set_ylabel("accuracy, %")
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_sensitivity(curves, baseline_acc=None):
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, curve in curves.items():
        xs = [r * 100 for r, _ in curve]
        ys = [a * 100 for _, a in curve]
        ax.plot(xs, ys, marker=".", label=name)
    if baseline_acc is not None:
        ax.axhline(baseline_acc * 100, color="k", ls="--", lw=1, label="baseline")
    ax.set_xlabel("layer sparsity, %"); ax.set_ylabel("val accuracy, %")
    ax.set_title("Per-layer sensitivity (prune one layer, no fine-tune)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
