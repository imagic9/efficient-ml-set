"""Matplotlib helpers for the HW2 quantization runs (shared by driver + notebook)."""
import matplotlib.pyplot as plt


def plot_history(history, title="QAT fine-tuning"):
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


def plot_bits_vs_acc(series, baseline_acc=None, title="Accuracy vs bit-width"):
    """series: dict label -> list of (bits, accuracy). Accuracy in [0,1]."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, points in series.items():
        xs = [p[0] for p in points]
        ys = [p[1] * 100 for p in points]
        ax.plot(xs, ys, marker="o", label=label)
    if baseline_acc is not None:
        ax.axhline(baseline_acc * 100, color="k", ls="--", lw=1, label="fp32 baseline")
    ax.set_xlabel("bits per weight"); ax.set_ylabel("accuracy, %")
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_size_vs_acc(series, baseline=None, title="Accuracy vs model size"):
    """series: dict label -> list of (size_MB, accuracy). Pareto view."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, points in series.items():
        xs = [p[0] for p in points]
        ys = [p[1] * 100 for p in points]
        ax.plot(xs, ys, marker="o", ls="", label=label)
    if baseline is not None:
        ax.scatter([baseline[0]], [baseline[1] * 100], color="k", marker="*",
                   s=160, label="fp32 baseline", zorder=5)
    ax.set_xlabel("model size, MB"); ax.set_ylabel("accuracy, %")
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_bit_sensitivity(curves, baseline_acc=None):
    """curves: dict layer -> list of (bits, val_acc) from the bit-sensitivity scan."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, curve in curves.items():
        xs = [b for b, _ in curve]
        ys = [a * 100 for _, a in curve]
        ax.plot(xs, ys, marker=".", label=name)
    if baseline_acc is not None:
        ax.axhline(baseline_acc * 100, color="k", ls="--", lw=1, label="baseline")
    ax.set_xlabel("bits for this layer (rest fp32)"); ax.set_ylabel("val accuracy, %")
    ax.set_title("Per-layer bit-width sensitivity (quantize one layer, no fine-tune)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_weight_hist(weights, centroids, title="Weight distribution + centroids"):
    """Show a layer's weight histogram with the learned centroids overlaid."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(weights, bins=120, color="tab:blue", alpha=0.6)
    for c in centroids:
        ax.axvline(c, color="tab:red", lw=0.8)
    ax.set_xlabel("weight value"); ax.set_ylabel("count")
    ax.set_title(title); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
