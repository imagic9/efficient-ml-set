"""Matplotlib helpers for the HW2/HW3 runs (shared by driver + notebook)."""
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


def plot_kd_comparison(regimes, ce_acc, kd_acc, baseline_acc=None,
                       title="KD vs CE-only recovery (test)"):
    """Grouped bars: for each compressed student, CE-only vs KD final accuracy.

    regimes: list of labels; ce_acc/kd_acc: matching lists of accuracies in [0,1].
    """
    x = range(len(regimes))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar([i - w / 2 for i in x], [a * 100 for a in ce_acc], w,
                label="CE-only fine-tune", color="tab:gray")
    b2 = ax.bar([i + w / 2 for i in x], [a * 100 for a in kd_acc], w,
                label="KD fine-tune", color="tab:blue")
    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.1f}", (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha="center", va="bottom", fontsize=8)
    if baseline_acc is not None:
        ax.axhline(baseline_acc * 100, color="k", ls="--", lw=1,
                   label="fp32 teacher")
    ax.set_xticks(list(x)); ax.set_xticklabels(regimes, fontsize=9)
    ax.set_ylabel("test accuracy, %"); ax.set_title(title)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    lo = min(min(ce_acc), min(kd_acc)) * 100
    ax.set_ylim(max(0, lo - 4), (baseline_acc or max(kd_acc)) * 100 + 2)
    fig.tight_layout()
    return fig


def plot_kd_ablation(xs, ys, xlabel, baseline_val=None, ce_val=None,
                     title="KD ablation (val)"):
    """Single-sweep line plot for the temperature / alpha ablations (val accuracy)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, [y * 100 for y in ys], marker="o", color="tab:blue", label="KD (val)")
    if ce_val is not None:
        ax.axhline(ce_val * 100, color="tab:gray", ls="--", lw=1, label="CE-only (val)")
    if baseline_val is not None:
        ax.axhline(baseline_val * 100, color="k", ls=":", lw=1, label="fp32 teacher (val)")
    ax.set_xlabel(xlabel); ax.set_ylabel("val accuracy, %")
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_netaug_2x2(cells, teacher_acc=None, full_acc=None,
                    title="NetAug × KD on a tiny VGG11 (0.25×, test)"):
    """2×2 grouped bars. cells: {"ce","kd","netaug_ce","netaug_kd"} -> accuracy [0,1].
    Groups on x = {standard training, NetAug}; bars per group = {CE, KD}."""
    groups = ["standard", "NetAug"]
    ce = [cells["ce"], cells["netaug_ce"]]
    kd = [cells["kd"], cells["netaug_kd"]]
    x = range(len(groups)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 5))
    b1 = ax.bar([i - w / 2 for i in x], [a * 100 for a in ce], w, label="CE loss", color="tab:gray")
    b2 = ax.bar([i + w / 2 for i in x], [a * 100 for a in kd], w, label="KD loss", color="tab:blue")
    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.1f}", (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha="center", va="bottom", fontsize=8)
    if full_acc is not None:
        ax.axhline(full_acc * 100, color="k", ls="--", lw=1, label="full fp32 VGG11 (teacher)")
    ax.set_xticks(list(x)); ax.set_xticklabels(groups)
    ax.set_ylabel("test accuracy, %"); ax.set_title(title)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    lo = min(min(ce), min(kd)) * 100
    hi = (full_acc or max(max(ce), max(kd))) * 100
    ax.set_ylim(max(0, lo - 3), hi + 2)
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
