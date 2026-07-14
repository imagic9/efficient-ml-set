"""Matplotlib helpers for the HW4 NAS run (shared by drivers + notebook)."""
import matplotlib.pyplot as plt


def plot_history(history, title="Training", has_val=True):
    """Accuracy/loss curves. Works for supernet (train-only) and standalone runs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_acc"], label="train")
    if has_val and "val_acc" in history:
        ax1.plot(epochs, history["val_acc"], label="val")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("accuracy"); ax1.set_title(f"{title}: accuracy")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, history["train_loss"], label="train")
    if has_val and "val_loss" in history:
        ax2.plot(epochs, history["val_loss"], label="val")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("loss"); ax2.set_title(f"{title}: loss")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_search_convergence(records, title="TPE search: running-best proxy loss"):
    """Running-best validation loss vs. trial number (the convergence curve)."""
    trials = [r["trial"] + 1 for r in records]
    per_trial = [r["val_loss"] for r in records]
    best = [r["best_loss_so_far"] for r in records]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(trials, per_trial, s=14, color="tab:gray", alpha=0.5, label="trial loss")
    ax.plot(trials, best, color="tab:blue", lw=2, label="running best")
    ax.set_xlabel("trial number"); ax.set_ylabel("validation loss (one-shot proxy)")
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_acc_vs_params(records, best_arch=None,
                       title="Proxy accuracy vs. parameter count"):
    """Scatter of every trial: proxy val accuracy vs. #params, coloured by width."""
    from .search_space import WIDTHS
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    for j, w in enumerate(WIDTHS):
        pts = [r for r in records if r["arch"]["width"] == w]
        if not pts:
            continue
        ax.scatter([r["params"] / 1e6 for r in pts], [r["val_acc"] * 100 for r in pts],
                   s=22, color=cmap(j / max(1, len(WIDTHS) - 1)),
                   alpha=0.7, label=f"width {w}x")
    if best_arch is not None:
        from .search_space import count_arch_params
        br = [r for r in records if r["arch"] == best_arch]
        if br:
            ax.scatter([count_arch_params(best_arch) / 1e6], [br[0]["val_acc"] * 100],
                       marker="*", s=260, color="crimson", zorder=5,
                       edgecolor="k", label="selected best")
    ax.set_xlabel("parameters, millions"); ax.set_ylabel("proxy val accuracy, %")
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_proxy_correlation(proxy_acc, real_acc, tau=None, rho=None,
                           title="One-shot proxy vs. from-scratch accuracy"):
    """Bonus: does the proxy rank architectures the way real training does?"""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter([a * 100 for a in proxy_acc], [a * 100 for a in real_acc],
               s=60, color="tab:blue", zorder=3)
    lo = min(min(proxy_acc), min(real_acc)) * 100 - 1
    hi = max(max(proxy_acc), max(real_acc)) * 100 + 1
    ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="y = x")
    ax.set_xlabel("one-shot proxy val accuracy, %")
    ax.set_ylabel("short from-scratch val accuracy, %")
    sub = []
    if tau is not None:
        sub.append(f"Kendall τ = {tau:.2f}")
    if rho is not None:
        sub.append(f"Spearman ρ = {rho:.2f}")
    ax.set_title(title + ("\n" + "   ".join(sub) if sub else ""))
    ax.legend(); ax.grid(alpha=0.3); ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig
