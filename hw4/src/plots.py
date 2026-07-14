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


def _best_vs_unique(records):
    """Running-best proxy loss as a function of #distinct architectures evaluated.

    Cache hits add no information, so the honest x-axis is the count of unique
    architectures, not raw trial number. Returns (unique_counts, running_best).
    """
    xs, ys, seen_best, last = [], [], float("inf"), None
    for r in records:
        seen_best = min(seen_best, r["val_loss"])
        u = r["n_unique_so_far"]
        if u != last:                          # one point per new unique architecture
            xs.append(u); ys.append(seen_best); last = u
        else:
            ys[-1] = seen_best                 # a cache hit can't raise the best
    return xs, ys


def plot_search_convergence(records, rand_records=None,
                            title="Search convergence: running-best proxy loss"):
    """Running-best validation loss vs. #unique architectures, TPE vs. random."""
    fig, ax = plt.subplots(figsize=(8, 5))
    tx, ty = _best_vs_unique(records)
    ax.plot(tx, ty, color="tab:blue", lw=2, marker=".", ms=5, label="TPE")
    if rand_records:
        rx, ry = _best_vs_unique(rand_records)
        ax.plot(rx, ry, color="tab:orange", lw=2, marker=".", ms=5,
                ls="--", label="random search")
    ax.set_xlabel("number of distinct architectures evaluated")
    ax.set_ylabel("running-best validation loss (one-shot proxy)")
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
                           tau_p=None, rho_p=None,
                           title="One-shot proxy vs. from-scratch accuracy"):
    """Bonus: does the proxy rank architectures the way real training does?

    Points span the whole proxy range (stratified sample), so a strong overall
    trend here is evidence of good *coarse* filtering. y and x use independent
    scales (proxy is systematically lower than real training) -- what matters is
    monotonicity, i.e. rank agreement, not the diagonal.
    """
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter([a * 100 for a in proxy_acc], [a * 100 for a in real_acc],
               s=55, color="tab:blue", zorder=3)
    ax.set_xlabel("one-shot proxy val accuracy, %")
    ax.set_ylabel("short from-scratch val accuracy, %")
    pfmt = lambda p: "p<0.001" if p < 0.001 else f"p={p:.3f}"
    sub = []
    if tau is not None:
        sub.append(f"Kendall τ = {tau:.2f}" + (f" ({pfmt(tau_p)})" if tau_p is not None else ""))
    if rho is not None:
        sub.append(f"Spearman ρ = {rho:.2f}" + (f" ({pfmt(rho_p)})" if rho_p is not None else ""))
    ax.set_title(title + ("\n" + "    ".join(sub) if sub else ""))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
