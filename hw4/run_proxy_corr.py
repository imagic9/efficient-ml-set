"""HW4 bonus: how informative is the one-shot proxy? (rank-correlation study)

The report asks "how informative was the one-shot proxy?". We answer it with a
number, not a hunch. Take the top-K architectures the proxy ranked highest, train
each briefly *from scratch* (a short but honest signal), and measure how well the
proxy's ranking agrees with the from-scratch ranking via Kendall's tau and
Spearman's rho. High correlation => the cheap proxy is trustworthy for selection;
low correlation => the weight-sharing ranking is noisy and the search is only as
good as luck.

This is validation-only (short from-scratch runs are scored on val); no test set
is touched here.

    python run_proxy_corr.py --top-k 8 --short-epochs 15
    python run_proxy_corr.py --smoke
"""
import argparse
import os
import time

import torch
from scipy.stats import kendalltau, spearmanr

from src.data import build_loaders
from src.engine import train
from src.search_space import StandaloneNet, count_arch_params, arch_key
from src.nas_search import top_k_by_proxy
from src import plots
from src.utils import set_seed, get_device, save_json, load_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--short-epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.top_k, args.short_epochs = 3, 1

    set_seed(args.seed)
    device = get_device()
    train_loader, val_loader, _ = build_loaders(
        args.data_dir, batch_size=args.batch_size, seed=args.seed)

    search = load_json(os.path.join(args.results_dir, "search.json"))
    top = top_k_by_proxy(search["records"], args.top_k)
    print(f"scoring top-{len(top)} proxy architectures with "
          f"{args.short_epochs}-epoch from-scratch training")

    rows = []
    t0 = time.time()
    for rank, r in enumerate(top):
        arch = r["arch"]
        net = StandaloneNet(arch).to(device)
        _, best_val, _ = train(net, train_loader, val_loader, device,
                               epochs=args.short_epochs, lr=args.lr,
                               log_prefix=f"[corr {rank+1}/{len(top)}] ")
        rows.append({
            "arch": arch, "params": count_arch_params(arch),
            "proxy_val_acc": r["val_acc"], "short_val_acc": best_val,
        })
        print(f"  arch#{rank+1}  proxy={r['val_acc']:.4f}  "
              f"short-scratch={best_val:.4f}")
    print(f"proxy-correlation study done in {(time.time()-t0)/60:.1f} min")

    proxy = [x["proxy_val_acc"] for x in rows]
    real = [x["short_val_acc"] for x in rows]
    tau, tau_p = kendalltau(proxy, real)
    rho, rho_p = spearmanr(proxy, real)
    print(f"\nKendall tau = {tau:.3f} (p={tau_p:.3f})   "
          f"Spearman rho = {rho:.3f} (p={rho_p:.3f})")

    out = {
        "config": {"top_k": args.top_k, "short_epochs": args.short_epochs,
                   "lr": args.lr, "seed": args.seed},
        "rows": rows,
        "kendall_tau": tau, "kendall_p": tau_p,
        "spearman_rho": rho, "spearman_p": rho_p,
    }
    save_json(out, os.path.join(args.results_dir, "proxy_corr.json"))
    plots.plot_proxy_correlation(proxy, real, tau=tau, rho=rho).savefig(
        os.path.join(args.results_dir, "proxy_correlation.png"), dpi=130)
    print(f"saved proxy_corr.json + plot to {args.results_dir}/")


if __name__ == "__main__":
    main()
