"""HW4 bonus: how informative is the one-shot proxy? (rank-correlation study)

The report asks "how informative was the one-shot proxy?". We answer it with two
numbers, on a sample that spans the *whole* proxy range (not just the top):

  * a **stratified** sample across proxy quantiles (low / mid / high) -- to test
    whether the proxy is good at *coarse* filtering (does it rank clearly-bad and
    clearly-good designs correctly?);
  * the **top slice** of that sample -- to test whether it is good at *fine* ranking
    (can it pick the winner among the near-best designs?).

Each sampled architecture is trained briefly from scratch over a few seeds (averaged,
to damp init/data noise), and we correlate the proxy ranking with the from-scratch
ranking via Kendall's tau and Spearman's rho -- reporting the p-value so a null result
is not oversold. Validation-only; no test set is touched.

    python run_proxy_corr.py --n-bins 4 --per-bin 4 --top-k 6 --seeds 42,43 --short-epochs 15
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
from src.nas_search import stratified_by_proxy, top_k_by_proxy
from src import plots
from src.utils import set_seed, get_device, save_json, load_json


def short_train_mean(arch, data_dir, batch_size, device, epochs, lr, seeds, tag):
    """Short from-scratch training averaged over seeds; returns mean val accuracy."""
    accs = []
    for s in seeds:
        set_seed(s)
        train_loader, val_loader, _ = build_loaders(
            data_dir, batch_size=batch_size, seed=42, shuffle_seed=s)
        net = StandaloneNet(arch).to(device)
        _, best_val, _ = train(net, train_loader, val_loader, device,
                               epochs=epochs, lr=lr, log_prefix=f"[{tag} s{s}] ")
        accs.append(best_val)
    return sum(accs) / len(accs), accs


def corr(proxy, real):
    tau, tau_p = kendalltau(proxy, real)
    rho, rho_p = spearmanr(proxy, real)
    return {"kendall_tau": tau, "kendall_p": tau_p,
            "spearman_rho": rho, "spearman_p": rho_p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--n-bins", type=int, default=4)
    ap.add_argument("--per-bin", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=6,
                    help="size of the top slice for the fine-ranking correlation")
    ap.add_argument("--short-epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seeds", default="42,43")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    if args.smoke:
        args.n_bins, args.per_bin, args.top_k, args.short_epochs = 2, 2, 2, 1
        seeds = [42]

    device = get_device()
    search = load_json(os.path.join(args.results_dir, "search.json"))
    records = search["records"]

    # stratified sample across the proxy range, unioned with the very top slice
    strat = stratified_by_proxy(records, n_bins=args.n_bins, per_bin=args.per_bin)
    top = top_k_by_proxy(records, args.top_k)
    sample, seen = [], set()
    for r in strat + top:
        k = arch_key(r["arch"])
        if k not in seen:
            seen.add(k); sample.append(r)
    sample.sort(key=lambda r: r["val_acc"])
    print(f"stratified sample: {len(sample)} distinct archs across proxy range "
          f"[{sample[0]['val_acc']:.3f}..{sample[-1]['val_acc']:.3f}], "
          f"{args.short_epochs}-epoch from-scratch x {len(seeds)} seeds")

    rows = []
    t0 = time.time()
    for i, r in enumerate(sample):
        arch = r["arch"]
        mean_val, per_seed = short_train_mean(
            arch, args.data_dir, args.batch_size, device, args.short_epochs,
            args.lr, seeds, f"corr {i+1}/{len(sample)}")
        rows.append({"arch": arch, "params": count_arch_params(arch),
                     "proxy_val_acc": r["val_acc"], "short_val_acc": mean_val,
                     "short_per_seed": per_seed})
        print(f"  arch#{i+1}  proxy={r['val_acc']:.4f}  short-scratch={mean_val:.4f}")
    print(f"proxy-correlation study done in {(time.time()-t0)/60:.1f} min")

    proxy = [x["proxy_val_acc"] for x in rows]
    real = [x["short_val_acc"] for x in rows]
    full = corr(proxy, real)                         # coarse filtering, whole range
    tk = sorted(rows, key=lambda x: x["proxy_val_acc"], reverse=True)[:args.top_k]
    top_corr = corr([x["proxy_val_acc"] for x in tk], [x["short_val_acc"] for x in tk])

    print(f"\nFULL (n={len(rows)}, coarse):  Kendall tau={full['kendall_tau']:.3f} "
          f"(p={full['kendall_p']:.3f})  Spearman rho={full['spearman_rho']:.3f} "
          f"(p={full['spearman_p']:.3f})")
    print(f"TOP-{args.top_k} (fine):        Kendall tau={top_corr['kendall_tau']:.3f} "
          f"(p={top_corr['kendall_p']:.3f})  Spearman rho={top_corr['spearman_rho']:.3f} "
          f"(p={top_corr['spearman_p']:.3f})")

    out = {
        "config": {"n_bins": args.n_bins, "per_bin": args.per_bin, "top_k": args.top_k,
                   "short_epochs": args.short_epochs, "seeds": seeds},
        "rows": rows,
        "full": full, "top": top_corr, "top_k": args.top_k,
    }
    save_json(out, os.path.join(args.results_dir, "proxy_corr.json"))
    plots.plot_proxy_correlation(
        proxy, real, tau=full["kendall_tau"], rho=full["spearman_rho"],
        tau_p=full["kendall_p"], rho_p=full["spearman_p"]).savefig(
        os.path.join(args.results_dir, "proxy_correlation.png"), dpi=130)
    print(f"saved proxy_corr.json + plot to {args.results_dir}/")


if __name__ == "__main__":
    main()
