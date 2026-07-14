"""HW4 step 1-3: train the one-shot supernet, then search it (TPE vs. random).

Pipeline:
  1. Build the weight-sharing supernet and train it with Single-Path One-Shot
     (uniform random path per step) on the CIFAR-10 *train* split -- or reload a
     previously trained supernet with --load-supernet (so TPE and the random-search
     control run over the *same* shared weights).
  2. Search the discrete space with Hyperopt **TPE**, and again with **random**
     search (the control) at the same budget, scoring each candidate by its one-shot
     proxy validation loss (BN recalibrated per candidate). VALIDATION only.
  3. Save the trained supernet, both trial logs, and the plots (running-best proxy
     loss vs. #unique architectures with the random baseline overlaid; proxy accuracy
     vs. #params).

The best architecture is written to results/search.json; run_retrain.py trains it
from scratch and run_proxy_corr.py measures how trustworthy the proxy ranking was.

    python run_search.py --data-dir ./data --supernet-epochs 100 --evals 250
    python run_search.py --load-supernet results/supernet.pt --evals 250   # reuse weights
    python run_search.py --smoke     # tiny wiring check
"""
import argparse
import os
import time

import torch

from src.data import build_loaders
from src.supernet import SuperNet, spos_train
from src.nas_search import run_search
from src.search_space import count_arch_params, space_size
from src import plots
from src.utils import set_seed, get_device, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--supernet-epochs", type=int, default=100)
    ap.add_argument("--evals", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--recal-batches", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load-supernet", default=None,
                    help="path to a trained supernet .pt to reuse (skips training)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run: 1 supernet epoch, 12 evals (wiring check)")
    args = ap.parse_args()

    if args.smoke:
        args.supernet_epochs, args.evals, args.recal_batches = 1, 12, 8

    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.results_dir, exist_ok=True)
    print(f"device={device}  search space size={space_size()} architectures")

    train_loader, val_loader, _ = build_loaders(
        args.data_dir, batch_size=args.batch_size, seed=args.seed)

    # 1. train (or reload) the supernet --------------------------------------- #
    supernet = SuperNet().to(device)
    n_super = sum(p.numel() for p in supernet.parameters())
    print(f"supernet params (at max width, all ops present) = {n_super/1e6:.2f}M")
    hist = None
    if args.load_supernet and os.path.exists(args.load_supernet):
        supernet.load_state_dict(torch.load(args.load_supernet, map_location=device))
        print(f"reused supernet weights from {args.load_supernet} (no training)")
    else:
        t0 = time.time()
        hist = spos_train(supernet, train_loader, device, epochs=args.supernet_epochs,
                          lr=args.lr, seed=args.seed, log_prefix="[supernet] ")
        print(f"supernet trained in {(time.time()-t0)/60:.1f} min")
        torch.save(supernet.state_dict(), os.path.join(args.results_dir, "supernet.pt"))

    # 2. search: TPE + random control over the SAME supernet ------------------ #
    def go(algo):
        t0 = time.time()
        best_arch, records = run_search(
            supernet, train_loader, val_loader, device, max_evals=args.evals,
            seed=args.seed, recal_batches=args.recal_batches, algo=algo)
        print(f"{algo} search ({args.evals} evals) done in {(time.time()-t0)/60:.1f} min")
        return best_arch, records

    best_arch, records = go("tpe")
    _, rand_records = go("random")

    best_rec = min(records, key=lambda r: r["val_loss"])
    print(f"\nBEST (tpe): ops={'/'.join(best_arch['ops'])} width={best_arch['width']} "
          f"act={best_arch['act']} | proxy val_acc={best_rec['val_acc']:.4f} "
          f"params={count_arch_params(best_arch)/1e6:.2f}M")

    # 3. persist + plots ------------------------------------------------------- #
    out = {
        "config": {
            "supernet_epochs": args.supernet_epochs, "evals": args.evals,
            "batch_size": args.batch_size, "lr": args.lr,
            "recal_batches": args.recal_batches, "seed": args.seed,
            "space_size": space_size(),
        },
        "supernet_params": n_super,
        "supernet_history": hist,
        "best_arch": best_arch,
        "best_proxy_val_acc": best_rec["val_acc"],
        "best_params": count_arch_params(best_arch),
        "records": records,
        "random_records": rand_records,
    }
    save_json(out, os.path.join(args.results_dir, "search.json"))

    plots.plot_search_convergence(records, rand_records).savefig(
        os.path.join(args.results_dir, "search_convergence.png"), dpi=130)
    plots.plot_acc_vs_params(records, best_arch).savefig(
        os.path.join(args.results_dir, "acc_vs_params.png"), dpi=130)
    print(f"saved search.json + plots to {args.results_dir}/")


if __name__ == "__main__":
    main()
