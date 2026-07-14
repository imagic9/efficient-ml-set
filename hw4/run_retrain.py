"""HW4 step 4: retrain the best-found architecture from scratch and compare.

Weight-shared proxy scores are only a ranking signal; the deliverable number comes
from training the selected design as an ordinary network, from scratch, under a
normal recipe. We compare it on two axes -- test accuracy and parameter count --
against two baselines:

  * VGG11 baseline (the frozen HW1 fp32 model, ~9.2M params, ~90.7% test) -- the
    same baseline used across HW1-HW3;
  * an in-space "default" design (all conv3x3, width 1.0, ReLU) trained with the
    identical recipe -- isolates "did the search help vs. a sensible default in the
    same space", not just "vs. a different architecture family".

Methodology: model selection on the VALIDATION split; the TEST set is evaluated
exactly once per final model (best design, default design, and the baseline).

    python run_retrain.py --baseline ../hw1/results/baseline.pt --epochs 90
    python run_retrain.py --smoke
"""
import argparse
import os
import time

import torch

from src.data import build_loaders
from src.engine import train, evaluate
from src.model import build_vgg11_cifar, count_parameters
from src.search_space import StandaloneNet, count_arch_params
from src import plots
from src.utils import set_seed, get_device, save_json, load_json


DEFAULT_ARCH = {"ops": ["conv3x3"] * 4, "width": 1.0, "act": "relu"}


def train_from_scratch(arch, loaders, device, epochs, lr, tag):
    """Train a StandaloneNet(arch) from scratch; return (test_acc, val_acc, params, history)."""
    train_loader, val_loader, test_loader = loaders
    net = StandaloneNet(arch).to(device)
    params = count_parameters(net)
    print(f"[{tag}] params={params/1e6:.2f}M  ops={'/'.join(arch['ops'])} "
          f"width={arch['width']} act={arch['act']}")
    hist, best_val, best_state = train(
        net, train_loader, val_loader, device, epochs=epochs, lr=lr,
        log_prefix=f"[{tag}] ")
    net.load_state_dict(best_state)
    _, test_acc = evaluate(net, test_loader, device)          # TEST touched once
    print(f"[{tag}] best_val={best_val:.4f}  TEST={test_acc:.4f}")
    return {"arch": arch, "params": params, "val_acc": best_val,
            "test_acc": test_acc, "history": hist}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--baseline", default="../hw1/results/baseline.pt")
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.epochs = 2

    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.results_dir, exist_ok=True)
    loaders = build_loaders(args.data_dir, batch_size=args.batch_size, seed=args.seed)
    _, _, test_loader = loaders

    search = load_json(os.path.join(args.results_dir, "search.json"))
    best_arch = search["best_arch"]

    # VGG11 baseline: load frozen HW1 model, evaluate TEST once for the reference.
    baseline = None
    if os.path.exists(args.baseline):
        vgg = build_vgg11_cifar().to(device)
        vgg.load_state_dict(torch.load(args.baseline, map_location=device))
        _, base_test = evaluate(vgg, test_loader, device)
        baseline = {"params": count_parameters(vgg), "test_acc": base_test}
        print(f"[baseline VGG11] params={baseline['params']/1e6:.2f}M "
              f"TEST={base_test:.4f}")
    else:
        print(f"[baseline VGG11] {args.baseline} not found -- skipping baseline eval")

    t0 = time.time()
    best = train_from_scratch(best_arch, loaders, device, args.epochs, args.lr, "best")
    default = train_from_scratch(DEFAULT_ARCH, loaders, device, args.epochs, args.lr,
                                 "default")
    print(f"retrain done in {(time.time()-t0)/60:.1f} min")

    out = {
        "config": {"epochs": args.epochs, "lr": args.lr,
                   "batch_size": args.batch_size, "seed": args.seed},
        "baseline_vgg11": baseline,
        "best": best,
        "default": default,
        "proxy_val_acc_of_best": search.get("best_proxy_val_acc"),
    }
    save_json(out, os.path.join(args.results_dir, "retrain.json"))

    plots.plot_history(best["history"], title="Best design (from scratch)").savefig(
        os.path.join(args.results_dir, "retrain_best_history.png"), dpi=130)
    print(f"saved retrain.json + history plot to {args.results_dir}/")

    if baseline:
        dp = best["params"] / baseline["params"]
        print(f"\nSUMMARY  best test={best['test_acc']:.4f} @ {best['params']/1e6:.2f}M "
              f"({dp:.2f}x baseline params)  vs baseline {baseline['test_acc']:.4f} "
              f"@ {baseline['params']/1e6:.2f}M  vs default {default['test_acc']:.4f}")


if __name__ == "__main__":
    main()
