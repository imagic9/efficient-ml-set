"""HW4 step 4: retrain the best-found architecture from scratch and compare.

Weight-shared proxy scores are only a ranking signal; the deliverable number comes
from training the selected design as an ordinary network, from scratch, under a
normal recipe. We compare it on two axes -- test accuracy and parameter count --
against two baselines:

  * VGG11 baseline (the frozen HW1 fp32 model, ~9.5M params, ~90.7% test) -- the
    same baseline used across HW1-HW3;
  * an in-space "default" design (all conv3x3, width 1.0, ReLU) trained with the
    identical recipe -- isolates "did the search help vs. a sensible default in the
    same space", not just "vs. a different architecture family".

The best vs. default gap is small, so a single run would be dominated by init/data
noise. We therefore train each design over **several seeds** and report mean +/- std.
Both designs share the same seed set, and the train-batch order is driven by a
per-seed Generator (independent of how much RNG model construction consumes), so at
each seed the two designs see the *same* data order -- a controlled comparison.

Methodology: model selection on the VALIDATION split; the TEST set is evaluated
once per final (design, seed) model; the train/val split is fixed across all seeds.

    python run_retrain.py --baseline ../hw1/results/baseline.pt --epochs 120 --seeds 42,43,44
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


def _mean_std(xs):
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)    # sample std (n-1, unbiased)
    return m, v ** 0.5


def train_one(arch, data_dir, batch_size, device, epochs, lr, seed, tag):
    """Train a StandaloneNet(arch) from scratch at one seed. TEST touched once."""
    set_seed(seed)                                       # controls model init
    # per-seed deterministic batch order, independent of init-RNG consumption
    train_loader, val_loader, test_loader = build_loaders(
        data_dir, batch_size=batch_size, seed=42, shuffle_seed=seed)
    net = StandaloneNet(arch).to(device)
    hist, best_val, best_state = train(net, train_loader, val_loader, device,
                                       epochs=epochs, lr=lr, log_prefix=f"[{tag} s{seed}] ")
    net.load_state_dict(best_state)
    _, test_acc = evaluate(net, test_loader, device)
    print(f"[{tag} s{seed}] best_val={best_val:.4f}  TEST={test_acc:.4f}")
    return {"seed": seed, "val_acc": best_val, "test_acc": test_acc, "history": hist}


def train_multi_seed(arch, data_dir, batch_size, device, epochs, lr, seeds, tag):
    params = count_arch_params(arch)
    print(f"[{tag}] params={params/1e6:.2f}M  ops={'/'.join(arch['ops'])} "
          f"width={arch['width']} act={arch['act']}  seeds={seeds}")
    runs = [train_one(arch, data_dir, batch_size, device, epochs, lr, s, tag)
            for s in seeds]
    test_m, test_s = _mean_std([r["test_acc"] for r in runs])
    val_m, val_s = _mean_std([r["val_acc"] for r in runs])
    print(f"[{tag}] TEST {test_m*100:.2f}+/-{test_s*100:.2f}%  "
          f"VAL {val_m*100:.2f}+/-{val_s*100:.2f}%")
    return {"arch": arch, "params": params, "runs": runs,
            "test_mean": test_m, "test_std": test_s,
            "val_mean": val_m, "val_std": val_s,
            "history": runs[0]["history"]}          # a representative curve for plotting


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--baseline", default="../hw1/results/baseline.pt")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seeds", default="42,43,44",
                    help="comma-separated seeds for mean+/-std")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    if args.smoke:
        args.epochs, seeds = 2, [42, 43]

    device = get_device()
    os.makedirs(args.results_dir, exist_ok=True)

    search = load_json(os.path.join(args.results_dir, "search.json"))
    best_arch = search["best_arch"]

    # VGG11 baseline: load frozen HW1 model, evaluate TEST once for the reference.
    baseline = None
    if os.path.exists(args.baseline):
        _, _, test_loader = build_loaders(args.data_dir, batch_size=args.batch_size, seed=42)
        vgg = build_vgg11_cifar().to(device)
        vgg.load_state_dict(torch.load(args.baseline, map_location=device))
        _, base_test = evaluate(vgg, test_loader, device)
        baseline = {"params": count_parameters(vgg), "test_acc": base_test}
        print(f"[baseline VGG11] params={baseline['params']/1e6:.2f}M TEST={base_test:.4f}")
    else:
        print(f"[baseline VGG11] {args.baseline} not found -- skipping baseline eval")

    t0 = time.time()
    best = train_multi_seed(best_arch, args.data_dir, args.batch_size, device,
                            args.epochs, args.lr, seeds, "best")
    default = train_multi_seed(DEFAULT_ARCH, args.data_dir, args.batch_size, device,
                               args.epochs, args.lr, seeds, "default")
    print(f"retrain done in {(time.time()-t0)/60:.1f} min")

    out = {
        "config": {"epochs": args.epochs, "lr": args.lr,
                   "batch_size": args.batch_size, "seeds": seeds},
        "baseline_vgg11": baseline,
        "best": best,
        "default": default,
        "proxy_val_acc_of_best": search.get("best_proxy_val_acc"),
    }
    save_json(out, os.path.join(args.results_dir, "retrain.json"))

    plots.plot_history(best["history"],
                       title="Best design (from scratch, seed %d)" % seeds[0]).savefig(
        os.path.join(args.results_dir, "retrain_best_history.png"), dpi=130)
    print(f"saved retrain.json + history plot to {args.results_dir}/")

    if baseline:
        dp = best["params"] / baseline["params"]
        gap = best["test_mean"] - default["test_mean"]
        print(f"\nSUMMARY  best {best['test_mean']*100:.2f}+/-{best['test_std']*100:.2f}% "
              f"@ {best['params']/1e6:.2f}M ({dp:.2f}x baseline params)  vs baseline "
              f"{baseline['test_acc']*100:.2f}% @ {baseline['params']/1e6:.2f}M  vs default "
              f"{default['test_mean']*100:.2f}+/-{default['test_std']*100:.2f}% "
              f"(best-default = {gap*100:+.2f} pp)")


if __name__ == "__main__":
    main()
