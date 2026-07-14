"""Bonus part 2: structured (channel) pruning, uniform vs sensitivity-guided.

Reuses the trained baseline checkpoint (results/baseline.pt), so it is fast. All
sweep comparisons are on validation; the test set is touched once per final
headline model. Produces results/structured.json and figures.

    python run_structured.py --data-dir ./data --out results
    python run_structured.py --smoke
"""
import argparse
import os

import torch

from src.data import build_loaders
from src.model import build_vgg11_cifar
from src.engine import evaluate
from src import structured
from src import plots
from src.utils import set_seed, get_device, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--state", default=None, help="baseline checkpoint (default: <out>/baseline.pt)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ft-epochs", type=int, default=12)
    ap.add_argument("--scales", type=float, nargs="+", default=[0.2, 0.35, 0.5, 0.65])
    ap.add_argument("--methods", nargs="+", default=["uniform", "sensitivity"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.ft_epochs = 1
        args.scales = [0.3, 0.6]

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)
    device = get_device()
    state_path = args.state or os.path.join(args.out, "baseline.pt")
    state = torch.load(state_path, map_location=device)
    print(f"device={device}  baseline={state_path}")

    train_loader, val_loader, test_loader = build_loaders(args.data_dir)
    build = lambda: build_vgg11_cifar()

    base = build().to(device)
    base.load_state_dict(state)
    base_macs, base_params = structured.macs_params(base, device)
    base_val = evaluate(base, val_loader, device)[1]
    print(f"baseline: MACs={base_macs/1e6:.1f}M params={base_params/1e6:.2f}M val={base_val:.4f}")

    # --- channel sensitivity scan + per-layer robustness profile ----------
    print("=== channel sensitivity scan ===")
    ratios = [0.2, 0.4, 0.6] if args.smoke else None
    curves = structured.channel_sensitivity(build, state, val_loader, device, ratios)
    profile = structured.profile_from_curves(curves, base_val, tol=0.05)

    # --- sweep methods x scales (validation) ------------------------------
    print("=== sweep: uniform vs sensitivity channel pruning ===")
    sweep = {m: [] for m in args.methods}     # method -> [(macs_frac, val), ...]
    keep_models = {m: [] for m in args.methods}
    for scale in args.scales:
        for method in args.methods:
            r = structured.prune_and_finetune(
                build, state, method, scale, profile, device,
                train_loader, val_loader, args.ft_epochs,
                tag=f"{method[:4]}{int(scale*100)}")
            frac = r["macs"] / base_macs
            sweep[method].append((frac, r["val"]))
            keep_models[method].append((frac, r))
            print(f"  {method:11s} scale={scale:.2f}: MACs={frac*100:.0f}% "
                  f"params={r['params']/1e6:.2f}M val={r['val']:.4f}")

    # --- headline: for each method, test the point nearest ~50% MACs ------
    headline = {}
    for method in args.methods:
        target = 0.5
        best = min(keep_models[method], key=lambda fr: abs(fr[0] - target))
        frac, r = best
        test = evaluate(r["model"], test_loader, device)[1]   # test ONCE
        headline[method] = {"macs_frac": frac, "params_M": r["params"] / 1e6,
                            "val": r["val"], "test": test}
        print(f"HEADLINE {method}: MACs={frac*100:.0f}% val={r['val']:.4f} TEST={test:.4f}")

    save_json({"base_macs_M": base_macs / 1e6, "base_params_M": base_params / 1e6,
               "base_val": base_val, "curves": curves, "profile": profile,
               "sweep": sweep, "headline": headline, "scales": args.scales},
              os.path.join(args.out, "structured.json"))
    plots.plot_sensitivity(curves, base_val).savefig(
        os.path.join(args.out, "structured_sensitivity.png"), dpi=120)

    # accuracy vs remaining MACs (x smaller = more pruned)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for method in args.methods:
        pts = sorted(sweep[method])
        ax.plot([p[0] * 100 for p in pts], [p[1] * 100 for p in pts],
                marker="o", label=method)
    ax.axhline(base_val * 100, color="k", ls="--", lw=1, label="baseline (dense)")
    ax.set_xlabel("remaining MACs, % of dense"); ax.set_ylabel("validation accuracy, %")
    ax.set_title("Structured pruning: uniform vs sensitivity-guided")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(args.out, "structured_macs_vs_acc.png"), dpi=120)

    print("\nDONE. Structured results in", args.out)


if __name__ == "__main__":
    main()
