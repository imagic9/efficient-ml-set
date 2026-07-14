"""End-to-end HW1 pipeline: baseline -> one-shot -> iterative -> sensitivity.

Methodology: the test set is measured EXACTLY ONCE per final deliverable model,
after all decisions are made. Every intermediate number, curve and method
comparison uses the validation set.

Bonus compares three per-layer allocation strategies for iterative pruning across
a sweep of sparsity levels (all on validation):
  * uniform-per-layer : same ratio in every layer (naive baseline)
  * global-magnitude  : one global threshold (strong, implicitly non-uniform)
  * sensitivity-guided: per-layer budget shaped by a sensitivity scan

    python run_all.py --data-dir ./data --out results     # full run
    python run_all.py --smoke                              # quick wiring check
"""
import argparse
import os
import time

import torch

from src.data import build_loaders
from src.model import build_vgg11_cifar, count_parameters
from src.engine import train, evaluate
from src import prune
from src import sensitivity
from src import plots
from src.utils import set_seed, get_device, save_json

METHODS = [
    ("uniform_layer", "uniform per-layer"),
    ("global", "global magnitude"),
    ("sensitivity", "sensitivity-guided"),
]


def load_state(model, state):
    model.load_state_dict(state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--baseline-epochs", type=int, default=80)
    ap.add_argument("--oneshot-epochs", type=int, default=25)
    ap.add_argument("--iter-steps", type=int, default=5)
    ap.add_argument("--iter-ft-epochs", type=int, default=12)
    ap.add_argument("--final-sparsity", type=float, default=0.80)
    ap.add_argument("--sweep", type=float, nargs="+", default=[0.80, 0.90, 0.95],
                    help="bonus: sparsity levels to compare the three methods")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.baseline_epochs = 2
        args.oneshot_epochs = 2
        args.iter_steps = 3
        args.iter_ft_epochs = 1
        args.sweep = [0.80, 0.95]

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)
    device = get_device()
    print(f"device={device}  torch={torch.__version__}")

    train_loader, val_loader, test_loader = build_loaders(
        args.data_dir, batch_size=args.batch_size)
    val_acc = lambda m: evaluate(m, val_loader, device)[1]
    test_acc = lambda m: evaluate(m, test_loader, device)[1]

    # ------------------------------------------------------------------ #
    # 1. Baseline
    # ------------------------------------------------------------------ #
    print("\n=== [1] Baseline training ===")
    model = build_vgg11_cifar().to(device)
    print(f"params: {count_parameters(model)/1e6:.2f}M")
    t0 = time.time()
    hist, best_val, best_state = train(
        model, train_loader, val_loader, device,
        epochs=args.baseline_epochs, lr=0.05, log_prefix="[base] ")
    load_state(model, best_state)
    base_test = test_acc(model)                     # test touched ONCE
    print(f"baseline val={best_val:.4f} TEST={base_test:.4f} ({time.time()-t0:.0f}s)")
    torch.save(best_state, os.path.join(args.out, "baseline.pt"))
    save_json({"history": hist, "val": best_val, "test": base_test,
               "params_M": count_parameters(model) / 1e6},
              os.path.join(args.out, "baseline.json"))
    plots.plot_history(hist, "Baseline").savefig(
        os.path.join(args.out, "baseline_history.png"), dpi=120)

    # ------------------------------------------------------------------ #
    # 2. One-shot 80% + fine-tune (global magnitude)
    # ------------------------------------------------------------------ #
    print("\n=== [2] One-shot pruning ===")
    m_os = build_vgg11_cifar().to(device)
    load_state(m_os, best_state)
    pr_os = prune.FineGrainedPruner.from_global(m_os, args.final_sparsity)
    pr_os.apply(m_os)
    os_val_before = val_acc(m_os)
    print(f"one-shot sparsity={prune.model_sparsity(m_os):.3f} val_before_ft={os_val_before:.4f}")
    _, os_val, os_state = train(m_os, train_loader, val_loader, device,
                                epochs=args.oneshot_epochs, lr=0.01,
                                pruner=pr_os, log_prefix="[1shot] ")
    load_state(m_os, os_state)
    pr_os.apply(m_os)
    os_test = test_acc(m_os)                         # test ONCE
    print(f"one-shot val={os_val:.4f} TEST={os_test:.4f}")
    save_json({"sparsity": prune.model_sparsity(m_os), "val_before_ft": os_val_before,
               "val": os_val, "test": os_test},
              os.path.join(args.out, "oneshot.json"))

    # ------------------------------------------------------------------ #
    # 3. Sensitivity scan (used by the sensitivity-guided method)
    # ------------------------------------------------------------------ #
    print("\n=== [3] Sensitivity scan ===")
    m_scan = build_vgg11_cifar().to(device)
    load_state(m_scan, best_state)
    curves = sensitivity.scan(m_scan, val_loader, device)

    # ------------------------------------------------------------------ #
    # Reusable iterative-pruning routine (all intermediate metrics = val)
    # ------------------------------------------------------------------ #
    def run_iterative(method, target, tag, measure_test=False):
        model = build_vgg11_cifar().to(device)
        load_state(model, best_state)
        schedule = prune.geometric_schedule(target, args.iter_steps)
        final_dict = None
        if method == "sensitivity":
            final_dict, _, _ = sensitivity.select_scaled(model, curves, best_val, target)
        val_points = [(0.0, best_val)]
        val_after_cut = []
        for step, cum in enumerate(schedule, 1):
            if method == "uniform_layer":
                pruner = prune.FineGrainedPruner.from_uniform(model, cum)
            elif method == "global":
                pruner = prune.FineGrainedPruner.from_global(model, cum)
            else:
                frac = cum / target
                targets = {n: min(0.99, final_dict[n] * frac) for n in final_dict}
                pruner = prune.FineGrainedPruner.from_dict(model, targets)
            pruner.apply(model)
            sp = prune.model_sparsity(model)
            vc = val_acc(model)
            _, _, state = train(model, train_loader, val_loader, device,
                                epochs=args.iter_ft_epochs, lr=0.01, pruner=pruner,
                                log_prefix=f"[{tag} {step}] ")
            load_state(model, state)
            pruner.apply(model)
            vf = val_acc(model)
            print(f"  {tag} step {step}: sparsity={sp:.3f} val_cut={vc:.4f} val_ft={vf:.4f}")
            val_after_cut.append((sp, vc))
            val_points.append((sp, vf))
        out = {"method": method, "target": target,
               "final_sparsity": val_points[-1][0], "final_val": val_points[-1][1],
               "val_points": val_points, "val_after_cut": val_after_cut}
        if measure_test:
            out["test"] = test_acc(model)            # test ONCE for this final model
        return out

    # ------------------------------------------------------------------ #
    # 4. Bonus sweep: three methods across sparsity levels (validation)
    #    At the headline 80% level we also touch the test set once per method.
    # ------------------------------------------------------------------ #
    print("\n=== [4] Iterative pruning: three methods across sparsity ===")
    sweep = {key: [] for key, _ in METHODS}
    headline = {}          # method -> result dict at final_sparsity (with test)
    global80_curve = None  # keep for the iterative-vs-oneshot figure
    for target in args.sweep:
        for key, label in METHODS:
            is_headline = abs(target - args.final_sparsity) < 1e-6
            res = run_iterative(key, target, tag=f"{key[:4]}{int(target*100)}",
                                measure_test=is_headline)
            sweep[key].append((res["final_sparsity"], res["final_val"]))
            if is_headline:
                headline[key] = res
                if key == "global":
                    global80_curve = res
            print(f"  [{label}] target {target:.2f}: val={res['final_val']:.4f} "
                  f"@{res['final_sparsity']:.3f}"
                  + (f"  TEST={res['test']:.4f}" if is_headline else ""))

    # ------------------------------------------------------------------ #
    # Save + plots
    # ------------------------------------------------------------------ #
    save_json({"schedule": prune.geometric_schedule(args.final_sparsity, args.iter_steps),
               "val_points": global80_curve["val_points"],
               "val_after_cut": global80_curve["val_after_cut"],
               "test": headline["global"]["test"],
               "oneshot_test": os_test, "baseline_test": base_test},
              os.path.join(args.out, "iterative.json"))
    plots.plot_sparsity_vs_acc(
        {"iterative (val, after ft)": global80_curve["val_points"],
         "iterative (val, right after cut)": [(0.0, best_val)] + global80_curve["val_after_cut"]},
        "Iterative global-magnitude pruning (validation)").savefig(
        os.path.join(args.out, "iterative_vs_oneshot.png"), dpi=120)

    save_json({"curves": curves, "sweep": sweep, "sweep_targets": args.sweep,
               "headline_test": {k: headline[k]["test"] for k in headline},
               "headline_sparsity": {k: headline[k]["final_sparsity"] for k in headline}},
              os.path.join(args.out, "sensitivity.json"))
    plots.plot_sensitivity(curves, best_val).savefig(
        os.path.join(args.out, "sensitivity.png"), dpi=120)
    plots.plot_sparsity_vs_acc(
        {label: sweep[key] for key, label in METHODS},
        "Bonus: per-layer allocation strategies (validation)").savefig(
        os.path.join(args.out, "sensitivity_vs_uniform.png"), dpi=120)

    print("\nSUMMARY (test measured once per final model):")
    print(f"  baseline              : {base_test:.4f} @ 0%")
    print(f"  one-shot 80% (global) : {os_test:.4f} @ 80%")
    for key, label in METHODS:
        print(f"  iterative {label:22s}: {headline[key]['test']:.4f} "
              f"@ {headline[key]['final_sparsity']*100:.1f}%")
    print("\nDONE. Results in", args.out)


if __name__ == "__main__":
    main()
