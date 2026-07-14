"""Confirm the high-sparsity bonus result on the TEST set.

The sweep in run_all.py compares methods on validation. Here we take the single
most interesting operating point (high sparsity, where sensitivity-guided
overtakes the naive uniform baseline) and measure the test set once per final
model. Reuses the trained baseline and the saved sensitivity curves.

    python run_confirm.py --data-dir ./data --out results --target 0.95
"""
import argparse
import json
import os

import torch

from src.data import build_loaders
from src.model import build_vgg11_cifar
from src.engine import train, evaluate
from src import prune, sensitivity
from src.utils import set_seed, get_device, save_json

METHODS = ["uniform_layer", "global", "sensitivity"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--iter-steps", type=int, default=5)
    ap.add_argument("--iter-ft-epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    train_loader, val_loader, test_loader = build_loaders(args.data_dir)

    best_state = torch.load(os.path.join(args.out, "baseline.pt"), map_location=device)
    best_val = json.load(open(os.path.join(args.out, "baseline.json")))["val"]
    curves = json.load(open(os.path.join(args.out, "sensitivity.json")))["curves"]

    def run(method):
        model = build_vgg11_cifar().to(device)
        model.load_state_dict(best_state)
        schedule = prune.geometric_schedule(args.target, args.iter_steps)
        final_dict = None
        if method == "sensitivity":
            final_dict, _, _ = sensitivity.select_scaled(model, curves, best_val, args.target)
        for cum in schedule:
            if method == "uniform_layer":
                pr = prune.FineGrainedPruner.from_uniform(model, cum)
            elif method == "global":
                pr = prune.FineGrainedPruner.from_global(model, cum)
            else:
                frac = cum / args.target
                pr = prune.FineGrainedPruner.from_dict(
                    model, {n: min(0.99, final_dict[n] * frac) for n in final_dict})
            pr.apply(model)
            _, _, state = train(model, train_loader, val_loader, device,
                                epochs=args.iter_ft_epochs, lr=0.01, pruner=pr,
                                log_prefix=f"[{method[:4]}{int(args.target*100)}] ")
            model.load_state_dict(state)
            pr.apply(model)
        sp = prune.model_sparsity(model)
        val = evaluate(model, val_loader, device)[1]
        test = evaluate(model, test_loader, device)[1]   # test ONCE per final model
        print(f"{method:14s} @ {sp*100:.1f}%: val={val:.4f} TEST={test:.4f}")
        return {"sparsity": sp, "val": val, "test": test}

    result = {m: run(m) for m in METHODS}
    save_json({"target": args.target, "results": result},
              os.path.join(args.out, f"confirm{int(args.target*100)}.json"))
    print("\nDONE. Confirmation saved.")


if __name__ == "__main__":
    main()
