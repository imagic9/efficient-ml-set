"""HW2 ablations that back up two design choices in the QAT recipe.

  1) Gradient pooling: MEAN (what we use) vs SUM (the literal slide-36 diagram).
     Shows whether the sum form -- whose per-centroid step scales with cluster
     size -- is actually unstable in our setup.

  2) What trains during QAT: centroids only (adapt_extras=False) vs centroids +
     the small fp32 biases/BatchNorm affine params (adapt_extras=True, default).
     Quantifies how much the "extras" actually contribute.

Both use the same protocol as the main runs (val for selection, test once each).

    python run_ablation.py --baseline ../hw1/results/baseline.pt --data-dir ./data
"""
import argparse
import os

import torch

from src.data import build_loaders
from src.model import build_vgg11_cifar
from src.engine import evaluate
from src.kmeans_quant import KMeansQuantizer, quantizable_layers
from src import qat
from src.utils import set_seed, get_device, save_json


def uniform_bits(model, b):
    return {name: b for name, _ in quantizable_layers(model)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="../hw1/results/baseline.pt")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--qat-epochs", type=int, default=10)
    ap.add_argument("--qat-lr", type=float, default=1e-3)
    ap.add_argument("--kmeans-iters", type=int, default=30)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.qat_epochs = 2
        args.kmeans_iters = 8

    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    print(f"device={device}  torch={torch.__version__}")

    train_loader, val_loader, test_loader = build_loaders(
        args.data_dir, batch_size=args.batch_size)
    test_acc = lambda m: evaluate(m, test_loader, device)[1]
    base_state = torch.load(args.baseline, map_location=device)

    def run(bits, pool_mode, adapt_extras, tag, optim_name="adam"):
        set_seed(args.seed)                          # same init/order for every cell
        model = build_vgg11_cifar().to(device)
        model.load_state_dict(base_state)
        q = KMeansQuantizer.quantize(model, uniform_bits(model, bits),
                                     iters=args.kmeans_iters)
        q.to(device)
        hist, qat_val, best = qat.qat_finetune(
            model, q, train_loader, val_loader, device, epochs=args.qat_epochs,
            lr=args.qat_lr, adapt_extras=adapt_extras, pool_mode=pool_mode,
            optim_name=optim_name, log_prefix=f"[{tag}] ")
        qat.restore_best(model, q, best)
        q.reconstruct(model)
        t = test_acc(model)
        # did training stay stable? (val never collapsed to near-chance)
        stable = min(hist["val_acc"]) > 0.2
        print(f"  {tag}: QAT val={qat_val:.4f} TEST={t:.4f} "
              f"{'stable' if stable else 'DIVERGED'}")
        return {"qat_val": qat_val, "qat_test": t, "history": hist, "stable": stable}

    out = {}

    # 1a) pooling under Adam (what we use): mean vs sum -------------------- #
    print("\n=== pooling ablation (2-bit, Adam): mean vs sum ===")
    out["pooling_2bit_adam"] = {
        "mean": run(2, "mean", True, "adam pool=mean"),
        "sum":  run(2, "sum",  True, "adam pool=sum"),
    }

    # 1b) pooling under plain SGD: this is where sum's cluster-size scaling
    #     bites -- sum should diverge while mean stays stable --------------- #
    print("\n=== pooling ablation (2-bit, SGD): mean vs sum ===")
    out["pooling_2bit_sgd"] = {
        "mean": run(2, "mean", True, "sgd pool=mean", optim_name="sgd"),
        "sum":  run(2, "sum",  True, "sgd pool=sum",  optim_name="sgd"),
    }

    # 2) what trains: centroids only vs + biases/BN ----------------------- #
    for b in (2, 3):
        print(f"\n=== extras ablation ({b}-bit): centroids-only vs +bias/BN ===")
        out[f"extras_{b}bit"] = {
            "adapt":  run(b, "mean", True,  f"{b}b adapt"),
            "frozen": run(b, "mean", False, f"{b}b frozen"),
        }

    save_json(out, os.path.join(args.out, "ablation.json"))

    print("\nSUMMARY")
    for opt in ("adam", "sgd"):
        p = out[f"pooling_2bit_{opt}"]
        print(f"  pooling 2-bit [{opt}]: mean TEST={p['mean']['qat_test']:.4f} "
              f"({'stable' if p['mean']['stable'] else 'DIVERGED'}) | "
              f"sum TEST={p['sum']['qat_test']:.4f} "
              f"({'stable' if p['sum']['stable'] else 'DIVERGED'})")
    for b in (2, 3):
        e = out[f"extras_{b}bit"]
        d = (e["adapt"]["qat_test"] - e["frozen"]["qat_test"]) * 100
        print(f"  extras {b}-bit: +bias/BN TEST={e['adapt']['qat_test']:.4f}  "
              f"centroids-only TEST={e['frozen']['qat_test']:.4f}  (extras add {d:+.2f} pp)")
    print("\nDONE. Results in", args.out)


if __name__ == "__main__":
    main()
