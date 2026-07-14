"""Multi-seed error bars for the mixed-vs-uniform Pareto claim.

Single-run accuracies wobble by a couple tenths of a point between seeds (data
shuffling + dropout during QAT), so a ~1 pp gap deserves a mean +/- std over a few
seeds rather than one number. This re-runs the uniform Pareto references and the
mixed-precision model over several seeds, both dense and on the pruned model, and
reports mean +/- std. The k-means codebook itself is seed-independent (linear init),
so only the QAT fine-tuning varies.

    python run_seeds.py --seeds 0 1 2 --baseline ../hw1/results/baseline.pt \
        --pruned ../hw1/results/iterative_final.pt --data-dir ./data
"""
import argparse
import os

import torch

from src.data import build_loaders
from src.model import build_vgg11_cifar
from src.engine import evaluate
from src.kmeans_quant import (KMeansQuantizer, quantizable_layers,
                              model_size_bits, sparse_quant_size_bits)
from src import qat, mixed
from src.prune import model_sparsity
from src.utils import set_seed, get_device, save_json


def uniform_bits(model, b):
    return {name: b for name, _ in quantizable_layers(model)}


def mean_std(xs):
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / n
    return m, var ** 0.5


def pareto_interp(points, x):
    pts = sorted(points)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0) if x1 > x0 else 0.0
            return y0 + t * (y1 - y0)
    return pts[0][1] if x < pts[0][0] else pts[-1][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="../hw1/results/baseline.pt")
    ap.add_argument("--pruned", default="../hw1/results/iterative_final.pt")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--avg-bits", type=float, default=2.5)
    ap.add_argument("--uniform-ref", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--qat-epochs", type=int, default=10)
    ap.add_argument("--qat-lr", type=float, default=1e-3)
    ap.add_argument("--kmeans-iters", type=int, default=30)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.seeds = [0, 1]
        args.qat_epochs = 1
        args.kmeans_iters = 8

    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    print(f"device={device}  torch={torch.__version__}  seeds={args.seeds}")

    train_loader, val_loader, test_loader = build_loaders(
        args.data_dir, batch_size=args.batch_size)
    test_acc = lambda m: evaluate(m, test_loader, device)[1]

    base_state = torch.load(args.baseline, map_location=device)
    base = build_vgg11_cifar().to(device)
    base.load_state_dict(base_state)
    fp32_MB = sum(p.numel() for p in base.parameters()) * 32 / 8 / 1e6

    # allocation policy computed once (deterministic sensitivity scan) --------
    dcurves = mixed.bit_sensitivity_scan(build_vgg11_cifar, base_state, device,
                                         val_loader, bit_levels=(2, 3, 4),
                                         kmeans_iters=max(8, args.kmeans_iters // 2))
    dense_mixed_bits, dense_avg = mixed.allocate_mixed_bits(base, dcurves, args.avg_bits)

    pruned_state = torch.load(args.pruned, map_location=device)
    m_p = build_vgg11_cifar().to(device)
    m_p.load_state_dict(pruned_state)
    sparsity = model_sparsity(m_p)
    pruned_test = test_acc(m_p)
    pcurves = mixed.bit_sensitivity_scan(build_vgg11_cifar, pruned_state, device,
                                         val_loader, bit_levels=(2, 3, 4),
                                         kmeans_iters=max(8, args.kmeans_iters // 2),
                                         keep_pruned_zeros=True)
    prune_mixed_bits, prune_avg = mixed.allocate_mixed_bits(m_p, pcurves, args.avg_bits)

    def qat_test(bits_dict, state, seed, keep_zeros=False):
        set_seed(seed)
        model = build_vgg11_cifar().to(device)
        model.load_state_dict(state)
        q = KMeansQuantizer.quantize(model, bits_dict, iters=args.kmeans_iters,
                                     keep_pruned_zeros=keep_zeros)
        q.to(device)
        _, _, best = qat.qat_finetune(model, q, train_loader, val_loader, device,
                                      epochs=args.qat_epochs, lr=args.qat_lr,
                                      log_prefix=f"[s{seed}] ")
        qat.restore_best(model, q, best)
        q.reconstruct(model)
        return test_acc(model), (model, q)

    def sweep(bits_dict, state, keep_zeros, size_fn):
        tests, size = [], None
        for s in args.seeds:
            t, (m, q) = qat_test(bits_dict, state, s, keep_zeros)
            tests.append(t)
            if size is None:
                size = size_fn(m, q)
        mu, sd = mean_std(tests)
        return {"tests": tests, "mean": mu, "std": sd, "size_MB": size}

    dense_size = lambda m, q: model_size_bits(base, q) / 8 / 1e6
    sparse_size = lambda m, q: sparse_quant_size_bits(m, q) / 8 / 1e6

    out = {"seeds": args.seeds, "fp32_test": test_acc(base), "fp32_MB": fp32_MB}

    # ---- dense ----------------------------------------------------------- #
    print("\n=== dense: uniform refs + mixed ===")
    d_uni = {}
    for b in args.uniform_ref:
        d_uni[b] = sweep(uniform_bits(base, b), base_state, False, dense_size)
        print(f"  uniform {b}-bit: {d_uni[b]['mean']*100:.2f}±{d_uni[b]['std']*100:.2f}%  "
              f"{d_uni[b]['size_MB']:.2f}MB  {d_uni[b]['tests']}")
    d_mix = sweep(dense_mixed_bits, base_state, False, dense_size)
    d_bar = pareto_interp([(d_uni[b]["size_MB"], d_uni[b]["mean"]) for b in args.uniform_ref],
                          d_mix["size_MB"])
    print(f"  mixed {dense_avg:.2f}-bit: {d_mix['mean']*100:.2f}±{d_mix['std']*100:.2f}%  "
          f"{d_mix['size_MB']:.2f}MB  bar={d_bar*100:.2f}%  "
          f"({(d_mix['mean']-d_bar)*100:+.2f} pp)")

    out["dense"] = {"uniform": {str(b): d_uni[b] for b in args.uniform_ref},
                    "mixed": {**d_mix, "avg_bits": dense_avg, "pareto_bar": d_bar,
                              "beats": bool(d_mix["mean"] > d_bar)}}

    # ---- pruned ---------------------------------------------------------- #
    print("\n=== pruned: uniform refs + mixed ===")
    p_uni = {}
    for b in args.uniform_ref:
        p_uni[b] = sweep(uniform_bits(base, b), pruned_state, True, sparse_size)
        print(f"  prune+uniform {b}-bit: {p_uni[b]['mean']*100:.2f}±{p_uni[b]['std']*100:.2f}%  "
              f"{p_uni[b]['size_MB']:.2f}MB")
    p_mix = sweep(prune_mixed_bits, pruned_state, True, sparse_size)
    p_bar = pareto_interp([(p_uni[b]["size_MB"], p_uni[b]["mean"]) for b in args.uniform_ref],
                          p_mix["size_MB"])
    print(f"  prune+mixed {prune_avg:.2f}-bit: {p_mix['mean']*100:.2f}±{p_mix['std']*100:.2f}%  "
          f"{p_mix['size_MB']:.2f}MB  bar={p_bar*100:.2f}%  "
          f"({(p_mix['mean']-p_bar)*100:+.2f} pp)")

    out["prune"] = {"sparsity": sparsity, "pruned_only_test": pruned_test,
                    "uniform": {str(b): p_uni[b] for b in args.uniform_ref},
                    "mixed": {**p_mix, "avg_bits": prune_avg, "pareto_bar": p_bar,
                              "beats": bool(p_mix["mean"] > p_bar)}}

    save_json(out, os.path.join(args.out, "seeds.json"))
    print("\nDONE. Results in", args.out)


if __name__ == "__main__":
    main()
