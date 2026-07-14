"""HW3 core: self-distillation of compressed VGG11 students on CIFAR-10.

Teacher  = our uncompressed fp32 VGG11 (HW1 baseline.pt, ~90.7% test), frozen.
Student  = the same VGG11 architecture, compressed with the HW1/HW2 tooling:

    pruned          -- 80% unstructured (global magnitude) sparsity
    quant2          -- 2-bit K-Means weight sharing (the hardest case)
    prune_quant     -- 80% pruned + 4-bit quantized non-zeros (Deep Compression)

For every regime we run TWO fine-tunes from the *same* compressed init, with
identical optimiser / schedule / epochs / seed -- the only difference is the loss:

    CE-only : plain cross-entropy recovery (the honest baseline)
    KD      : Hinton distillation against the teacher's soft targets

so any accuracy gap is attributable to distillation alone. This is the central
KD-vs-CE comparison the report is built around.

Methodology (same discipline as HW1/HW2): teacher is loaded not retrained; every
intermediate number is on the VALIDATION set; the TEST set is touched exactly
once per final model (CE and KD are each one final model).

    python run_distill.py --baseline ../hw1/results/baseline.pt --data-dir ./data
    python run_distill.py --smoke        # quick wiring check
"""
import argparse
import copy
import os
import time

import torch

from src.data import build_loaders
from src.model import build_vgg11_cifar, count_parameters
from src.engine import evaluate
from src.prune import FineGrainedPruner, model_sparsity
from src.kmeans_quant import (KMeansQuantizer, quantizable_layers,
                              model_size_bits, sparse_quant_size_bits)
from src import qat, plots
from src.distill import DistillLoss, kd_train
from src.utils import set_seed, get_device, save_json


def uniform_bits(model, bits):
    return {name: bits for name, _ in quantizable_layers(model)}


@torch.no_grad()
def pruned_sparse_size_bits(model):
    """Deep-Compression-style size of a pruned (unquantized) model:
    32 bits per surviving weight + a 1-bit position bitmap, other params fp32."""
    from src.prune import prunable_layers
    total = sum(p.numel() for p in model.parameters()) * 32
    for _, m in prunable_layers(model):
        w = m.weight.data
        n, nnz = w.numel(), int((w != 0).sum().item())
        total -= n * 32
        total += nnz * 32 + n * 1
    return total


# --------------------------------------------------------------------------- #
# build a compressed student (fresh, deterministic) for a given regime
# --------------------------------------------------------------------------- #
def make_student(regime, baseline_path, device, kmeans_iters=30):
    """Return (model, quantizer_or_None, pruner_or_None, size_bits, tag)."""
    model = build_vgg11_cifar().to(device)
    model.load_state_dict(torch.load(baseline_path, map_location=device))

    if regime == "pruned":
        pruner = FineGrainedPruner.from_global(model, sparsity=0.80)
        pruner.apply(model)
        return model, None, pruner, pruned_sparse_size_bits(model), "pruned 80%"

    if regime == "quant2":
        q = KMeansQuantizer.quantize(model, uniform_bits(model, 2), iters=kmeans_iters)
        q.to(device)
        return model, q, None, model_size_bits(model, q), "2-bit quant"

    if regime == "prune_quant":
        pruner = FineGrainedPruner.from_global(model, sparsity=0.80)
        pruner.apply(model)
        q = KMeansQuantizer.quantize(model, uniform_bits(model, 4), iters=kmeans_iters,
                                     keep_pruned_zeros=True)
        q.to(device)
        return model, q, pruner, sparse_quant_size_bits(model, q), "prune80%+4-bit"

    raise ValueError(regime)


def finetune(model, quantizer, pruner, teacher, train_loader, val_loader, device,
             epochs, T, alpha, prune_lr, log_prefix):
    """One fine-tune. teacher=None -> CE-only; teacher set -> KD. Returns (hist, best_val, best)."""
    if quantizer is None:
        # dense / pruned student: SGD + cosine, masks re-applied by the pruner
        distill = DistillLoss(temperature=T, alpha=alpha) if teacher is not None \
            else DistillLoss(alpha=1.0)                     # alpha=1 == pure CE
        return kd_train(model, teacher, train_loader, val_loader, device, epochs,
                        distill, lr=prune_lr, pruner=pruner, log_prefix=log_prefix)
    # quantized student: centroid QAT (teacher/distill are optional inside qat_finetune)
    distill = DistillLoss(temperature=T, alpha=alpha) if teacher is not None else None
    return qat.qat_finetune(model, quantizer, train_loader, val_loader, device,
                            epochs=epochs, log_prefix=log_prefix,
                            teacher=teacher, distill=distill)


def restore(model, quantizer, best):
    if quantizer is None:
        model.load_state_dict({k: v.to(next(model.parameters()).device)
                               for k, v in best.items()})
    else:
        qat.restore_best(model, quantizer, best)
        quantizer.reconstruct(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="../hw1/results/baseline.pt")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--regimes", nargs="+",
                    default=["pruned", "quant2", "prune_quant"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--prune-lr", type=float, default=0.01)
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--kmeans-iters", type=int, default=30)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.regimes = ["pruned", "quant2"]
        args.epochs = 1
        args.kmeans_iters = 8

    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    print(f"device={device}  torch={torch.__version__}")

    train_loader, val_loader, test_loader = build_loaders(
        args.data_dir, batch_size=args.batch_size)
    val_acc = lambda m: evaluate(m, val_loader, device)[1]
    test_acc = lambda m: evaluate(m, test_loader, device)[1]

    # ---- teacher: fp32 baseline, frozen (test touched once) ---------------- #
    teacher = build_vgg11_cifar().to(device)
    teacher.load_state_dict(torch.load(args.baseline, map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    t_val, t_test = val_acc(teacher), test_acc(teacher)
    fp32_MB = sum(p.numel() for p in teacher.parameters()) * 32 / 8 / 1e6
    print(f"teacher fp32: val={t_val:.4f} TEST={t_test:.4f} size={fp32_MB:.2f}MB "
          f"params={count_parameters(teacher)/1e6:.2f}M")

    results = {"teacher": {"val": t_val, "test": t_test, "size_MB": fp32_MB},
               "config": {"epochs": args.epochs, "T": args.temperature,
                          "alpha": args.alpha, "prune_lr": args.prune_lr,
                          "seed": args.seed}}

    for regime in args.regimes:
        print(f"\n{'='*60}\n=== student regime: {regime} ===\n{'='*60}")
        reg_out = {}

        # --- CE-only recovery (teacher=None) -------------------------------- #
        set_seed(args.seed)                        # identical init + batch order
        model, q, pruner, size_bits, tag = make_student(
            regime, args.baseline, device, args.kmeans_iters)
        pre_val = val_acc(model)
        size_MB = size_bits / 8 / 1e6
        comp = fp32_MB / size_MB
        print(f"  [{tag}] compressed init: val={pre_val:.4f}  "
              f"size={size_MB:.2f}MB ({comp:.1f}x)  sparsity={model_sparsity(model):.2f}")

        t0 = time.time()
        hist_ce, ce_val, best_ce = finetune(
            model, q, pruner, None, train_loader, val_loader, device,
            args.epochs, args.temperature, args.alpha, args.prune_lr,
            log_prefix=f"  [CE {regime}] ")
        restore(model, q, best_ce)
        ce_test = test_acc(model)
        print(f"  CE-only: val={ce_val:.4f} TEST={ce_test:.4f}  ({time.time()-t0:.0f}s)")

        # --- KD recovery (same init + seed, teacher on) --------------------- #
        set_seed(args.seed)
        model, q, pruner, size_bits, tag = make_student(
            regime, args.baseline, device, args.kmeans_iters)
        t0 = time.time()
        hist_kd, kd_val, best_kd = finetune(
            model, q, pruner, teacher, train_loader, val_loader, device,
            args.epochs, args.temperature, args.alpha, args.prune_lr,
            log_prefix=f"  [KD {regime}] ")
        restore(model, q, best_kd)
        kd_test = test_acc(model)
        print(f"  KD     : val={kd_val:.4f} TEST={kd_test:.4f}  ({time.time()-t0:.0f}s)")
        print(f"  >>> KD - CE : val {(kd_val-ce_val)*100:+.2f} pp | "
              f"test {(kd_test-ce_test)*100:+.2f} pp")

        reg_out = {
            "tag": tag, "size_MB": size_MB, "compression_x": comp,
            "pre_finetune_val": pre_val,
            "ce": {"val": ce_val, "test": ce_test, "history": hist_ce},
            "kd": {"val": kd_val, "test": kd_test, "history": hist_kd},
        }
        results[regime] = reg_out

        plots.plot_history(hist_kd, f"KD {tag}").savefig(
            os.path.join(args.out, f"kd_{regime}_history.png"), dpi=120)

    # ---- comparison bar chart --------------------------------------------- #
    regimes = [r for r in args.regimes if r in results]
    tags = [results[r]["tag"] for r in regimes]
    ce_t = [results[r]["ce"]["test"] for r in regimes]
    kd_t = [results[r]["kd"]["test"] for r in regimes]
    plots.plot_kd_comparison(tags, ce_t, kd_t, baseline_acc=t_test).savefig(
        os.path.join(args.out, "kd_vs_ce.png"), dpi=120)

    save_json(results, os.path.join(args.out, "distill.json"))

    print("\nSUMMARY (test once per final model):")
    print(f"  teacher fp32 : {t_test:.4f}  @ {fp32_MB:.2f}MB")
    for r in regimes:
        d = results[r]
        print(f"  {d['tag']:<16}: CE {d['ce']['test']:.4f} | KD {d['kd']['test']:.4f} "
              f"({(d['kd']['test']-d['ce']['test'])*100:+.2f} pp) @ {d['size_MB']:.2f}MB "
              f"({d['compression_x']:.1f}x)")
    print("\nDONE. Results in", args.out)


if __name__ == "__main__":
    main()
