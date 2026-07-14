"""HW2 core: K-Means weight-sharing quantization + centroid QAT on VGG11/CIFAR-10.

Methodology (same discipline as HW1):
  * the fp32 baseline is loaded, not retrained (reused from HW1);
  * every intermediate number -- post-training-quant accuracy, QAT curves --
    uses the VALIDATION set;
  * the TEST set is measured exactly once per final deliverable model
    (fp32 baseline, and each bit-width after QAT).

For each bit-width b in the sweep we report two points:
  * PTQ  : cluster the weights, no fine-tuning (how much accuracy quantization
           costs on its own);
  * QAT  : then fine-tune the centroids via gradient pooling (slide 36).

    python run_kmeans.py --baseline ../hw1/results/baseline.pt --data-dir ./data
    python run_kmeans.py --smoke        # quick wiring check
"""
import argparse
import os
import time

import torch

from src.data import build_loaders
from src.model import build_vgg11_cifar, count_parameters
from src.engine import evaluate
from src.kmeans_quant import (KMeansQuantizer, quantizable_layers,
                              compression_report, model_size_bits)
from src import qat
from src import plots
from src.utils import set_seed, get_device, save_json


def uniform_bits(model, bits):
    return {name: bits for name, _ in quantizable_layers(model)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="../hw1/results/baseline.pt",
                    help="fp32 VGG11 state_dict from HW1")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--qat-epochs", type=int, default=10)
    ap.add_argument("--qat-lr", type=float, default=1e-3)
    ap.add_argument("--kmeans-iters", type=int, default=30)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.bits = [2, 4]
        args.qat_epochs = 1
        args.kmeans_iters = 8

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)
    device = get_device()
    print(f"device={device}  torch={torch.__version__}")

    train_loader, val_loader, test_loader = build_loaders(
        args.data_dir, batch_size=args.batch_size)
    val_acc = lambda m: evaluate(m, val_loader, device)[1]
    test_acc = lambda m: evaluate(m, test_loader, device)[1]

    # ------------------------------------------------------------------ #
    # fp32 baseline (loaded from HW1), test touched once
    # ------------------------------------------------------------------ #
    base = build_vgg11_cifar().to(device)
    base.load_state_dict(torch.load(args.baseline, map_location=device))
    fp32_val, fp32_test = val_acc(base), test_acc(base)
    fp32_MB = sum(p.numel() for p in base.parameters()) * 32 / 8 / 1e6
    print(f"fp32 baseline: val={fp32_val:.4f} TEST={fp32_test:.4f} "
          f"size={fp32_MB:.2f}MB params={count_parameters(base)/1e6:.2f}M")

    results = {"fp32": {"val": fp32_val, "test": fp32_test, "size_MB": fp32_MB}}
    ptq_points, qat_points, size_points = [], [], []

    for b in args.bits:
        print(f"\n=== {b}-bit (K={2**b} centroids/layer) ===")
        model = build_vgg11_cifar().to(device)
        model.load_state_dict(torch.load(args.baseline, map_location=device))

        # --- PTQ: cluster only, no fine-tuning -------------------------- #
        t0 = time.time()
        quantizer = KMeansQuantizer.quantize(
            model, uniform_bits(model, b), iters=args.kmeans_iters)
        quantizer.to(device)
        ptq_val = val_acc(model)
        comp = compression_report(base, quantizer)
        size_MB = model_size_bits(base, quantizer) / 8 / 1e6
        print(f"  PTQ  val={ptq_val:.4f}  size={size_MB:.2f}MB "
              f"({comp['compression_x']:.1f}x)  ({time.time()-t0:.0f}s cluster)")

        # --- QAT: centroid fine-tuning (slide 36) ----------------------- #
        hist, qat_val, best = qat.qat_finetune(
            model, quantizer, train_loader, val_loader, device,
            epochs=args.qat_epochs, lr=args.qat_lr, log_prefix=f"[{b}bit] ")
        qat.restore_best(model, quantizer, best)
        quantizer.reconstruct(model)
        qat_test = test_acc(model)                      # test ONCE per final model
        print(f"  QAT  val={qat_val:.4f}  TEST={qat_test:.4f}")

        results[f"{b}bit"] = {
            "ptq_val": ptq_val, "qat_val": qat_val, "qat_test": qat_test,
            "size_MB": size_MB, "compression_x": comp["compression_x"],
            "avg_bits": comp["avg_bits_per_quantized_weight"], "history": hist,
        }
        ptq_points.append((b, ptq_val))
        qat_points.append((b, qat_val))
        size_points.append((size_MB, qat_test))
        plots.plot_history(hist, f"QAT {b}-bit").savefig(
            os.path.join(args.out, f"qat_{b}bit_history.png"), dpi=120)

    # ------------------------------------------------------------------ #
    # save + plots
    # ------------------------------------------------------------------ #
    save_json(results, os.path.join(args.out, "kmeans.json"))
    plots.plot_bits_vs_acc(
        {"PTQ (no fine-tune)": ptq_points, "QAT (fine-tuned)": qat_points},
        baseline_acc=fp32_val, title="K-Means quantization: accuracy vs bit-width (val)"
    ).savefig(os.path.join(args.out, "bits_vs_acc.png"), dpi=120)
    plots.plot_size_vs_acc(
        {"QAT (test)": size_points}, baseline=(fp32_MB, fp32_test),
        title="Accuracy vs model size"
    ).savefig(os.path.join(args.out, "size_vs_acc.png"), dpi=120)

    print("\nSUMMARY (test once per final model):")
    print(f"  fp32 baseline : {fp32_test:.4f}  @ {fp32_MB:.2f}MB")
    for b in args.bits:
        r = results[f"{b}bit"]
        print(f"  {b}-bit QAT    : {r['qat_test']:.4f}  @ {r['size_MB']:.2f}MB "
              f"({r['compression_x']:.1f}x)   [PTQ val {r['ptq_val']:.4f} -> "
              f"QAT val {r['qat_val']:.4f}]")
    print("\nDONE. Results in", args.out)


if __name__ == "__main__":
    main()
