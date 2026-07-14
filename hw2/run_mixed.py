"""HW2 bonus: mixed-precision quantization (sensitivity-driven) + pruning pipeline.

Two experiments, both judged on equal footing and with the test set touched once per final:

  A) Mixed-precision vs uniform at the SAME average bit budget.
     A per-layer bit-width sensitivity scan drives a mixed allocation whose
     parameter-weighted average equals the uniform baseline's bit-width. If the
     bonus is real, mixed precision beats uniform at (essentially) equal size.

  B) Deep Compression: prune first (reuse HW1's 80%-sparse iterative model),
     then K-Means quantize the surviving non-zero weights and fine-tune. Reports
     the combined size in a sparse+quantized format.

    python run_mixed.py --baseline ../hw1/results/baseline.pt \
        --pruned ../hw1/results/iterative_final.pt --data-dir ./data
    python run_mixed.py --smoke
"""
import argparse
import os

import torch

from src.data import build_loaders
from src.model import build_vgg11_cifar
from src.engine import evaluate
from src.kmeans_quant import (KMeansQuantizer, quantizable_layers,
                              compression_report, model_size_bits,
                              sparse_quant_size_bits)
from src import qat, mixed, plots
from src.prune import model_sparsity
from src.utils import set_seed, get_device, save_json


def uniform_bits(model, b):
    return {name: b for name, _ in quantizable_layers(model)}


def pareto_interp(points, x):
    """Linear-interpolate accuracy at size x on the uniform (size, acc) Pareto line.

    `points` is a list of (size_MB, acc) for the integer uniform bit-widths, sorted
    by size. Returns the accuracy the uniform frontier would reach at model size x --
    the bar a mixed-precision point of that size must clear to count as a win.
    """
    pts = sorted(points)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0) if x1 > x0 else 0.0
            return y0 + t * (y1 - y0)
    return pts[0][1] if x < pts[0][0] else pts[-1][1]     # clamp outside the range


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="../hw1/results/baseline.pt")
    ap.add_argument("--pruned", default="../hw1/results/iterative_final.pt",
                    help="HW1 iterative-pruned (~80% sparse) state_dict")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--avg-bits", type=float, default=2.5,
                    help="target average bit budget for the mixed allocation "
                         "(picked in the lossy 2-3 bit range where mixed can help)")
    ap.add_argument("--uniform-ref", type=int, nargs="+", default=[2, 3],
                    help="integer uniform bit-widths forming the uniform Pareto line")
    ap.add_argument("--prune-quant-bits", type=int, default=4,
                    help="bit-width for the headline lossless prune+quant model")
    ap.add_argument("--qat-epochs", type=int, default=10)
    ap.add_argument("--qat-lr", type=float, default=1e-3)
    ap.add_argument("--kmeans-iters", type=int, default=30)
    ap.add_argument("--skip-prune", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
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

    base_state = torch.load(args.baseline, map_location=device)
    base = build_vgg11_cifar().to(device)
    base.load_state_dict(base_state)
    fp32_val, fp32_test = val_acc(base), test_acc(base)
    fp32_MB = sum(p.numel() for p in base.parameters()) * 32 / 8 / 1e6
    print(f"fp32 baseline: val={fp32_val:.4f} TEST={fp32_test:.4f} {fp32_MB:.2f}MB")

    out = {"fp32": {"val": fp32_val, "test": fp32_test, "size_MB": fp32_MB}}

    def run_qat(bits_dict, tag, state, keep_zeros=False):
        model = build_vgg11_cifar().to(device)
        model.load_state_dict(state)
        q = KMeansQuantizer.quantize(model, bits_dict, iters=args.kmeans_iters,
                                     keep_pruned_zeros=keep_zeros)
        q.to(device)
        ptq_val = val_acc(model)
        hist, qat_val, best = qat.qat_finetune(
            model, q, train_loader, val_loader, device,
            epochs=args.qat_epochs, lr=args.qat_lr, log_prefix=f"[{tag}] ")
        qat.restore_best(model, q, best)
        q.reconstruct(model)
        t = test_acc(model)
        return model, q, {"ptq_val": ptq_val, "qat_val": qat_val, "qat_test": t,
                          "history": hist}

    # ================================================================== #
    # A) Mixed-precision beats the uniform Pareto frontier (dense model)
    #    Compared at a lossy budget (~2.5 bits) where allocation can matter.
    # ================================================================== #
    print("\n=== [A] bit-width sensitivity scan (dense) ===")
    curves = mixed.bit_sensitivity_scan(
        build_vgg11_cifar, base_state, device, val_loader,
        bit_levels=(2, 3, 4), kmeans_iters=max(8, args.kmeans_iters // 2))
    plots.plot_bit_sensitivity(curves, fp32_val).savefig(
        os.path.join(args.out, "bit_sensitivity.png"), dpi=120)

    mixed_bits, mixed_avg = mixed.allocate_mixed_bits(base, curves, args.avg_bits)
    print(f"  mixed allocation (avg={mixed_avg:.3f} bits): "
          + ", ".join(f"{n.split('.')[-2] if '.' in n else n}:{b}"
                      for n, b in mixed_bits.items()))

    uni_pareto = []            # (size_MB, test) for each integer uniform bit-width
    uni_results = {}
    for b in args.uniform_ref:
        print(f"\n=== [A] uniform {b}-bit QAT ===")
        m_u, q_u, r_u = run_qat(uniform_bits(base, b), f"uni{b}", base_state)
        s = model_size_bits(base, q_u) / 8 / 1e6
        uni_pareto.append((s, r_u["qat_test"]))
        uni_results[b] = {**r_u, "bits": b, "size_MB": s}
        print(f"  uniform {b}-bit: TEST={r_u['qat_test']:.4f}  {s:.2f}MB")

    print(f"\n=== [A] mixed-precision QAT (avg {mixed_avg:.2f} bits) ===")
    m_m, q_m, r_m = run_qat(mixed_bits, "mixed", base_state)
    size_m = model_size_bits(base, q_m) / 8 / 1e6
    bar = pareto_interp(uni_pareto, size_m)           # uniform frontier at mixed's size
    print(f"  mixed {mixed_avg:.2f}-bit: TEST={r_m['qat_test']:.4f}  {size_m:.2f}MB")
    print(f"  uniform Pareto bar at {size_m:.2f}MB = {bar*100:.2f}%  "
          f"-> mixed is {(r_m['qat_test']-bar)*100:+.2f} pp {'ABOVE' if r_m['qat_test']>bar else 'below'} the line")

    out["mixed_bits"] = mixed_bits
    out["mixed_avg_bits"] = mixed_avg
    out["uniform_ref"] = uni_results
    out["uniform_pareto"] = uni_pareto
    out["mixed"] = {**r_m, "avg_bits": mixed_avg, "size_MB": size_m,
                    "pareto_bar": bar, "beats_pareto": bool(r_m["qat_test"] > bar)}
    out["bit_sensitivity"] = curves

    # ================================================================== #
    # B) Improve the iterative-pruning result with mixed-precision quant.
    #    Prune first (HW1's 80% model), then quantize the surviving non-zeros;
    #    show mixed-precision beats uniform-precision quant of the same pruned net.
    # ================================================================== #
    if not args.skip_prune and os.path.exists(args.pruned):
        print("\n=== [B] improve pruning: prune -> mixed-precision quantization ===")
        pruned_state = torch.load(args.pruned, map_location=device)
        m_p = build_vgg11_cifar().to(device)
        m_p.load_state_dict(pruned_state)
        sp = model_sparsity(m_p)
        pruned_test = test_acc(m_p)
        print(f"  pruned model: sparsity={sp:.3f} TEST={pruned_test:.4f}")

        # bit sensitivity on the pruned model (cluster only the surviving non-zeros)
        pcurves = mixed.bit_sensitivity_scan(
            build_vgg11_cifar, pruned_state, device, val_loader,
            bit_levels=(2, 3, 4), kmeans_iters=max(8, args.kmeans_iters // 2),
            keep_pruned_zeros=True)
        p_mixed_bits, p_mixed_avg = mixed.allocate_mixed_bits(m_p, pcurves, args.avg_bits)

        def prune_run(bits_dict, tag):
            m, q, r = run_qat(bits_dict, tag, pruned_state, keep_zeros=True)
            r["sparse_MB"] = sparse_quant_size_bits(m, q) / 8 / 1e6
            r["compression_x"] = fp32_MB / r["sparse_MB"]
            return r

        pareto_p = []
        pu = {}
        for b in args.uniform_ref:
            r = prune_run(uniform_bits(base, b), f"p-uni{b}")
            pareto_p.append((r["sparse_MB"], r["qat_test"]))
            pu[b] = {**r, "bits": b}
            print(f"  prune+uniform {b}-bit: TEST={r['qat_test']:.4f}  {r['sparse_MB']:.2f}MB "
                  f"({r['compression_x']:.1f}x)")

        r_pm = prune_run(p_mixed_bits, "p-mixed")
        bar_p = pareto_interp(pareto_p, r_pm["sparse_MB"])
        print(f"  prune+mixed {p_mixed_avg:.2f}-bit: TEST={r_pm['qat_test']:.4f}  "
              f"{r_pm['sparse_MB']:.2f}MB ({r_pm['compression_x']:.1f}x)  "
              f"-> {(r_pm['qat_test']-bar_p)*100:+.2f} pp {'ABOVE' if r_pm['qat_test']>bar_p else 'below'} uniform line")

        # headline: near-lossless prune + 4-bit quant (max compression)
        r_hi = prune_run(uniform_bits(base, args.prune_quant_bits), "p-uni4")
        print(f"  prune+quant {args.prune_quant_bits}-bit (headline): TEST={r_hi['qat_test']:.4f}  "
              f"{r_hi['sparse_MB']:.2f}MB -> {r_hi['compression_x']:.1f}x vs fp32")

        out["prune"] = {"sparsity": sp, "pruned_only_test": pruned_test}
        out["prune_uniform_ref"] = pu
        out["prune_uniform_pareto"] = pareto_p
        out["prune_mixed_bits"] = p_mixed_bits
        out["prune_mixed"] = {**r_pm, "avg_bits": p_mixed_avg, "pareto_bar": bar_p,
                              "beats_pareto": bool(r_pm["qat_test"] > bar_p)}
        out["prune_quant_headline"] = {**r_hi, "bits": args.prune_quant_bits}
        out["prune_bit_sensitivity"] = pcurves

        plots.plot_size_vs_acc(
            {"prune+uniform (Pareto)": pareto_p,
             "prune+mixed": [(r_pm["sparse_MB"], r_pm["qat_test"])]},
            baseline=(fp32_MB, fp32_test),
            title="Improve pruning: mixed vs uniform quantization (sparse size)"
        ).savefig(os.path.join(args.out, "prune_mixed_pareto.png"), dpi=120)

    save_json(out, os.path.join(args.out, "mixed.json"))
    print("\nDONE. Results in", args.out)


if __name__ == "__main__":
    main()
