"""HW3 KD ablations: temperature and alpha sweeps on the 2-bit student.

The 2-bit K-Means student is the hardest case (largest compression gap), so it is
where distillation has the most room to help -- the right place to probe the two
KD knobs:

    temperature T -- how much the teacher's distribution is softened
    alpha         -- CE vs soft-label mix (alpha=1 is pure CE, our reference line)

Everything here is measured on the VALIDATION set only -- these sweeps pick
hyper-parameters, so touching test would be leakage. The final headline numbers
come from run_distill.py, which tests once per model.

    python run_kd_ablation.py --baseline ../hw1/results/baseline.pt --data-dir ./data
"""
import argparse
import os

import torch

from src.data import build_loaders
from src.model import build_vgg11_cifar
from src.engine import evaluate
from src.kmeans_quant import KMeansQuantizer, quantizable_layers
from src import qat, plots
from src.distill import DistillLoss
from src.utils import set_seed, get_device, save_json


def uniform_bits(model, bits):
    return {name: bits for name, _ in quantizable_layers(model)}


def fresh_2bit_student(baseline_path, device, kmeans_iters):
    model = build_vgg11_cifar().to(device)
    model.load_state_dict(torch.load(baseline_path, map_location=device))
    q = KMeansQuantizer.quantize(model, uniform_bits(model, 2), iters=kmeans_iters)
    q.to(device)
    return model, q


def run_one(baseline_path, teacher, train_loader, val_loader, device, epochs,
            T, alpha, seed, kmeans_iters, tag):
    """One 2-bit QAT fine-tune with the given (T, alpha); returns best val acc."""
    set_seed(seed)                                  # same init + batch order every point
    model, q = fresh_2bit_student(baseline_path, device, kmeans_iters)
    use_teacher = teacher if alpha < 1.0 else None  # alpha=1 -> pure CE, no teacher needed
    distill = DistillLoss(temperature=T, alpha=alpha) if use_teacher is not None else None
    _, best_val, _ = qat.qat_finetune(
        model, q, train_loader, val_loader, device, epochs=epochs,
        teacher=use_teacher, distill=distill, log_prefix=f"  [{tag}] ")
    return best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="../hw1/results/baseline.pt")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--temps", type=float, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7, 1.0])
    ap.add_argument("--fixed-alpha", type=float, default=0.5)   # for the T sweep
    ap.add_argument("--fixed-temp", type=float, default=4.0)    # for the alpha sweep
    ap.add_argument("--kmeans-iters", type=int, default=30)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.temps = [1, 4]
        args.alphas = [0.5, 1.0]
        args.epochs = 1
        args.kmeans_iters = 8

    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    print(f"device={device}  torch={torch.__version__}")

    train_loader, val_loader, test_loader = build_loaders(
        args.data_dir, batch_size=args.batch_size)

    teacher = build_vgg11_cifar().to(device)
    teacher.load_state_dict(torch.load(args.baseline, map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    t_val = evaluate(teacher, val_loader, device)[1]
    print(f"teacher fp32 val={t_val:.4f}")

    out = {"teacher_val": t_val, "fixed_alpha": args.fixed_alpha,
           "fixed_temp": args.fixed_temp, "epochs": args.epochs}

    # --- temperature sweep (alpha fixed) ----------------------------------- #
    print(f"\n=== temperature sweep (alpha={args.fixed_alpha}) ===")
    temp_points = []
    for T in args.temps:
        v = run_one(args.baseline, teacher, train_loader, val_loader, device,
                    args.epochs, T, args.fixed_alpha, args.seed, args.kmeans_iters,
                    tag=f"T={T}")
        print(f"  T={T}: val={v:.4f}")
        temp_points.append((T, v))
    out["temperature_sweep"] = temp_points

    # --- alpha sweep (T fixed) --------------------------------------------- #
    print(f"\n=== alpha sweep (T={args.fixed_temp}) ===")
    alpha_points = []
    ce_ref = None
    for a in args.alphas:
        v = run_one(args.baseline, teacher, train_loader, val_loader, device,
                    args.epochs, args.fixed_temp, a, args.seed, args.kmeans_iters,
                    tag=f"a={a}")
        print(f"  alpha={a}: val={v:.4f}")
        alpha_points.append((a, v))
        if a >= 1.0:
            ce_ref = v                              # pure-CE reference line
    out["alpha_sweep"] = alpha_points
    out["ce_only_val"] = ce_ref

    save_json(out, os.path.join(args.out, "kd_ablation.json"))

    plots.plot_kd_ablation(
        [t for t, _ in temp_points], [v for _, v in temp_points],
        xlabel="temperature T", baseline_val=t_val, ce_val=ce_ref,
        title=f"2-bit student: KD temperature sweep (alpha={args.fixed_alpha}, val)"
    ).savefig(os.path.join(args.out, "kd_temp_sweep.png"), dpi=120)

    plots.plot_kd_ablation(
        [a for a, _ in alpha_points], [v for _, v in alpha_points],
        xlabel="alpha (CE weight)", baseline_val=t_val,
        title=f"2-bit student: KD alpha sweep (T={args.fixed_temp}, val)"
    ).savefig(os.path.join(args.out, "kd_alpha_sweep.png"), dpi=120)

    print("\nDONE. Ablation results in", args.out)


if __name__ == "__main__":
    main()
