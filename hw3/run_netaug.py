"""HW3 bonus: NetAug (Network Augmentation) + KD on a tiny (0.25x) VGG11.

Target student = a width-compressed VGG11 (0.25x channels) trained from scratch on
CIFAR-10 -- a small model that *underfits*, which is exactly the regime NetAug helps.
Teacher = our full fp32 VGG11 (~90.7%), frozen.

A 2x2 comparison, every variant training the SAME base sub-network from the SAME
init (seeded), for the same epochs/optimiser -- only the training method differs:

                 |  CE loss        |  KD loss (teacher soft targets)
  -----------------------------------------------------------------
  standard       |  ce             |  kd
  NetAug         |  netaug_ce      |  netaug_kd

If NetAug and KD each help the underfitting tiny net, netaug_kd should be the best
corner -- the two techniques compound. Test is measured once per final base model.

    python run_netaug.py --baseline ../hw1/results/baseline.pt --data-dir ./data
    python run_netaug.py --smoke
"""
import argparse
import os
import time

import torch

from src.data import build_loaders
from src.model import build_vgg11_cifar, count_parameters
from src.engine import evaluate
from src.netaug import ElasticVGG11, netaug_train, evaluate_base, recalibrate_base_bn
from src.distill import DistillLoss
from src.utils import set_seed, get_device, save_json


def build_base(base_mult, aug_mult, device, seed):
    set_seed(seed)                          # identical base init for every variant
    return ElasticVGG11(base_mult=base_mult, aug_mult=aug_mult).to(device)


def run_variant(name, netaug, use_teacher, teacher, train_loader, val_loader,
                test_loader, device, args):
    model = build_base(args.base_mult, args.aug_mult, device, args.seed)
    distill = (DistillLoss(temperature=args.temperature, alpha=args.alpha)
               if use_teacher else DistillLoss(alpha=1.0))     # alpha=1 == pure CE
    set_seed(args.seed)                     # identical batch order across variants
    t0 = time.time()
    hist, best_val, best = netaug_train(
        model, teacher if use_teacher else None, train_loader, val_loader, device,
        epochs=args.epochs, distill=distill, netaug=netaug, aug_weight=args.aug_weight,
        lr=args.lr, log_prefix=f"  [{name}] ")
    model.load_state_dict({k: v.to(device) for k, v in best.items()})
    # final base BN refresh so the test number matches the trained base weights
    recalibrate_base_bn(model, train_loader, device)
    test_acc = evaluate_base(model, test_loader, device)[1]
    print(f"  {name:<12}: val={best_val:.4f} TEST={test_acc:.4f}  ({time.time()-t0:.0f}s)")
    return {"val": best_val, "test": test_acc, "history": hist}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="../hw1/results/baseline.pt")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--base-mult", type=float, default=0.25)
    ap.add_argument("--aug-mult", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--aug-weight", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.epochs = 2

    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    print(f"device={device}  torch={torch.__version__}")

    train_loader, val_loader, test_loader = build_loaders(
        args.data_dir, batch_size=args.batch_size)

    # teacher: full fp32 VGG11, frozen (test touched once, for context)
    teacher = build_vgg11_cifar().to(device)
    teacher.load_state_dict(torch.load(args.baseline, map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    t_test = evaluate(teacher, test_loader, device)[1]
    full_params = count_parameters(teacher)

    tiny = build_base(args.base_mult, args.aug_mult, device, args.seed)
    base_params = tiny.base_param_count()
    print(f"teacher (full VGG11): TEST={t_test:.4f}  params={full_params/1e6:.2f}M")
    print(f"student (tiny {args.base_mult}x VGG11): params={base_params/1e6:.3f}M "
          f"({full_params/base_params:.1f}x fewer)")

    variants = [
        ("ce",         False, False),
        ("kd",         False, True),
        ("netaug_ce",  True,  False),
        ("netaug_kd",  True,  True),
    ]
    cells = {}
    results = {"teacher_test": t_test, "full_params": full_params,
               "base_params": base_params, "base_mult": args.base_mult,
               "config": {"epochs": args.epochs, "lr": args.lr, "T": args.temperature,
                          "alpha": args.alpha, "aug_weight": args.aug_weight,
                          "aug_mult": args.aug_mult, "seed": args.seed}}
    for name, netaug, use_teacher in variants:
        print(f"\n=== {name} (netaug={netaug}, kd={use_teacher}) ===")
        r = run_variant(name, netaug, use_teacher, teacher, train_loader, val_loader,
                        test_loader, device, args)
        results[name] = r
        cells[name] = r["test"]

    save_json(results, os.path.join(args.out, "netaug.json"))
    from src import plots
    plots.plot_netaug_2x2(cells, teacher_acc=t_test, full_acc=t_test).savefig(
        os.path.join(args.out, "netaug_2x2.png"), dpi=120)

    print("\nSUMMARY (test once per final base model):")
    print(f"  full fp32 VGG11 (teacher): {t_test*100:.2f}%  ({full_params/1e6:.2f}M)")
    print(f"  tiny {args.base_mult}x VGG11 ({base_params/1e6:.3f}M):")
    for name, _, _ in variants:
        print(f"    {name:<12}: {results[name]['test']*100:.2f}%")
    kd_gain = (results["kd"]["test"] - results["ce"]["test"]) * 100
    na_gain = (results["netaug_ce"]["test"] - results["ce"]["test"]) * 100
    both = (results["netaug_kd"]["test"] - results["ce"]["test"]) * 100
    print(f"  KD alone {kd_gain:+.2f} pp | NetAug alone {na_gain:+.2f} pp | "
          f"NetAug+KD {both:+.2f} pp (vs plain CE)")
    print("\nDONE. Results in", args.out)


if __name__ == "__main__":
    main()
