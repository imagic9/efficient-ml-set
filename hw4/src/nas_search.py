"""Architecture search with Hyperopt's TPE over the one-shot supernet.

Tree-structured Parzen Estimator (Bergstra et al., 2011) is a sequential
model-based optimiser: it models P(config | good) vs. P(config | bad) from the
trials seen so far and proposes configs that maximise their ratio. It handles our
purely discrete, conditional-free space directly through `hp.choice`.

Each trial is cheap because of weight sharing: we do NOT train the candidate. We
select its sub-path in the pre-trained supernet, recalibrate that path's BatchNorm
on a few training batches, and evaluate it on the validation set. The proxy score
is the validation loss; validation accuracy and parameter count are recorded too
(for the accuracy-vs-params scatter and later analysis). Identical architectures
revisited by TPE are served from a cache so the trial log stays honest.
"""
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from .search_space import OPS, WIDTHS, ACTS, NUM_STAGES, arch_key, count_arch_params
from .supernet import recalibrate_bn, evaluate_subnet


def build_space():
    """Hyperopt search space matching the three searched axes."""
    space = {f"op{i}": hp.choice(f"op{i}", OPS) for i in range(NUM_STAGES)}
    space["width"] = hp.choice("width", WIDTHS)
    space["act"] = hp.choice("act", ACTS)
    return space


def _params_to_arch(params):
    return {
        "ops": [params[f"op{i}"] for i in range(NUM_STAGES)],
        "width": params["width"],
        "act": params["act"],
    }


def search_tpe(model, train_loader, val_loader, device, max_evals=200, seed=42,
               recal_batches=64):
    """Run TPE search over the supernet. Returns (best_arch, records).

    `records` is an ordered list, one entry per trial (in trial order):
        {trial, arch, params, val_loss, val_acc, best_loss_so_far, cached}
    where `params` is the standalone parameter count and `best_loss_so_far` gives
    the running-best proxy loss for the convergence plot.
    """
    model.eval()
    cache, records = {}, []

    def objective(params):
        arch = _params_to_arch(params)
        key = arch_key(arch)
        cached = key in cache
        if cached:
            val_loss, val_acc = cache[key]
        else:
            recalibrate_bn(model, arch, train_loader, device, recal_batches)
            val_loss, val_acc = evaluate_subnet(model, arch, val_loader, device)
            cache[key] = (val_loss, val_acc)
        best = min([r["val_loss"] for r in records] + [val_loss])
        records.append({
            "trial": len(records),
            "arch": arch,
            "params": count_arch_params(arch),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_loss_so_far": best,
            "cached": cached,
        })
        print(f"  trial {len(records):3d}/{max_evals}  "
              f"ops={'/'.join(arch['ops'])} w={arch['width']} act={arch['act']:9s} "
              f"params={records[-1]['params']/1e6:.2f}M  "
              f"val_acc={val_acc:.4f} val_loss={val_loss:.4f}"
              f"{'  (cached)' if cached else ''}")
        return {"loss": val_loss, "status": STATUS_OK}

    trials = Trials()
    fmin(objective, build_space(), algo=tpe.suggest, max_evals=max_evals,
         trials=trials, rstate=np.random.default_rng(seed), show_progressbar=False)

    best = min(records, key=lambda r: r["val_loss"])
    return best["arch"], records


def top_k_by_proxy(records, k):
    """The k distinct architectures with the highest proxy validation accuracy."""
    seen, out = set(), []
    for r in sorted(records, key=lambda r: r["val_acc"], reverse=True):
        key = arch_key(r["arch"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
        if len(out) >= k:
            break
    return out
