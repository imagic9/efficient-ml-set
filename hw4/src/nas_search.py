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
from hyperopt import fmin, tpe, rand, hp, STATUS_OK, Trials

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


_ALGO = {"tpe": tpe.suggest, "random": rand.suggest}


def run_search(model, train_loader, val_loader, device, max_evals=200, seed=42,
               recal_batches=64, algo="tpe"):
    """Run a Hyperopt search (algo="tpe" or "random") over the supernet.

    Returns (best_arch, records). `records` is an ordered list, one per trial:
        {trial, arch, params, val_loss, val_acc, best_loss_so_far,
         n_unique_so_far, cached}
    `n_unique_so_far` counts *distinct* architectures evaluated up to and including
    this trial -- the honest x-axis for a convergence plot, since cache hits add no
    new information. Random search is the control that isolates what TPE contributes.
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
            "n_unique_so_far": len(cache),
            "cached": cached,
        })
        print(f"  [{algo}] trial {len(records):3d}/{max_evals}  "
              f"ops={'/'.join(arch['ops'])} w={arch['width']} act={arch['act']:9s} "
              f"params={records[-1]['params']/1e6:.2f}M  "
              f"val_acc={val_acc:.4f} val_loss={val_loss:.4f}"
              f"{'  (cached)' if cached else ''}")
        return {"loss": val_loss, "status": STATUS_OK}

    trials = Trials()
    fmin(objective, build_space(), algo=_ALGO[algo], max_evals=max_evals,
         trials=trials, rstate=np.random.default_rng(seed), show_progressbar=False)

    best = min(records, key=lambda r: r["val_loss"])
    return best["arch"], records


def distinct_records(records):
    """One record per distinct architecture (first occurrence), proxy val_acc kept."""
    seen, out = set(), []
    for r in records:
        key = arch_key(r["arch"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def top_k_by_proxy(records, k):
    """The k distinct architectures with the highest proxy validation accuracy."""
    ranked = sorted(distinct_records(records), key=lambda r: r["val_acc"], reverse=True)
    return ranked[:k]


def stratified_by_proxy(records, n_bins=4, per_bin=5):
    """Sample distinct archs evenly across the proxy-accuracy range.

    Ranking only the top-k cannot show whether the proxy is good at *coarse*
    filtering -- for that we need candidates spanning the whole proxy range. We sort
    the distinct archs by proxy val_acc, cut into `n_bins` equal-size quantile bins,
    and take up to `per_bin` from each (evenly spaced within the bin). Returns them
    sorted by proxy val_acc (ascending), de-duplicated.
    """
    ranked = sorted(distinct_records(records), key=lambda r: r["val_acc"])
    n = len(ranked)
    picked, seen = [], set()
    for b in range(n_bins):
        segment = ranked[b * n // n_bins:(b + 1) * n // n_bins]
        if not segment:
            continue
        # evenly spaced indices within the bin, up to per_bin of them
        k = min(per_bin, len(segment))
        idxs = sorted(set(round(i * (len(segment) - 1) / max(1, k - 1)) for i in range(k)))
        for j in idxs:
            key = arch_key(segment[j]["arch"])
            if key not in seen:
                seen.add(key)
                picked.append(segment[j])
    return sorted(picked, key=lambda r: r["val_acc"])
