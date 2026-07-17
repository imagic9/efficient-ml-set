#!/usr/bin/env python3
"""D3 — dependency-aware structured pruning of MobileNetV2 expansion channels.

`hw1/src/structured.py` is the ancestor, not the implementation: that code
pruned every conv of a CIFAR VGG under a plain accuracy metric. Here DESIGN
§8.3 narrows the contract to exactly one family of roots — the 16 expansion
`1x1` convolutions of the `t=6` inverted-residual blocks — because everything
else in MobileNetV2 is width-coupled in ways that would silently change the
comparison: projection outputs feed residual adds, the `t=1` block shares its
width with the stem, and the classifier is the task head.

Three lessons from probing `torch-pruning` 1.6.1 are load-bearing here:

- **A group is skipped if any out-channel member is in `ignored_layers`.** The
  depthwise conv is an out-channel member of the expansion group, so ignoring
  depthwise convs — the intuitive reading of "only expansion roots" — silently
  ignores the whole group and prunes nothing. Depthwise convs of `t=6` blocks
  must therefore stay *out* of the ignored list; root deduplication keeps them
  from forming their own groups because the expansion conv visits them first.
- **Projection convs participate on the in-channel side**, which does not
  disqualify a group, so they can (and must) be ignored as roots.
- **`round_to=8` rounds surviving widths**, and the realized width is what the
  evidence records; the request is only a request (DESIGN §8.3 step 4 requires
  both, separately).

Usage (sensitivity, D3):
    python -m wildlife_trigger.optimize.prune \
        --config configs/optimize/m3_prune.yaml --sensitivity
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch_pruning as tp
import yaml

from ..models.mobilenet import build_mobilenet_v2
from ..runs import atomic_write_json, sha256_file

# The frozen deployment input (DESIGN §5.5, decided by C1a).
EXAMPLE_SHAPE = (1, 3, 192, 256)

# The frozen architecture's fixed widths, asserted after every mutation.
# MobileNetV2 width_mult=1.0: stem 32, t=1 block 16-out, final conv 1280.
STEM_OUT = 32
T1_DW = 32
T1_PROJECT_OUT = 16
FINAL_CONV_OUT = 1280

# torchvision's projection output widths for features.2..17, in order. Residual
# adds happen at these widths; they are the contract's "never mutated" list.
PROJECT_OUT = (24, 24, 32, 32, 32, 64, 64, 64, 64, 96, 96, 96, 160, 160, 160, 320)

EXPANSION_BLOCKS = tuple(range(2, 18))


@dataclass(frozen=True)
class PruningPlan:
    """The classified model: what may root a group, and what never moves."""

    expansion: dict[int, nn.Conv2d]  # block index -> expansion 1x1 conv
    ignored: list[nn.Module]  # roots torch-pruning must not start from

    def conv_name(self, block: int) -> str:
        return f"features.{block}.conv.0.0"


def classify(model: nn.Module) -> PruningPlan:
    """Split the frozen MobileNetV2 into expansion roots and everything else.

    Refuses any architecture whose shape differs from the frozen one — a plan
    computed for the wrong model would prune plausible-looking but different
    channels, and nothing downstream could tell.
    """
    features = model.features
    expansion: dict[int, nn.Conv2d] = {}
    ignored: list[nn.Module] = [features[0][0], features[18][0]]

    for index in range(1, 18):
        conv = features[index].conv
        if index == 1:
            if len(conv) != 3:
                raise ValueError(
                    f"features.1 has {len(conv)} conv stages, expected the t=1 "
                    "block's 3; this is not the frozen MobileNetV2"
                )
            # The t=1 depthwise is width-coupled to the stem: ignoring it as a
            # root is what keeps the stem width fixed.
            ignored.extend([conv[0][0], conv[1]])
            continue
        if len(conv) != 4:
            raise ValueError(
                f"features.{index} has {len(conv)} conv stages, expected the "
                "t=6 block's 4; this is not the frozen MobileNetV2"
            )
        expand = conv[0][0]
        if expand.kernel_size != (1, 1) or expand.groups != 1:
            raise ValueError(
                f"features.{index}.conv.0.0 is not a 1x1 dense conv; refusing "
                "to root a pruning group on it"
            )
        expansion[index] = expand
        # Projection as a root would prune residual widths. As an in-channel
        # *member* of the expansion group it still updates — that is the point.
        ignored.append(conv[2])

    classifier = model.classifier[-1]
    if not isinstance(classifier, nn.Linear):
        raise ValueError("classifier[-1] is not Linear; not the frozen model")
    ignored.append(classifier)

    if len(expansion) != len(EXPANSION_BLOCKS):
        raise ValueError(
            f"found {len(expansion)} expansion convs, expected "
            f"{len(EXPANSION_BLOCKS)}; not the frozen MobileNetV2"
        )
    return PruningPlan(expansion=expansion, ignored=ignored)


def verify_group(model: nn.Module, plan: PruningPlan, block: int) -> dict:
    """Prove one root's dependency group is exactly the DESIGN §8.3 coupling.

    Evidence, not trust: the group must span the expansion conv/BN, the
    depthwise conv/BN (out-channel side) and the projection conv (in-channel
    side) of the SAME block, and nothing else that owns weights. An extra
    member means the solver reached outside the contract; a missing one means
    the coupling silently broke.
    """
    conv = model.features[block].conv
    expected_out = {id(conv[0][0]), id(conv[0][1]), id(conv[1][0]), id(conv[1][1])}
    expected_in = {id(conv[2])}

    graph = tp.dependency.DependencyGraph().build_dependency(
        model, example_inputs=torch.zeros(EXAMPLE_SHAPE)
    )
    group = graph.get_pruning_group(
        plan.expansion[block],
        tp.prune_conv_out_channels,
        idxs=list(range(plan.expansion[block].out_channels)),
    )

    seen_out, seen_in, members = set(), set(), []
    for dependency, _ in group:
        module = dependency.target.module
        handler = getattr(dependency.handler, "__name__", str(dependency.handler))
        members.append({"module": type(module).__name__, "handler": handler})
        if not isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
            continue  # ElementWiseOps carry no weights
        if "out_channels" in handler:
            seen_out.add(id(module))
        elif "in_channels" in handler:
            seen_in.add(id(module))

    if seen_out != expected_out or seen_in != expected_in:
        raise RuntimeError(
            f"features.{block}'s dependency group does not match the DESIGN "
            f"§8.3 coupling (out side: {len(seen_out)} vs 4 expected, in side: "
            f"{len(seen_in)} vs 1). The solver reached outside the contract; "
            "nothing may be pruned from this graph."
        )
    return {"block": block, "members": members, "verified": True}


def build_pruner(
    model: nn.Module,
    plan: PruningPlan,
    ratios: dict[int, float],
    round_to: int = 8,
):
    """One-shot MagnitudePruner over the given per-block expansion ratios."""
    unknown = sorted(set(ratios) - set(plan.expansion))
    if unknown:
        raise ValueError(f"blocks {unknown} are not expansion blocks")
    return tp.pruner.MagnitudePruner(
        model,
        torch.zeros(EXAMPLE_SHAPE),
        importance=tp.importance.GroupMagnitudeImportance(p=1),
        pruning_ratio=0.0,  # nothing moves unless named in the dict
        pruning_ratio_dict={plan.expansion[b]: r for b, r in ratios.items()},
        ignored_layers=plan.ignored,
        round_to=round_to,
        global_pruning=False,
    )


def profile(model: nn.Module) -> dict:
    """MACs/params from the one counter used before and after every mutation."""
    macs, params = tp.utils.count_ops_and_params(model, torch.zeros(EXAMPLE_SHAPE))
    return {"macs": int(macs), "params": int(params), "counter": "torch_pruning.utils.count_ops_and_params"}


def check_invariants(model: nn.Module, num_classes: int = 16) -> dict:
    """The §8.3 step-6 assertions. Raises on the first violation.

    Forward/backward run on the frozen input; the residual-add widths are
    asserted directly against the frozen table rather than inferred from the
    forward pass succeeding — a forward pass also succeeds when *both* sides
    of an add were mutated together, which is exactly the contract violation
    this exists to catch.
    """
    features = model.features
    widths = {}

    if features[0][0].out_channels != STEM_OUT:
        raise RuntimeError(f"stem width moved: {features[0][0].out_channels}")

    t1_dw = features[1].conv[0][0]
    if not (t1_dw.in_channels == t1_dw.out_channels == t1_dw.groups == T1_DW):
        raise RuntimeError("the t=1 depthwise block moved")
    if features[1].conv[1].out_channels != T1_PROJECT_OUT:
        raise RuntimeError("features.1 projection output moved")

    for offset, block in enumerate(EXPANSION_BLOCKS):
        conv = features[block].conv
        expand, depthwise, project = conv[0][0], conv[1][0], conv[2]
        if not (
            depthwise.in_channels == depthwise.out_channels == depthwise.groups
            == expand.out_channels == project.in_channels
        ):
            raise RuntimeError(
                f"features.{block}: depthwise/projection coupling broken "
                f"(expand {expand.out_channels}, dw {depthwise.in_channels}/"
                f"{depthwise.out_channels}/g{depthwise.groups}, proj_in "
                f"{project.in_channels})"
            )
        width = expand.out_channels
        if width < 8 or width % 8:
            raise RuntimeError(
                f"features.{block}: surviving width {width} is not a positive "
                "multiple of 8 — the SIMD-alignment contract is broken"
            )
        if project.out_channels != PROJECT_OUT[offset]:
            raise RuntimeError(
                f"features.{block}: projection output moved to "
                f"{project.out_channels} (frozen: {PROJECT_OUT[offset]}) — "
                "residual widths are not ours to prune"
            )
        if conv[0][1].num_features != width or conv[1][1].num_features != width:
            raise RuntimeError(f"features.{block}: BatchNorm width out of step")
        widths[f"features.{block}"] = width

    if features[18][0].out_channels != FINAL_CONV_OUT:
        raise RuntimeError("final 1x1 conv width moved")
    if model.classifier[-1].out_features != num_classes:
        raise RuntimeError("classifier width moved")

    was_training = model.training
    model.train()
    example = torch.zeros(EXAMPLE_SHAPE)
    output = model(example)
    if output.shape != (1, num_classes):
        raise RuntimeError(f"forward produced {tuple(output.shape)}")
    output.sum().backward()
    model.zero_grad(set_to_none=True)
    model.train(was_training)

    return {"expansion_widths": widths, "forward_backward": "ok"}


def check_onnx_export(model: nn.Module, expected_widths: dict[str, int]) -> dict:
    """Invariant 5: the mutated model exports, and the graph carries the cuts.

    Shapes are read back from the exported initializers, not assumed from the
    modules: `torch.onnx.export` re-traces the model, and a mutation that
    confused the tracer would otherwise ship the *old* widths silently.
    """
    import onnx

    from ..models.export import export_onnx

    was_training = model.training
    model.eval()
    try:
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "pruned.onnx"
            export_onnx(model, path, torch.zeros(EXAMPLE_SHAPE))
            graph = onnx.load(str(path)).graph
            exported = {}
            for initializer in graph.initializer:
                shape = tuple(initializer.dims)
                # expansion convs are the only 1x1 (Cout, Cin, 1, 1) weights
                # whose Cout we mutated; match them by their unique Cin per
                # block via name ordering instead: torch keeps module names.
                exported[initializer.name] = shape
            mismatches = []
            for name, width in expected_widths.items():
                weight = f"{name}.conv.0.0.weight"
                if weight not in exported:
                    mismatches.append(f"{weight} missing from export")
                elif exported[weight][0] != width:
                    mismatches.append(
                        f"{weight}: exported {exported[weight][0]}, module {width}"
                    )
            if mismatches:
                raise RuntimeError(
                    "ONNX export does not carry the pruned widths: "
                    + "; ".join(mismatches)
                )
            return {"onnx_export": "ok", "verified_weights": len(expected_widths)}
    finally:
        model.train(was_training)


def prune_expansion(
    model: nn.Module,
    ratios: dict[int, float],
    round_to: int = 8,
    num_classes: int = 16,
    verify_groups: bool = True,
    export_check: bool = True,
) -> dict:
    """Mutate `model` in place under the full D3 contract and return evidence."""
    plan = classify(model)
    before = profile(model)
    widths_before = {b: plan.expansion[b].out_channels for b in ratios}

    if verify_groups:
        for block in sorted(ratios):
            verify_group(model, plan, block)

    build_pruner(model, plan, ratios, round_to=round_to).step()

    invariants = check_invariants(model, num_classes=num_classes)
    after = profile(model)
    if export_check:
        invariants.update(check_onnx_export(model, invariants["expansion_widths"]))

    realized = {}
    for block, requested in sorted(ratios.items()):
        width_before = widths_before[block]
        width_after = plan.expansion[block].out_channels
        realized[f"features.{block}"] = {
            "requested_ratio": requested,
            "width_before": width_before,
            "width_after": width_after,
            "realized_ratio": round(1 - width_after / width_before, 6),
        }

    return {
        "requested": {f"features.{b}": r for b, r in sorted(ratios.items())},
        "realized": realized,
        "round_to": round_to,
        "profile_before": before,
        "profile_after": after,
        "mac_reduction": round(1 - after["macs"] / before["macs"], 6),
        "param_reduction": round(1 - after["params"] / before["params"], 6),
        "invariants": invariants,
    }


# ---------------------------------------------------------------------------
# D3 sensitivity: one group at a time, measured on validation at the yardstick.
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _bobcat_scores(model: nn.Module, loader, device: torch.device) -> dict:
    """Bobcat probabilities/presence/seq_ids for one split, deployment regime."""
    model.eval()
    column = loader.dataset.class_names.index("bobcat")
    scores, present, seq_ids = [], [], []
    for batch in loader:
        logits = model(batch["image"].to(device, non_blocking=True))
        probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
        scores.append(probabilities[:, column])
        present.append(batch["present"].numpy()[:, column])
        for i in batch["index"].tolist():
            seq_ids.append(loader.dataset.records[i]["seq_id"])
    return {
        "scores": np.concatenate(scores).astype(float),
        "present": np.concatenate(present).astype(float),
        "seq_ids": seq_ids,
    }


def evaluate_at_yardstick(model: nn.Module, loaders: dict, device) -> dict:
    """Per-domain bobcat metrics at 0.5 plus the frozen §7.2 primary."""
    from .. import metrics

    per_domain = {}
    for name, loader in loaders.items():
        data = _bobcat_scores(model, loader, device)
        measured = metrics.target_presence_metrics(
            data["scores"], data["present"], data["seq_ids"], 0.5
        )
        per_domain[name] = {
            key: measured[key]
            for key in (
                "frame_f2",
                "frame_recall",
                "frame_precision",
                "sequence_balanced_recall",
                "false_fire_rate",
            )
        }
    primary = float(
        np.mean([m["frame_f2"] for m in per_domain.values()])
    )
    return {"per_domain": per_domain, "primary": primary}


def load_m0(run_dir: Path, device) -> tuple[nn.Module, dict, dict]:
    """The M0 checkpoint into a fresh frozen architecture, hash-checked."""
    history = json.loads((run_dir / "history.json").read_text())
    hashes = json.loads((run_dir / "hashes.json").read_text())
    checkpoint_path = run_dir / "best.pt"
    measured = sha256_file(checkpoint_path)
    if measured != hashes["checkpoint:best"]["sha256"]:
        raise RuntimeError(
            f"{checkpoint_path} hashes to {measured[:12]}…, not the run's "
            "recorded checkpoint; sensitivity curves for unknown weights are "
            "not evidence"
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("epoch") != history["best_epoch"]:
        raise RuntimeError("best.pt is not this run's selected epoch")
    model = build_mobilenet_v2(
        num_classes=len(history["class_names"]), pretrained=False
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, history, checkpoint


def run_sensitivity(config_path: Path, output_dir: Path) -> dict:
    """The registered D3 measurement (sensitivity_protocol.md), end to end."""
    from ..validate.dump_predictions import (
        build_validation_loaders,
        enforce_deployment_regime,
    )

    config = yaml.safe_load(config_path.read_text())
    run_dir = Path(config["m0_run"])
    ratios = [float(r) for r in config["ratios"]]
    round_to = int(config["round_to"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enforce_deployment_regime()
    model, history, checkpoint = load_m0(run_dir, device)
    plan = classify(model)
    loaders = build_validation_loaders(history["config"], history["class_names"])

    group_evidence = [verify_group(model, plan, b) for b in sorted(plan.expansion)]
    baseline_profile = profile(model)
    baseline = evaluate_at_yardstick(model, loaders, device)
    print(
        f"baseline: primary {baseline['primary']:.4f} "
        f"(macs {baseline_profile['macs']:,}, params {baseline_profile['params']:,})"
    )

    state = {k: v.clone() for k, v in checkpoint["model"].items()}
    groups = []
    for block in sorted(plan.expansion):
        width = plan.expansion[block].out_channels
        curve = []
        for ratio in ratios:
            fresh = build_mobilenet_v2(
                num_classes=len(history["class_names"]), pretrained=False
            ).to(device)
            fresh.load_state_dict(state)
            report = prune_expansion(
                fresh,
                {block: ratio},
                round_to=round_to,
                num_classes=len(history["class_names"]),
                verify_groups=False,  # proven once per block above
                export_check=False,  # proven per candidate in D4; 64x here buys nothing
            )
            measured = evaluate_at_yardstick(fresh, loaders, device)
            realized = report["realized"][f"features.{block}"]
            point = {
                "requested_ratio": ratio,
                "width_before": realized["width_before"],
                "width_after": realized["width_after"],
                "realized_ratio": realized["realized_ratio"],
                "macs": report["profile_after"]["macs"],
                "mac_reduction": report["mac_reduction"],
                "primary": measured["primary"],
                "delta_primary": baseline["primary"] - measured["primary"],
                "per_domain": measured["per_domain"],
            }
            curve.append(point)
            print(
                f"features.{block} @{ratio}: width {realized['width_before']}"
                f"->{realized['width_after']}, primary {measured['primary']:.4f} "
                f"(Δ {point['delta_primary']:+.4f})"
            )
            del fresh
            torch.cuda.empty_cache()
        groups.append(
            {
                "block": block,
                "conv": plan.conv_name(block),
                "width": width,
                "sensitivity_index": float(
                    np.mean([p["delta_primary"] for p in curve])
                ),
                "curve": curve,
            }
        )

    ranking = sorted(groups, key=lambda g: -g["sensitivity_index"])
    report = {
        "tool": "wildlife_trigger.optimize.prune --sensitivity",
        "protocol": "results/optimize/m3_prune/sensitivity_protocol.md",
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "m0_run": str(run_dir),
            "checkpoint_sha256": json.loads((run_dir / "hashes.json").read_text())[
                "checkpoint:best"
            ]["sha256"],
            "ratios": ratios,
            "round_to": round_to,
        },
        "environment": {
            "torch": torch.__version__,
            "torch_pruning_dist": "1.6.1",
            "device": str(device),
            "tf32_disabled": True,
        },
        "baseline": {"profile": baseline_profile, **baseline},
        "analytic_macs_reference": {
            "macs_at_256x192_16cls": 293402624,
            "note": (
                "the ladder's comparison.jsonl counts MACs analytically "
                "(input_cost.macs_at); this report's counter is "
                "torch_pruning.count_ops_and_params, used consistently before "
                "and after every mutation. The two conventions differ and are "
                "never mixed."
            ),
        },
        "group_verification": group_evidence,
        "groups": groups,
        "ranking": [
            {
                "block": g["block"],
                "conv": g["conv"],
                "width": g["width"],
                "sensitivity_index": g["sensitivity_index"],
            }
            for g in ranking
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "sensitivity.json", report)
    (output_dir / "sensitivity.md").write_text(render_markdown(report))
    print(f"wrote {output_dir / 'sensitivity.json'}")
    print(f"wrote {output_dir / 'sensitivity.md'}")
    return report


def render_markdown(report: dict) -> str:
    """The human-readable half of the evidence."""
    baseline = report["baseline"]
    lines = [
        "# D3 — expansion-group pruning sensitivity (no fine-tune)",
        "",
        f"Protocol: `{report['protocol']}` — applied mechanically.",
        "",
        f"Baseline (M0 `{report['config']['checkpoint_sha256'][:12]}…`, torch "
        f"GPU TF32-off): primary **{baseline['primary']:.4f}**, "
        f"{baseline['profile']['macs']:,} MACs / "
        f"{baseline['profile']['params']:,} params "
        "(torch-pruning counter; the ladder's analytic M0 reference is "
        f"{report['analytic_macs_reference']['macs_at_256x192_16cls']:,}).",
        "",
        "Sensitivity index = mean Δprimary over requested ratios "
        f"{report['config']['ratios']} (registered §3). Higher = more fragile.",
        "",
        "| rank | group | width | index | Δ@0.125 | Δ@0.25 | Δ@0.375 | Δ@0.5 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    by_block = {g["block"]: g for g in report["groups"]}
    for rank, entry in enumerate(report["ranking"], start=1):
        group = by_block[entry["block"]]
        deltas = {p["requested_ratio"]: p["delta_primary"] for p in group["curve"]}
        cells = " | ".join(
            f"{deltas[r]:+.4f}" for r in report["config"]["ratios"]
        )
        lines.append(
            f"| {rank} | `{entry['conv']}` | {entry['width']} | "
            f"{entry['sensitivity_index']:+.4f} | {cells} |"
        )
    lines += [
        "",
        "Full per-ratio curves (realized widths, MACs, per-domain metrics): "
        "`sensitivity.json`. D3 takes no selection decision; D4's allocation "
        "rule is registered separately before its numbers exist.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Run the registered D3 per-group sensitivity measurement.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/optimize/m3_prune")
    )
    args = parser.parse_args()

    if not args.sensitivity:
        parser.error(
            "only --sensitivity exists in D3; candidate creation is D4's task"
        )
    run_sensitivity(args.config, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
