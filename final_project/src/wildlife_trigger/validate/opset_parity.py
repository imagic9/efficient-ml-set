#!/usr/bin/env python3
"""Confirm every model form carries the same P0-accepted opset.

DESIGN §8 requires one export opset across M0-M4 so that a measured difference
between candidates is a property of the candidate rather than of the opset it was
exported at. The export path guards the *request*; this checks the *artifacts*,
after PTQ and QAT have each rewritten the graph. ORT's quantizer is entitled to
add a domain or bump a version, and the point of a contract is that it is verified
where it could break rather than where it was typed.

Additional domains (com.microsoft after quantization) are reported, not rejected:
the contract is about the default ONNX domain. Reporting them keeps the artifact's
real dependencies visible.

Usage:
    python -m wildlife_trigger.validate.opset_parity --models a.onnx b.onnx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx

from wildlife_trigger.models.export import DEFAULT_DOMAIN, P0_OPSET, graph_opsets


def check(models: list[Path], expected: int = P0_OPSET) -> dict:
    per_model = {}
    for path in models:
        opsets = graph_opsets(onnx.load(str(path), load_external_data=False))
        per_model[path.name] = {
            "default_domain_opset": opsets.get(DEFAULT_DOMAIN),
            "all_domains": opsets,
            "matches_contract": opsets.get(DEFAULT_DOMAIN) == expected,
        }

    observed = {info["default_domain_opset"] for info in per_model.values()}
    return {
        "expected_opset": expected,
        "models": per_model,
        "all_match_contract": all(i["matches_contract"] for i in per_model.values()),
        "observed_default_opsets": sorted(o for o in observed if o is not None),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, type=Path, nargs="+")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    result = check(args.models)
    for name, info in result["models"].items():
        mark = "ok" if info["matches_contract"] else "MISMATCH"
        print(f"    {name:24s} opset {info['default_domain_opset']}  {mark}")
        extra = {d: v for d, v in info["all_domains"].items() if d != DEFAULT_DOMAIN}
        if extra:
            print(f"      additional domains: {extra}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")

    return 0 if result["all_match_contract"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
