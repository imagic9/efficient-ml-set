"""D6: the pre-Pi shortlist and the benchmark manifest builder.

The shortlist is pure logic over comparison rows — tested against fabricated
ladders whose correct §8.5 outcome is obvious. The benchmark builder's
stratification and determinism are tested against a synthetic pool + score map,
so the seeded draw and the priority partition are proven without gx10.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from wildlife_trigger.data import benchmark_manifest as B
from wildlife_trigger.optimize import pre_pi_shortlist as S


def row(model_id, kind, primary, macs, byts, *, status="recall_floor_infeasible",
        rule_met=False, parity=True):
    return {
        "model_id": model_id,
        "kind": kind,
        "macs": macs,
        "model": {"bytes": byts},
        "validation_at_0p5": {"selection_score": primary, "cis_f2": primary * 1.7,
                              "trans_f2": primary * 0.3},
        "operating_point": {"status": status, "primary_rule_met": rule_met},
        "parity": {"passed": parity},
    }


LADDER = [
    row("M0", "fp32_baseline", 0.3663, 293_402_624, 8_950_645),
    row("M1", "int8_ptq", 0.3527, 293_402_624, 2_620_130),
    row("M2", "int8_qat", 0.3832, 293_402_624, 2_536_267),
    row("M3", "pruned_fp32", 0.3583, 205_614_080, 7_035_950),
    row("M4", "pruned_qat", 0.3730, 205_614_080, 2_014_806),
]


class TestShortlist:
    def test_the_real_ladder_shortlists_m0_m2_m4(self):
        report = S.shortlist(LADDER)
        assert set(report["shortlist"]) == {"M0", "M2", "M4"}

    def test_m1_dominated_by_m2_and_m3_by_m4(self):
        report = S.shortlist(LADDER)
        rejected = {r["model_id"]: r["dominated_by"] for r in report["rejected"]}
        assert "M2" in rejected["M1"]  # same MACs, higher F2, fewer bytes
        assert "M4" in rejected["M3"]  # same MACs, higher F2, fewer bytes

    def test_fallback_branch_when_no_recall_rule_met(self):
        report = S.shortlist(LADDER)
        assert report["recall_rule"]["fallback_branch"] is True
        assert report["recall_rule"]["met_by"] == []

    def test_recall_rule_met_is_recorded(self):
        ladder = [row("M0", "fp32_baseline", 0.4, 293_402_624, 8_950_645, rule_met=True)]
        report = S.shortlist(ladder)
        assert report["recall_rule"]["fallback_branch"] is False
        assert report["recall_rule"]["met_by"] == ["M0"]

    def test_m0_is_kept_even_if_optimized_beat_it(self):
        report = S.shortlist(LADDER)
        assert "M0" in report["shortlist"]  # baseline never competes away

    def test_latency_is_not_used_to_rank(self):
        report = S.shortlist(LADDER)
        assert report["latency_ranking_used"] is False

    def test_a_failed_gate_row_is_refused(self):
        bad = LADDER + [row("MX", "int8_ptq", 0.9, 1, 1, parity=False)]
        with pytest.raises(RuntimeError, match="failed parity gate"):
            S.shortlist(bad)

    def test_markdown_names_shortlist_and_rejections(self):
        md = S.render_markdown(S.shortlist(LADDER))
        assert "M0 · M2 · M4" in md
        assert "M1" in md and "M3" in md  # rejections listed
        assert "dominated by" in md

    def test_dominance_needs_no_axis_worse(self):
        # a candidate with better F2 but larger size and MACs does NOT dominate
        a = S.row_metrics(row("A", "k", 0.40, 300, 300))
        b = S.row_metrics(row("B", "k", 0.35, 200, 200))
        assert not S.dominates(a, b)
        assert not S.dominates(b, a)  # b cheaper but lower F2 — mutually non-dominated


class TestBenchmarkManifest:
    @pytest.fixture()
    def world(self, tmp_path):
        """A synthetic validation pool + M0 score map + policy."""
        manifests = tmp_path / "manifests"
        manifests.mkdir()

        rng = np.random.default_rng(0)
        threshold = 0.5
        pool = {"cis_val_clean": [], "trans_val": []}
        scores = {"class_names": np.array(["empty", "bobcat"])}
        cis_ids, cis_probs = [], []
        trans_ids, trans_probs = [], []

        def make(split, i, labels, score, multi=False, w=1024, h=747):
            iid = f"{split}_{i:04d}"
            rec = {
                "image_id": iid, "file_name": f"{iid}.jpg", "labels": labels,
                "seq_id": f"{split}_seq{i // 3}", "multi_class": multi,
                "observed_width": w, "observed_height": h,
            }
            pool[split].append(rec)
            (cis_ids if split == "cis_val_clean" else trans_ids).append(iid)
            (cis_probs if split == "cis_val_clean" else trans_probs).append(score)

        # 60 near-threshold, 200 bobcat, 30 multi, 40 rare, 20 edge, 400 empty,
        # 250 other — enough to exceed 1000 and exercise fill.
        n = 0
        for _ in range(60):
            make("trans_val", n, ["bobcat"], 0.5 + rng.uniform(-0.05, 0.05)); n += 1
        for _ in range(200):
            make("trans_val", n, ["bobcat"], 0.9); n += 1
        for _ in range(30):
            make("trans_val", n, ["bobcat", "coyote"], 0.95, multi=True); n += 1
        for _ in range(40):
            make("cis_val_clean", n, ["bird"], 0.9); n += 1
        for _ in range(20):
            make("cis_val_clean", n, ["squirrel"], 0.9, w=500, h=500); n += 1
        for _ in range(400):
            make("cis_val_clean", n, ["empty"], 0.01); n += 1
        for _ in range(400):
            make("cis_val_clean", n, ["opossum"], 0.02); n += 1

        for split in ("cis_val_clean", "trans_val"):
            (manifests / f"{split}.jsonl").write_text(
                "".join(json.dumps(r, sort_keys=True) + "\n" for r in pool[split])
            )
        scores["cis_val_clean/probabilities"] = np.array(
            [[1 - p, p] for p in cis_probs], dtype=np.float32
        )
        scores["cis_val_clean/image_ids"] = np.array(cis_ids)
        scores["trans_val/probabilities"] = np.array(
            [[1 - p, p] for p in trans_probs], dtype=np.float32
        )
        scores["trans_val/image_ids"] = np.array(trans_ids)
        npz = tmp_path / "m0.npz"
        np.savez(npz, **scores)

        policy = tmp_path / "policy.json"
        policy.write_text(json.dumps(
            {"targets": [{"class": "bobcat", "threshold": threshold}]}
        ))
        return manifests, npz, policy, tmp_path

    def test_builds_exactly_1000_frames(self, world):
        manifests, npz, policy, tmp = world
        out = tmp / "benchmark_val_1000.jsonl"
        prov = B.build(manifests, npz, policy, out, eps=0.1, seed=42)
        assert prov["frames"] == 1000
        lines = out.read_text().splitlines()
        assert len(lines) == 1000

    def test_threshold_adjacent_is_over_sampled(self, world):
        manifests, npz, policy, tmp = world
        out = tmp / "b.jsonl"
        prov = B.build(manifests, npz, policy, out, eps=0.1, seed=42)
        acc = prov["accounting"]
        # natural ~60/1130; benchmark fraction must be materially higher
        assert acc["threshold_adjacent_benchmark_fraction"] > \
            acc["threshold_adjacent_natural_fraction"]
        # all near-threshold frames included (take-all)
        taken = acc["strata_priority_partition"]["threshold_adjacent"]
        assert taken["taken"] == taken["available"]

    def test_is_deterministic(self, world):
        manifests, npz, policy, tmp = world
        a = B.build(manifests, npz, policy, tmp / "a.jsonl", eps=0.1, seed=42)
        b = B.build(manifests, npz, policy, tmp / "b.jsonl", eps=0.1, seed=42)
        assert a["output_sha256"] == b["output_sha256"]

    def test_validation_only_records_carry_split_and_stratum(self, world):
        manifests, npz, policy, tmp = world
        out = tmp / "b.jsonl"
        B.build(manifests, npz, policy, out, eps=0.1, seed=42)
        for line in out.read_text().splitlines():
            rec = json.loads(line)
            assert rec["source_split"] in ("cis_val_clean", "trans_val")
            assert rec["benchmark_stratum"] in B.STRATA
            assert "m0_bobcat_score" in rec

    def test_missing_scores_are_refused(self, world):
        manifests, npz, policy, tmp = world
        # drop a score row by rewriting the npz without one image
        data = dict(np.load(npz, allow_pickle=True))
        data["cis_val_clean/image_ids"] = data["cis_val_clean/image_ids"][:-1]
        data["cis_val_clean/probabilities"] = data["cis_val_clean/probabilities"][:-1]
        np.savez(npz, **data)
        with pytest.raises(RuntimeError, match="no M0 score"):
            B.build(manifests, npz, policy, tmp / "b.jsonl", eps=0.1, seed=42)
