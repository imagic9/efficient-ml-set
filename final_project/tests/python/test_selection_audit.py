"""The audit that closes PLAN C2's fourth bullet.

The point of the tool is to disagree when the engine and the rule disagree, so these
tests mostly feed it histories where they do. A checker that only ever says yes is the
thing it was written to replace.
"""

from __future__ import annotations

import json
from pathlib import Path

from wildlife_trigger.validate.selection_audit import audit, replay


def entry(epoch: int, phase: str, primary: float, recall: float, macro: float) -> dict:
    return {
        "epoch": epoch,
        "phase": phase,
        "selection_score": {
            "primary": primary,
            "tiebreak_1_sequence_balanced_recall": recall,
            "tiebreak_2_macro_f1": macro,
        },
    }


def write_run(tmp_path: Path, history: list[dict], best_epoch: int) -> Path:
    run = tmp_path / "run"
    run.mkdir(parents=True, exist_ok=True)
    best = next((e for e in history if e["epoch"] == best_epoch), None)
    (run / "history.json").write_text(
        json.dumps(
            {
                "run_name": "t",
                "best_epoch": best_epoch,
                "best_selection_score": best["selection_score"] if best else None,
                "history": history,
            }
        )
    )
    return run


def test_phase_a_epochs_are_never_selected() -> None:
    """A head-only model that scores well early must not win the run.

    The backbone has not moved yet, so selecting it would throw away the whole of phase
    B — and phase A's score is often the best-looking one on a small validation set.
    """
    history = [
        entry(0, "A", 0.99, 0.99, 0.99),  # would win on every level of the ladder
        entry(1, "B", 0.40, 0.60, 0.50),
        entry(2, "B", 0.41, 0.60, 0.50),
    ]
    assert replay(history)["epoch"] == 2


def test_replay_applies_the_tiebreaks_the_way_train_does() -> None:
    """Same comparator, same order, so the audit cannot pass a run the rule would fail."""
    history = [
        entry(1, "B", 0.42, 0.61, 0.50),
        entry(2, "B", 0.42, 0.62, 0.10),  # ties on primary, wins on recall
        entry(3, "B", 0.42, 0.62, 0.09),  # ties on both, loses on macro F1
    ]
    assert replay(history)["epoch"] == 2


def test_the_audit_agrees_with_a_correctly_selected_run(tmp_path: Path) -> None:
    history = [entry(0, "A", 0.9, 0.9, 0.9), entry(1, "B", 0.40, 0.6, 0.5), entry(2, "B", 0.45, 0.6, 0.5)]
    report = audit(write_run(tmp_path, history, best_epoch=2))

    assert report["agrees"] is True
    assert report["problems"] == []
    assert report["rule_selects_epoch"] == 2
    assert report["phase_b_epochs"] == 2


def test_the_audit_catches_a_summary_that_ignored_a_tiebreak(tmp_path: Path) -> None:
    """Issue #12's exact failure, replayed: equal primary, and the earlier epoch kept.

    A run produced by the old code would look like this — and would have looked correct
    from every other artifact in the directory.
    """
    history = [
        entry(1, "B", 0.4280, 0.61, 0.50),
        entry(2, "B", 0.4280, 0.62, 0.50),
    ]
    report = audit(write_run(tmp_path, history, best_epoch=1))

    assert report["agrees"] is False
    assert report["rule_selects_epoch"] == 2
    assert any("selects epoch 2" in p for p in report["problems"])


def test_the_audit_catches_a_checkpoint_from_another_run(tmp_path: Path) -> None:
    """best.pt overwritten by a later run into the same directory.

    Every number downstream would then belong to a different model, and nothing else in
    the directory would show it.
    """
    import pytest

    torch = pytest.importorskip("torch")

    history = [entry(1, "B", 0.40, 0.6, 0.5), entry(2, "B", 0.45, 0.6, 0.5)]
    run = write_run(tmp_path, history, best_epoch=2)
    torch.save({"epoch": 1, "phase": "B", "model": {}}, run / "best.pt")

    report = audit(run)

    assert report["agrees"] is False
    assert report["checkpoint_epoch"] == 1
    assert any("not this run's selected model" in p for p in report["problems"])
