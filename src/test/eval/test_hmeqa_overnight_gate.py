# Copyright (c) Chris Paxton 2026
"""Unit tests for overnight holdout gate (no GPU)."""

from __future__ import annotations

from emet.eval.harness import (
    DEFAULT_HOLDOUT8_IDS,
    PAPER_ROUTER_PRESET,
    apply_paper_router_preset,
    evaluate_holdout_gate,
)


def test_holdout8_ids_are_eight():
    ids = [x for x in DEFAULT_HOLDOUT8_IDS.split(",") if x.strip()]
    assert len(ids) == 8
    assert ids[0] == "15"


def test_gate_ok_when_agentic_competitive():
    gate = evaluate_holdout_gate(
        {
            "classic": {"accuracy": 0.5, "n": 8, "correct": 4},
            "agentic": {"accuracy": 0.5, "n": 8, "correct": 4},
        }
    )
    assert gate["need_retune"] is False
    assert gate["reason"] == "ok"
    assert gate["proceed_bal32"] is True


def test_gate_retune_under_scored():
    gate = evaluate_holdout_gate(
        {
            "classic": {"accuracy": 0.5, "n": 8, "correct": 4},
            "agentic": {"accuracy": 0.5, "n": 3, "correct": 1},
        }
    )
    assert gate["need_retune"] is True
    assert "under-scored" in gate["reason"]


def test_gate_retune_trails_classic():
    gate = evaluate_holdout_gate(
        {
            "classic": {"accuracy": 0.625, "n": 8, "correct": 5},
            "agentic": {"accuracy": 0.25, "n": 8, "correct": 2},
        }
    )
    assert gate["need_retune"] is True
    assert "<<" in gate["reason"]


def test_gate_retune_below_min_acc():
    gate = evaluate_holdout_gate(
        {
            "classic": {"accuracy": 0.25, "n": 8, "correct": 2},
            "agentic": {"accuracy": 0.125, "n": 8, "correct": 1},
        },
        min_agentic_acc=0.25,
    )
    assert gate["need_retune"] is True
    assert "< min" in gate["reason"]


def test_paper_router_preset_applies_on_defaults():
    v, rv, ar = apply_paper_router_preset(
        agentic_verifier="none",
        require_verified=True,
        agentic_router=False,
        verifier_source="DEFAULT",
        verified_source="DEFAULT",
        router_source="DEFAULT",
    )
    assert v == PAPER_ROUTER_PRESET["agentic_verifier"]
    assert rv is False
    assert ar is True


def test_paper_router_preset_keeps_explicit_flags():
    v, rv, ar = apply_paper_router_preset(
        agentic_verifier="yoloe",
        require_verified=True,
        agentic_router=False,
        verifier_source="COMMANDLINE",
        verified_source="COMMANDLINE",
        router_source="COMMANDLINE",
    )
    assert v == "yoloe"
    assert rv is True
    assert ar is False
