# Copyright (c) Chris Paxton 2026

from emet.eval.agentic_metrics import (
    balanced32_gate,
    episode_policy_metrics,
    summarize_policy_metrics,
)


def test_selective_metrics_and_balanced_gate():
    episode = episode_policy_metrics(
        {"question_id": 1, "scene": "s", "correct": True},
        [
            {
                "tool": "inspect_graph",
                "hypotheses": [{"answerability_gain": 0.8, "belief_reduction": 0.2}],
            },
            {
                "tool": "verify_siglip",
                "obs_id": 4,
                "fused_verified": True,
                "gt_in_view": True,
            },
            {
                "tool": "submit_answer",
                "verified": True,
                "answerable": True,
            },
        ],
    )
    summary = summarize_policy_metrics([episode, episode, episode, episode])
    assert summary["coverage"] == 1.0
    assert summary["selective_risk"] == 0.0
    assert summary["verified_precision"] == 1.0
    passed, reasons = balanced32_gate({"summary": summary})
    assert passed is True
    assert reasons == []


def test_tool_pick_submit_is_not_forced_submit():
    """Abstain after a fallback submit pick must not count as a forced letter."""
    episode = episode_policy_metrics(
        {"question_id": 17, "scene": "s", "correct": False},
        [
            {"event": "tool_pick", "tool": "submit_answer", "picked_by": "fallback"},
            {"tool": "abstain_unverified", "reason": "require_verified"},
        ],
    )
    assert episode["forced_submit"] is False
    assert episode["abstained"] is True
    assert episode["accepted"] is False


def test_unanswerable_verified_submit_is_forced():
    episode = episode_policy_metrics(
        {"question_id": 12, "scene": "s", "correct": False},
        [
            {"tool": "verify_siglip", "obs_id": 1, "fused_verified": True},
            {
                "tool": "submit_answer",
                "verified": True,
                "answerable": False,
                "final_answer": "B",
            },
        ],
    )
    assert episode["forced_submit"] is True
    assert episode["accepted"] is False


def test_gate_rejects_forced_or_zero_verified():
    summary = summarize_policy_metrics(
        [
            {
                "correct": False,
                "accepted": False,
                "verified_answer": False,
                "forced_submit": True,
            }
            for _ in range(4)
        ]
    )
    passed, reasons = balanced32_gate({"summary": summary})
    assert passed is False
    assert "verified-answer rate is zero" in reasons
    assert "forced submits are nonzero" in reasons
