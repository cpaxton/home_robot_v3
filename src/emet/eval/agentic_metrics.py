# Copyright (c) Chris Paxton 2026

"""Selective-risk and evidence-policy metrics for HM-EQA validation ladders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        row
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
        if isinstance(row, dict)
    ]


def episode_policy_metrics(
    metrics: dict[str, Any],
    trace: list[dict[str, Any]],
) -> dict[str, Any]:
    def _is_tool_result(row: dict[str, Any], name: str) -> bool:
        # tool_pick events also set tool=<name>; only count executed tool rows.
        return row.get("tool") == name and row.get("event") != "tool_pick"

    verifies = [row for row in trace if _is_tool_result(row, "verify_siglip")]
    submit = next(
        (row for row in reversed(trace) if _is_tool_result(row, "submit_answer")),
        None,
    )
    abstained = any(_is_tool_result(row, "abstain_unverified") for row in trace)
    fused = [row for row in verifies if row.get("fused_verified")]
    accepted = bool(submit and submit.get("verified") and submit.get("answerable", True) and not abstained)
    correct = metrics.get("correct")
    visible_labels = [bool(row["gt_in_view"]) for row in verifies if row.get("gt_in_view") is not None]
    false_confirmations = sum(bool(row.get("fused_verified")) and row.get("gt_in_view") is False for row in verifies)
    nav_rows = [
        row for row in trace if _is_tool_result(row, "navigate_to_obs") or _is_tool_result(row, "explore_frontier")
    ]
    path_length = sum(float(row.get("nav_dist_m") or 0.0) for row in nav_rows)
    hypotheses = [
        hypothesis
        for row in trace
        if _is_tool_result(row, "inspect_graph")
        for hypothesis in (row.get("hypotheses") or [])
    ]
    gains = [
        float(hypothesis.get("answerability_gain", 0.0)) + float(hypothesis.get("belief_reduction", 0.0))
        for hypothesis in hypotheses
    ]
    forced_submit = bool(
        submit is not None and not abstained and (not submit.get("verified") or submit.get("answerable") is False)
    )
    return {
        "question_id": metrics.get("question_id"),
        "scene": metrics.get("scene"),
        "correct": correct,
        "accepted": accepted,
        "abstained": abstained,
        "verified_answer": bool(accepted and fused),
        "forced_submit": forced_submit,
        "target_visible_at_any_verify": any(visible_labels) if visible_labels else None,
        "answerable_at_submit": submit.get("answerable") if submit else None,
        "n_verifies": len(verifies),
        "n_fused_verifies": len(fused),
        "false_confirmations": false_confirmations,
        "navigation_steps": len(nav_rows),
        "path_length_m": path_length,
        "hypotheses_tried": len({row.get("obs_id") for row in verifies if row.get("obs_id") is not None}),
        "mean_candidate_information_gain": (sum(gains) / len(gains) if gains else None),
    }


def summarize_policy_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labeled = [row for row in rows if row.get("correct") is not None]
    accepted = [row for row in labeled if row.get("accepted")]
    verified = [row for row in rows if row.get("verified_answer")]
    visible = [row for row in rows if row.get("target_visible_at_any_verify") is not None]
    false_confirmations = sum(int(row.get("false_confirmations") or 0) for row in rows)
    total_fused = sum(int(row.get("n_fused_verifies") or 0) for row in rows)
    return {
        "n_episodes": len(rows),
        "answer_accuracy": (sum(bool(row["correct"]) for row in labeled) / len(labeled) if labeled else None),
        "coverage": len(accepted) / len(labeled) if labeled else None,
        "selective_accuracy": (sum(bool(row["correct"]) for row in accepted) / len(accepted) if accepted else None),
        "selective_risk": (1.0 - sum(bool(row["correct"]) for row in accepted) / len(accepted) if accepted else None),
        "verified_answer_rate": len(verified) / len(rows) if rows else 0.0,
        "verified_precision": ((total_fused - false_confirmations) / total_fused if total_fused else None),
        "target_visibility_at_verify": (
            sum(bool(row["target_visible_at_any_verify"]) for row in visible) / len(visible) if visible else None
        ),
        "abstention_rate": (sum(bool(row.get("abstained")) for row in rows) / len(rows) if rows else 0.0),
        "false_confirmation_rate": (false_confirmations / total_fused if total_fused else None),
        "forced_submits": sum(bool(row.get("forced_submit")) for row in rows),
        "mean_navigation_steps": (
            sum(float(row.get("navigation_steps", 0)) for row in rows) / len(rows) if rows else 0.0
        ),
        "mean_path_length_m": (sum(float(row.get("path_length_m", 0.0)) for row in rows) / len(rows) if rows else 0.0),
        "mean_hypotheses_tried": (
            sum(float(row.get("hypotheses_tried", 0)) for row in rows) / len(rows) if rows else 0.0
        ),
    }


def summarize_run(root: str | Path) -> dict[str, Any]:
    root_p = Path(root).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(root_p.rglob("metrics.json")):
        episode_dir = metrics_path.parent
        trace = _jsonl(episode_dir / "agentic_trace.jsonl")
        if trace:
            rows.append(episode_policy_metrics(_json(metrics_path), trace))
    return {
        "root": str(root_p),
        "episodes": rows,
        "summary": summarize_policy_metrics(rows),
    }


def balanced32_gate(report: dict[str, Any]) -> tuple[bool, list[str]]:
    summary = report.get("summary") or {}
    reasons: list[str] = []
    if float(summary.get("verified_answer_rate") or 0.0) <= 0.0:
        reasons.append("verified-answer rate is zero")
    if int(summary.get("forced_submits") or 0) != 0:
        reasons.append("forced submits are nonzero")
    if int(summary.get("n_episodes") or 0) < 4:
        reasons.append("fewer than four probe episodes")
    return not reasons, reasons
