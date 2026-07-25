# Copyright (c) Chris Paxton 2026

"""Metrics for autonomous hidden-relocation detection and stale-memory decay."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _distance(a: Any, b: Any) -> float:
    if a is None or b is None:
        return float("inf")
    av = np.asarray(a, dtype=float).reshape(-1)
    bv = np.asarray(b, dtype=float).reshape(-1)
    n = min(av.size, bv.size, 3)
    return float(np.linalg.norm(av[:n] - bv[:n])) if n else float("inf")


def score_hidden_relocations(
    events: list[dict[str, Any]],
    moves: list[dict[str, Any]],
    *,
    match_radius_m: float = 0.75,
) -> dict[str, Any]:
    """Match autonomous events to hidden GT moves; GT never modifies policy state."""
    move_matches: set[int] = set()
    event_matches: set[int] = set()
    delays: list[float] = []
    relocation_errors: list[float] = []
    for event_index, event in enumerate(events):
        event_old = event.get("from_xyz", event.get("last_xyz"))
        event_new = event.get("to_xyz")
        best: tuple[float, int] | None = None
        for move_index, move in enumerate(moves):
            old_distance = _distance(event_old, move.get("old_pos"))
            if old_distance > match_radius_m:
                continue
            new_distance = (
                _distance(event_new, move.get("verified_pos", move.get("pos")))
                if event_new is not None
                else 0.0
            )
            score = old_distance + new_distance
            if best is None or score < best[0]:
                best = (score, move_index)
        if best is None:
            continue
        move_index = best[1]
        move_matches.add(move_index)
        event_matches.add(event_index)
        move = moves[move_index]
        if event.get("step") is not None and move.get("step") is not None:
            delays.append(max(0.0, float(event["step"]) - float(move["step"])))
        if event_new is not None:
            relocation_errors.append(
                _distance(event_new, move.get("verified_pos", move.get("pos")))
            )
    precision = len(event_matches) / len(events) if events else 0.0
    recall = len(move_matches) / len(moves) if moves else 0.0
    return {
        "n_moves": len(moves),
        "n_events": len(events),
        "true_change_events": len(event_matches),
        "false_invalidations": len(events) - len(event_matches),
        "missed_relocations": len(moves) - len(move_matches),
        "detection_precision": precision,
        "detection_recall": recall,
        "mean_detection_delay_steps": (
            sum(delays) / len(delays) if delays else None
        ),
        "mean_relocation_error_m": (
            sum(relocation_errors) / len(relocation_errors)
            if relocation_errors
            else None
        ),
    }


def stale_memory_half_life(
    stale_counts: list[int],
    *,
    steps: list[int] | None = None,
) -> float | None:
    """First interpolated step where stale count falls to half its initial value."""
    if not stale_counts or stale_counts[0] <= 0:
        return None
    x = steps or list(range(len(stale_counts)))
    target = float(stale_counts[0]) / 2.0
    for index in range(1, len(stale_counts)):
        before, after = float(stale_counts[index - 1]), float(stale_counts[index])
        if after > target:
            continue
        if before == after:
            return float(x[index])
        alpha = (before - target) / (before - after)
        return float(x[index - 1]) + alpha * (
            float(x[index]) - float(x[index - 1])
        )
    return math.inf


def change_conditioned_answer_accuracy(
    rows: list[dict[str, Any]],
) -> float | None:
    changed = [row for row in rows if row.get("change_expected")]
    if not changed:
        return None
    return sum(bool(row.get("answer_correct")) for row in changed) / len(changed)
