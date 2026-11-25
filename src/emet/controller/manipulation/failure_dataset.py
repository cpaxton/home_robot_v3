# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Load recorded manipulation attempts into calibration and training samples.

The execution sink (``failure_record.py``) writes one JSON object per failed
attempt with the planned joint waypoints and the observed EE/joint trace. This
module is the reader side of that loop: it validates records, exposes the
steady-state target/observed pair, and is deliberately embodiment-agnostic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

TRACKING_FAILURES = frozenset({"tracking_failed", "tracking_timeout"})


@dataclass(frozen=True)
class ManipulationAttempt:
    """One recorded attempt with the fields the calibration loop needs."""

    context: dict[str, Any]
    attempt: dict[str, Any]

    @property
    def robot_id(self) -> str:
        return str(self.context.get("robot_id") or self.context.get("embodiment") or "")

    @property
    def failure_mode(self) -> str:
        return str(self.attempt.get("failure_mode") or self.context.get("failure_mode") or "")

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(str(name) for name in (self.attempt.get("joint_names") or ()))

    @property
    def planned_waypoints(self) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for waypoint in self.attempt.get("planned_waypoints") or ():
            try:
                arr = np.asarray(waypoint, dtype=np.float64).reshape(-1)
            except (TypeError, ValueError):
                continue
            if arr.size and np.isfinite(arr).all():
                out.append(arr)
        return out

    @property
    def trace(self) -> list[dict[str, Any]]:
        return [sample for sample in (self.attempt.get("trace") or ()) if isinstance(sample, dict)]

    def final_target_q(self) -> np.ndarray | None:
        waypoints = self.planned_waypoints
        if not waypoints:
            return None
        target = waypoints[-1]
        if target.size != len(self.joint_names):
            return None
        return target

    def steady_state_q(self, *, window: int = 5) -> np.ndarray | None:
        """Median of the last ``window`` observed joint vectors."""
        if window < 1 or len(self.joint_names) == 0:
            return None
        samples: list[np.ndarray] = []
        for sample in self.trace:
            qpos = sample.get("qpos")
            if not isinstance(qpos, (list, tuple)) or len(qpos) != len(self.joint_names):
                continue
            arr = np.asarray(qpos, dtype=np.float64)
            if arr.size and np.isfinite(arr).all():
                samples.append(arr)
        if not samples:
            return None
        return np.median(np.asarray(samples[-window:]), axis=0)

    def joint_bias(self, *, window: int = 5) -> dict[str, float] | None:
        """Steady-state droop per joint: final target minus observed.

        A positive value means the joint settled short of its commanded target
        (gravity/load droop); a negative value means it overshot.
        """
        target = self.final_target_q()
        observed = self.steady_state_q(window=window)
        if target is None or observed is None:
            return None
        return {name: float(delta) for name, delta in zip(self.joint_names, target - observed, strict=True)}


def load_manipulation_attempts(path: str | Path) -> list[ManipulationAttempt]:
    """Read a JSONL sink (or a directory containing ``manip_failures.jsonl``)."""
    source = Path(path).expanduser()
    if source.is_dir():
        source = source / "manip_failures.jsonl"
    if not source.is_file():
        raise FileNotFoundError(f"manipulation record not found: {source}")

    attempts: list[ManipulationAttempt] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        attempt = record.get("attempt")
        context = record.get("context")
        if not isinstance(attempt, dict) or not attempt:
            continue
        attempts.append(ManipulationAttempt(context=dict(context or {}), attempt=attempt))
    return attempts


def select_attempts(
    attempts: Iterable[ManipulationAttempt],
    *,
    robot_id: str = "",
    failure_modes: Iterable[str] = TRACKING_FAILURES,
    require_waypoints: bool = True,
    exclude_feedforward: bool = True,
) -> list[ManipulationAttempt]:
    """Filter to the records usable for tracking calibration."""
    allowed = {str(mode) for mode in failure_modes}
    selected: list[ManipulationAttempt] = []
    for attempt in attempts:
        if robot_id and attempt.robot_id and attempt.robot_id != robot_id:
            continue
        if allowed and attempt.failure_mode not in allowed:
            continue
        if require_waypoints and not attempt.planned_waypoints:
            continue
        if exclude_feedforward and attempt.context.get("feedforward") is True:
            continue
        selected.append(attempt)
    return selected
