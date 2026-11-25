# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Persist manipulation attempts for offline RL-WBC training replay.

A failed (or successful) ``last_attempt`` carries the EE target, the observed
EE + joint trajectory, the planned waypoints, and the failure mode. Appending it
here gives the RL-WBC data pipeline a stable, replayable record of the exact cases
the kinematic controller struggles with (tracking lags, IK failures, joint bounds).

Set ``EMET_MANIP_FAILURE_DIR`` to redirect the sink (default ``~/runs/emet/manip_failures``).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


def default_failure_dir() -> Path:
    root = os.environ.get("EMET_MANIP_FAILURE_DIR")
    if root:
        return Path(root).expanduser()
    return Path(os.path.expanduser("~/runs/emet/manip_failures"))


def default_library_path(library_dir: str | Path | None = None) -> Path:
    root = library_dir or os.environ.get("EMET_CASE_LIBRARY_DIR") or os.path.expanduser("~/runs/emet/case_library")
    return Path(root).expanduser() / "cases.jsonl"


def record_manip_attempt(attempt: dict[str, Any], *, context: dict[str, Any] | None = None) -> Path | None:
    """Append one manipulation attempt to the training replay sink (JSONL)."""
    if not isinstance(attempt, dict) or not attempt:
        return None
    record: dict[str, Any] = {
        "ts": time.time(),
        "context": dict(context or {}),
        "attempt": attempt,
    }
    out_dir = default_failure_dir()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "manip_failures.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, allow_nan=False, default=str) + "\n")
        return path
    except OSError:
        return None


def scenario_from_attempt(
    attempt: dict[str, Any],
    *,
    context: dict[str, Any] | None = None,
    voxel: np.ndarray | None = None,
) -> Any:
    """Build a :class:`~emet.controller.manipulation.case_library.ScenarioRecord` from an attempt.

    The goal is the EE target (position from ``last_attempt``), the robot state is the
    initial observed joint configuration (first trace sample), and ``voxel`` is the local
    occupancy crop when available (a hook for the agent-level recorder; empty otherwise).
    """
    from emet.controller.manipulation.case_library import ScenarioRecord

    trace = attempt.get("trace") or []
    q0: list[float] = []
    if trace and isinstance(trace[0], dict):
        q0 = list(trace[0].get("qpos") or [])
    ctx = dict(context or {})
    goal: dict[str, Any] = {
        "kind": "ee_target",
        "ee_target_pos": attempt.get("target_xyz"),
        "object": ctx.get("object_body"),
        "receptacle": ctx.get("receptacle_body"),
    }
    meta: dict[str, Any] = {
        "failure_mode": attempt.get("failure_mode") or ctx.get("failure_mode") or "",
        "source": "recorded",
    }
    meta.update({k: v for k, v in ctx.items() if k not in ("op", "object_body", "receptacle_body", "failure_mode")})
    return ScenarioRecord(
        voxel=np.asarray(voxel, dtype=np.uint8) if voxel is not None else np.zeros((0, 0, 0), dtype=np.uint8),
        goal=goal,
        robot_state={"q": q0, "joint_names": list(attempt.get("joint_names") or [])},
        meta=meta,
    )


def record_scenario(record: Any, *, library_dir: str | Path | None = None) -> bool:
    """Append a scenario record to the case library (deduplicated by fingerprint)."""
    from emet.controller.manipulation.case_library import CaseLibrary

    lib_path = default_library_path(library_dir)
    lib = CaseLibrary.load(lib_path) if lib_path.is_file() else CaseLibrary(lib_path)
    if not lib.add(record):
        return False
    try:
        lib.save(lib_path)
        return True
    except OSError:
        return False
