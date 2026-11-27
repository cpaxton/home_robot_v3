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

import fcntl
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


def _to_numpy(t: Any) -> np.ndarray:
    if hasattr(t, "detach"):  # torch tensor (possibly on GPU)
        t = t.detach().cpu()
    return np.asarray(t)


def export_initial_voxel_map(vm: Any) -> dict[str, Any]:
    """Export a voxel map's occupancy as a JSON-serializable **initial map** dict.

    Captures the 2D occupancy (obstacles + explored) plus reconstruction metadata
    (grid origin + resolution) so the scenario can be reproduced. Duck-typed against
    any ``SparseVoxelMap``-like object; never raises.
    """
    out: dict[str, Any] = {"kind": "voxel_map_2d"}
    try:
        origin = getattr(vm, "grid_origin", None)
        if origin is not None:
            arr = np.asarray(origin, dtype=np.float64).reshape(-1)
            out["grid_origin"] = [float(arr[0]), float(arr[1])] if arr.size >= 2 else [0.0, 0.0]
        out["grid_resolution"] = float(getattr(vm, "grid_resolution", 0.05))
        m2d = vm.get_2d_map() if callable(getattr(vm, "get_2d_map", None)) else None
        if m2d is not None and len(m2d) >= 2:
            out["obstacles"] = (_to_numpy(m2d[0]) > 0).astype(np.uint8).tolist()
            out["explored"] = (_to_numpy(m2d[1]) > 0).astype(np.uint8).tolist()
        else:
            out["obstacles"] = []
            out["explored"] = []
    except Exception:
        out["kind"] = "voxel_map_2d_unavailable"
        out["obstacles"] = []
        out["explored"] = []
    return out


def scenario_from_attempt(
    attempt: dict[str, Any],
    *,
    context: dict[str, Any] | None = None,
    initial_voxel: Any = None,
) -> Any:
    """Build a :class:`~emet.controller.manipulation.case_library.ScenarioRecord` from an attempt.

    The goal is the EE target (position from ``last_attempt``), the robot state is the
    initial observed joint configuration (first trace sample), and ``initial_voxel`` is
    the **initial** map — pass the result of :func:`export_initial_voxel_map` (the
    agent-level recorder supplies it; empty/placeholder otherwise).
    """
    from emet.controller.manipulation.case_library import ScenarioRecord

    trace = attempt.get("trace") or []
    q0: list[float] = list(attempt.get("initial_q") or [])
    if not q0 and trace and isinstance(trace[0], dict):
        q0 = list(trace[0].get("qpos") or [])
    ctx = dict(context or {})
    goal: dict[str, Any] = {
        "kind": "ee_target",
        "op": attempt.get("op"),
        "ee_target_pos": attempt.get("target_xyz"),
        "object": ctx.get("object_query") or ctx.get("object"),
        "receptacle": ctx.get("receptacle_query") or ctx.get("receptacle"),
    }
    meta: dict[str, Any] = {
        "failure_mode": attempt.get("failure_mode") or ctx.get("failure_mode") or "",
        "source": "recorded",
        "privileged": bool(ctx.get("privileged")),
    }
    if ctx.get("privileged"):
        meta["object_body"] = ctx.get("object_body")
        meta["receptacle_body"] = ctx.get("receptacle_body")
    meta.update({
        k: v
        for k, v in ctx.items()
        if k not in (
            "op", "object_query", "receptacle_query", "object_body", "receptacle_body",
            "failure_mode", "privileged",
        )
    })
    for key in ("embodiment", "schema_version", "action_mode", "robot_id"):
        if key in ctx:
            meta[key] = ctx[key]
    robot_state = ctx.get("robot_state")
    if not isinstance(robot_state, dict):
        robot_state = {"q": q0, "joint_names": list(attempt.get("joint_names") or [])}
    return ScenarioRecord(
        initial_voxel=initial_voxel if initial_voxel is not None else {"kind": "voxel_map_2d_unavailable"},
        goal=goal,
        robot_state=robot_state,
        meta=meta,
    )


def record_scenario(record: Any, *, library_dir: str | Path | None = None) -> bool:
    """Append a scenario record to the case library (deduplicated by fingerprint)."""
    from emet.controller.manipulation.case_library import CaseLibrary

    lib_path = default_library_path(library_dir)
    lock_path = lib_path.with_name(lib_path.name + ".lock")
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            lib = CaseLibrary.load(lib_path) if lib_path.is_file() else CaseLibrary(lib_path)
            if not lib.add(record):
                return False
            lib.save(lib_path)
            return True
    except OSError:
        return False
