# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Opt-in simulator actuation audit. Fixture setup precedes scored execution.

This guard is enforced at action dispatch and at state-writing simulator seams;
environment settings alone are not evidence that a trajectory was physical.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

FORBIDDEN_ACTIONS = frozenset(
    {
        "sim_set_body_pose",
        "sim_set_joint_qpos",
        "sim_attach_body",
        "sim_detach_body",
        "teleport_base",
        "teleport_body",
        "set_joint",
        "reset",
        "reset_simulation",
        "stationary_base_pose_hold",
    }
)


def physical_execution_enabled() -> bool:
    return os.environ.get("EMET_PHYSICAL_EXECUTION", "0") == "1"


def audit_physical_action(action: dict, *, source: str, implicit_teleport: bool = False) -> None:
    """Reject the entire action before applying any part of a forbidden command."""
    if not physical_execution_enabled():
        return
    path = os.environ.get("EMET_PHYSICAL_AUDIT")
    if not path:
        raise RuntimeError("physical_execution_requires_audit_path")
    forbidden = sorted(FORBIDDEN_ACTIONS.intersection(action))
    if bool(action.get("nav_teleport", False)) or implicit_teleport:
        forbidden.append("nav_teleport")
    row = {
        "wall_time": time.time(),
        "pid": os.getpid(),
        "source": source,
        "action_keys": sorted(action),
        "forbidden": forbidden,
        "accepted": not forbidden,
    }
    # O_APPEND prevents independent server processes overwriting each other's records.
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
    if forbidden:
        raise RuntimeError("forbidden_physical_actuation:" + ",".join(forbidden))
