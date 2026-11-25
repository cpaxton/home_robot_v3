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


def default_failure_dir() -> Path:
    root = os.environ.get("EMET_MANIP_FAILURE_DIR")
    if root:
        return Path(root).expanduser()
    return Path(os.path.expanduser("~/runs/emet/manip_failures"))


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
