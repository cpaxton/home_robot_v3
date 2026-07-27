# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Detect when a MuJoCo robot base has tipped / fallen over.

Uses the base body's world orientation (``xmat``): upright when body +Z aligns with
world +Z. Logs a **red** error once on the upright→fallen transition (and optionally
repeats while still down) so Robocasa / MolmoSpaces tip-overs are obvious in the
sim terminal instead of looking like a mysterious nav/mapping failure.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from emet.utils.logger import Logger

logger = Logger(__name__)

# ~55° from upright: bumps OK, on its side / back is a clear fall.
DEFAULT_MAX_TILT_DEG = 55.0
# Ignore the first moments after spawn (autoplace / stabilize can briefly tip).
DEFAULT_MIN_SIM_TIME_S = 0.75
# While still fallen, re-print at most this often so a scrolled terminal still notices.
DEFAULT_REPEAT_INTERVAL_S = 5.0


def max_tilt_deg_from_env(default: float = DEFAULT_MAX_TILT_DEG) -> float:
    """Optional ``EMET_SIM_FALL_TILT_DEG`` override (degrees from upright)."""
    raw = os.environ.get("EMET_SIM_FALL_TILT_DEG", "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


@dataclass(frozen=True)
class BaseUprightStatus:
    """Snapshot of base orientation relative to world up."""

    found: bool
    upright: bool
    tilt_deg: float
    up_dot_z: float
    base_xyz: tuple[float, float, float]
    body_name: str
    reason: str


def body_up_dot_world_z(model: Any, data: Any, body_name: str) -> float | None:
    """Return body +Z · world +Z for ``body_name``, or ``None`` if missing."""
    import mujoco

    bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(body_name)))
    if bid < 0 or bid >= int(model.nbody):
        return None
    r = np.asarray(data.xmat[bid], dtype=np.float64).reshape(3, 3)
    return float(r[2, 2])


def assess_base_upright(
    model: Any,
    data: Any,
    *,
    base_body_name: str = "base_link",
    max_tilt_deg: float | None = None,
) -> BaseUprightStatus:
    """Classify whether the robot base is still upright.

    ``tilt_deg`` is ``acos(clamp(up·ẑ))`` in degrees (0 = upright, 90 = on its side).
    """
    name = str(base_body_name)
    limit = float(max_tilt_deg if max_tilt_deg is not None else max_tilt_deg_from_env())
    limit = max(1.0, min(89.0, limit))
    min_up = math.cos(math.radians(limit))

    import mujoco

    bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
    if bid < 0 or bid >= int(model.nbody):
        return BaseUprightStatus(
            found=False,
            upright=True,
            tilt_deg=0.0,
            up_dot_z=1.0,
            base_xyz=(0.0, 0.0, 0.0),
            body_name=name,
            reason=f"body {name!r} not found",
        )

    xyz = (
        float(data.xpos[bid, 0]),
        float(data.xpos[bid, 1]),
        float(data.xpos[bid, 2]),
    )
    up = body_up_dot_world_z(model, data, name)
    assert up is not None
    up_clamped = float(np.clip(up, -1.0, 1.0))
    tilt = float(math.degrees(math.acos(up_clamped)))
    if up >= min_up:
        return BaseUprightStatus(
            found=True,
            upright=True,
            tilt_deg=tilt,
            up_dot_z=up,
            base_xyz=xyz,
            body_name=name,
            reason="ok",
        )
    return BaseUprightStatus(
        found=True,
        upright=False,
        tilt_deg=tilt,
        up_dot_z=up,
        base_xyz=xyz,
        body_name=name,
        reason=f"tilt {tilt:.1f}° > max {limit:.1f}° (up·ẑ={up:.3f})",
    )


class FallOverMonitor:
    """Throttle fall-over logging for a physics loop."""

    def __init__(
        self,
        *,
        base_body_name: str = "base_link",
        max_tilt_deg: float | None = None,
        min_sim_time_s: float = DEFAULT_MIN_SIM_TIME_S,
        repeat_interval_s: float = DEFAULT_REPEAT_INTERVAL_S,
        log: Logger | None = None,
    ) -> None:
        self.base_body_name = str(base_body_name)
        self.max_tilt_deg = max_tilt_deg
        self.min_sim_time_s = float(min_sim_time_s)
        self.repeat_interval_s = float(repeat_interval_s)
        self._log = log or logger
        self._was_fallen = False
        self._last_report_wall_s = 0.0
        self._last_status: BaseUprightStatus | None = None

    @property
    def last_status(self) -> BaseUprightStatus | None:
        return self._last_status

    def maybe_report(self, model: Any, data: Any) -> BaseUprightStatus:
        """Assess uprightness; log a red error on fall (and periodically while down)."""
        import time

        status = assess_base_upright(
            model,
            data,
            base_body_name=self.base_body_name,
            max_tilt_deg=self.max_tilt_deg,
        )
        self._last_status = status
        if not status.found:
            return status

        sim_t = float(getattr(data, "time", 0.0) or 0.0)
        if sim_t < self.min_sim_time_s:
            return status

        fallen = not status.upright
        now = time.monotonic()
        if fallen and (not self._was_fallen or (now - self._last_report_wall_s) >= self.repeat_interval_s):
            x, y, z = status.base_xyz
            self._log.error(
                f"SIM ROBOT FALLEN OVER: base={status.body_name!r} "
                f"tilt={status.tilt_deg:.1f}° up·ẑ={status.up_dot_z:.3f} "
                f"xyz=({x:.3f}, {y:.3f}, {z:.3f}) sim_t={sim_t:.2f}s — "
                f"{status.reason}. Nav/mapping will be wrong until you reset the sim."
            )
            self._last_report_wall_s = now
        elif not fallen and self._was_fallen:
            self._log.alert(
                f"SIM ROBOT UPRIGHT AGAIN: base={status.body_name!r} "
                f"tilt={status.tilt_deg:.1f}° (was fallen)"
            )
        self._was_fallen = fallen
        return status
