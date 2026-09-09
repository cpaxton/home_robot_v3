# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""Shell env toggles for MuJoCo / Robosuite sim navigation (colored warnings at startup)."""

from __future__ import annotations

import os

from emet.utils.logger import alert, warning

_TRUE = frozenset({"1", "true", "yes", "on"})

_warned_sim_nav_env = False


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE


def env_sim_nav_debug() -> bool:
    """Verbose ``[sim_nav]`` logs on the sim server (per-action frames, drive progress)."""
    return _env_truthy("EMET_SIM_NAV_DEBUG")


def env_sim_nav_teleport() -> bool:
    """Client sets ``nav_teleport`` on every ``move_base_to`` (instant base snap; CI/e2e only)."""
    return _env_truthy("EMET_SIM_NAV_TELEPORT")


def env_zmq_timing() -> bool:
    """Periodic ZMQ SEND/RECV timing lines (``BaseZmqServer``). Off by default; use with ``--verbose`` or set ``1``."""
    return _env_truthy("EMET_ZMQ_TIMING")


def env_manip_mode() -> str:
    """Agent pick/place backend: ``teleport`` (default) or ``kinematic`` (IK + attach)."""
    raw = os.environ.get("EMET_MANIP_MODE", "").strip().lower()
    if raw in ("teleport", "kinematic"):
        return raw
    return ""


def env_manip_collision() -> str:
    """Arm collision filter: ``none``, ``aabb`` (table solids), ``voxel`` (2D map)."""
    raw = os.environ.get("EMET_MANIP_COLLISION", "").strip().lower()
    if raw in ("none", "voxel", "aabb"):
        return raw
    return ""


def env_manip_planner() -> str:
    """Kinematic arm path planner: ``rrt_connect`` (default), ``rrt``, or ``linear``."""
    raw = os.environ.get("EMET_MANIP_PLANNER", "").strip().lower()
    if raw in ("rrt_connect", "rrt", "linear"):
        return raw
    return ""


def env_sim_third_person() -> bool:
    """Include ``third_person_image`` in full ZMQ observations (extra MuJoCo render)."""
    return _env_truthy("EMET_SIM_THIRD_PERSON")


def env_sim_overhead() -> bool:
    """Include ``overhead_image`` (nadir FREE cam) in full ZMQ observations."""
    return _env_truthy("EMET_SIM_OVERHEAD")


def env_sim_third_person_camera() -> str:
    """Optional body name to follow for chase cam (default falls through to ``base_link``)."""
    raw = os.environ.get("EMET_SIM_THIRD_PERSON_CAMERA", "").strip()
    return raw or "third_person"


def warn_sim_nav_env_flags(*, force: bool = False) -> None:
    """Print yellow stderr warnings for active ``EMET_SIM_NAV_*`` env vars (once per process)."""
    global _warned_sim_nav_env
    if _warned_sim_nav_env and not force:
        return
    any_flag = env_sim_nav_debug() or env_sim_nav_teleport()
    if not any_flag:
        return
    _warned_sim_nav_env = True
    if env_sim_nav_debug():
        warning(
            "EMET_SIM_NAV_DEBUG=1 — sim server will log verbose [sim_nav] navigation "
            "(frames, goals, velocity-drive progress). Unset when not debugging."
        )
    if env_sim_nav_teleport():
        warning(
            "EMET_SIM_NAV_TELEPORT=1 — every move_base_to uses nav_teleport (instant base snap). "
            "For CI/e2e only; normal Dynagraph explore should leave this unset (velocity drive)."
        )
    alert(
        "Sim nav env flags active — grep logs for [sim_nav]. "
        "Use: uv run emet serve mujoco … from this repo so server code matches your tree."
    )
