# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Joint-space RRT-Connect for kinematic arm motion (reuse of ``emet.motion.algo``).

Plans collision-aware paths between two arm/torso configurations. Collision uses the
agent voxel / 2D obstacle map when provided — not CuRobo / not MJCF mesh geometry.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from emet.motion.algo import get_planner
from emet.motion.algo.shortcut import Shortcut
from emet.motion.base import ConfigurationSpace
from emet.motion.mujoco_arm_ik import interpolate_arm_waypoints, joint_qpos_addrs
from emet.utils.logger import Logger

logger = Logger(__name__)

# Native servos and soft joint stops can briefly overshoot a nominal endpoint.
# Admit at most 0.1 mm (slides) / 0.1 mrad (hinges), checking the actual measured
# geometry rather than clipping feedback. IK goals and sampling keep nominal limits.
JOINT_BOUND_TOLERANCE = 1e-4


@dataclass(frozen=True)
class ArmRrtPlanResult:
    success: bool
    waypoints: list[np.ndarray]
    planner: str
    reason: str | None = None
    detail: str | None = None
    departure: Any | None = None


class DeparturePath(list):
    """Waypoints retaining the edge policy needed by runtime revalidation."""

    def __init__(self, waypoints, departure, escape_end_index=None):
        super().__init__(waypoints)
        self.departure = departure
        self.escape_end_index = escape_end_index


class ContactClearancePath(list):
    """Ordinary motion retaining only original-patch clearance context."""

    def __init__(self, waypoints, clearance_scope):
        super().__init__(waypoints)
        self.clearance_scope = clearance_scope


def joint_limits_from_model(
    model: mujoco.MjModel,
    joint_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mins, maxs) for named 1-DoF joints (unlimited joints → ±π)."""
    mins: list[float] = []
    maxs: list[float] = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
        if jid < 0:
            raise ValueError(f"joint not found: {name!r}")
        if model.jnt_limited[jid]:
            mins.append(float(model.jnt_range[jid][0]))
            maxs.append(float(model.jnt_range[jid][1]))
        else:
            mins.append(-np.pi)
            maxs.append(np.pi)
    return np.asarray(mins, dtype=np.float64), np.asarray(maxs, dtype=np.float64)


def make_arm_configuration_space(
    model: mujoco.MjModel,
    joint_names: Sequence[str],
    *,
    step_size: float = 0.15,
    rng=None,
) -> ConfigurationSpace:
    mins, maxs = joint_limits_from_model(model, joint_names)
    return ConfigurationSpace(len(joint_names), mins, maxs, step_size=float(step_size), rng=rng)


def make_arm_validate_fn(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_names: Sequence[str],
    collision: Any | None = None,
    *,
    mins: np.ndarray | None = None,
    maxs: np.ndarray | None = None,
    deadline: float | None = None,
) -> Callable[[np.ndarray], bool]:
    """Return a validate(q) that checks joint bounds + optional link collision."""
    qadr = joint_qpos_addrs(model, joint_names)
    if mins is None or maxs is None:
        mins, maxs = joint_limits_from_model(model, joint_names)
    lo = np.asarray(mins, dtype=np.float64).reshape(-1)
    hi = np.asarray(maxs, dtype=np.float64).reshape(-1)

    def validate(q: np.ndarray) -> bool:
        if deadline is not None and time.monotonic() >= deadline:
            return False
        qq = np.asarray(q, dtype=np.float64).reshape(-1)
        if qq.shape[0] != len(qadr) or not np.isfinite(qq).all():
            return False
        if np.any(qq < lo - JOINT_BOUND_TOLERANCE) or np.any(qq > hi + JOINT_BOUND_TOLERANCE):
            return False
        for a, v in zip(qadr, qq, strict=True):
            data.qpos[a] = float(v)
        if collision is None:
            return True
        return not collision.configuration_collides(model, data)

    return validate


def arm_config_violation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_names: Sequence[str],
    q: np.ndarray,
    collision: Any | None = None,
    *,
    mins: np.ndarray | None = None,
    maxs: np.ndarray | None = None,
) -> str | None:
    """Return a detailed reason *q* is an invalid configuration, or ``None``.

    Distinguishes the two ``invalid_start`` predicates the generic reason hides:
    an out-of-range joint (``joint_bounds:<name>``) versus a colliding link
    (``collision``). Writes *q* into ``data`` only when a collision check runs,
    matching :func:`make_arm_validate_fn`.
    """
    qadr = joint_qpos_addrs(model, joint_names)
    qq = np.asarray(q, dtype=np.float64).reshape(-1)
    if qq.shape[0] != len(qadr) or not np.isfinite(qq).all():
        return "nonfinite_configuration"
    if mins is None or maxs is None:
        mins, maxs = joint_limits_from_model(model, joint_names)
    lo = np.asarray(mins, dtype=np.float64).reshape(-1)
    hi = np.asarray(maxs, dtype=np.float64).reshape(-1)
    violated = np.flatnonzero((qq < lo - JOINT_BOUND_TOLERANCE) | (qq > hi + JOINT_BOUND_TOLERANCE))
    if violated.size:
        i = int(violated[0])
        return f"joint_bounds:{joint_names[i]}={qq[i]:.6f} in [{lo[i]:.6f},{hi[i]:.6f}]"
    for a, v in zip(qadr, qq, strict=True):
        data.qpos[a] = float(v)
    if collision is not None and collision.configuration_collides(model, data):
        return "collision"
    return None


def plan_arm_joint_path(model, data, *, joint_names, q_start, q_goal, collision=None,
                        via_configurations=(), **kwargs):
    """Try certified posture detours, then plan the commanded environmental sweep.

    Postures are proposals from the embodiment, never trusted free-space states.
    Each leg uses the same edge checks; colliding guide states are discarded.
    """
    scope = (collision.motion_scope(model, data, joint_names, q_start)
             if collision is not None and hasattr(collision, 'motion_scope') else nullcontext())
    with scope:
        started = time.monotonic()
        budget = kwargs.get("max_time_s")
        q_start = np.asarray(q_start, float).reshape(-1)
        q_goal = np.asarray(q_goal, float).reshape(-1)
        if (len(via_configurations) == 0 or q_start.size != len(joint_names) or q_goal.size != len(joint_names)
                or not np.isfinite(q_start).all() or not np.isfinite(q_goal).all()
                or (budget is not None and (not np.isfinite(budget) or budget <= 0))):
            return _plan_arm_joint_path(model, data, joint_names=joint_names, q_start=q_start,
                                        q_goal=q_goal, collision=collision, **kwargs)
        # Cheap deterministic routes avoid spending the whole random search
        # budget rediscovering a robot's standard folded posture.
        for via in via_configurations:
            route = np.atleast_2d(np.asarray(via, float))
            if route.shape[1:] != q_start.shape or not 1 <= len(route) <= 32 or not np.isfinite(route).all():
                continue
            waypoints = [q_start.copy()]
            for destination in (*route, q_goal):
                remaining = None if budget is None else budget - (time.monotonic() - started)
                if remaining is not None and remaining <= 0:
                    break
                leg_options = {**kwargs, "planner": kwargs.get("planner", "rrt_connect"), "linear_fallback": True,
                               "max_time_s": remaining, "departure": False}
                leg = _plan_arm_joint_path(model, data, joint_names=joint_names, q_start=waypoints[-1],
                                          q_goal=destination, collision=collision, **leg_options)
                if not leg.success:
                    break
                waypoints.extend(leg.waypoints[1:])
            else:
                return ArmRrtPlanResult(True, waypoints, "posture_detour")
        # A failed route must not leave a hypothetical posture in the caller's
        # model, including when the remaining time budget is exhausted.
        data.qpos[joint_qpos_addrs(model, joint_names)] = q_start
        mujoco.mj_kinematics(model, data)
        if budget is not None:
            remaining = budget - (time.monotonic() - started)
            if remaining <= 0:
                return ArmRrtPlanResult(False, [], str(kwargs.get("planner", "rrt_connect")), "no_plan_within_budget")
            kwargs["max_time_s"] = remaining
        return _plan_arm_joint_path(model, data, joint_names=joint_names, q_start=q_start,
                                    q_goal=q_goal, collision=collision, **kwargs)


def _plan_arm_joint_path(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    joint_names: Sequence[str],
    q_start: np.ndarray,
    q_goal: np.ndarray,
    collision: Any | None = None,
    planner: str = "rrt_connect",
    max_iter: int = 400,
    step_size: float = 0.15,
    goal_tolerance: float = 0.05,
    shortcut: bool = True,
    shortcut_iter: int = 50,
    linear_fallback: bool = True,
    linear_steps: int = 15,
    verbose: bool = False,
    rng=None,
    departure=None,
    max_time_s=None,
    start_bound_slack=0.,
) -> ArmRrtPlanResult:
    """Plan a joint-space path from ``q_start`` to ``q_goal``.

    Default planner is RRT-Connect (same stack as base nav). On failure, optionally falls
    back to linear interpolation (still rejected if any waypoint collides when a checker
    is present).
    """
    q0 = np.asarray(q_start, dtype=np.float64).reshape(-1)
    q1 = np.asarray(q_goal, dtype=np.float64).reshape(-1)
    if q0.shape[0] != len(joint_names) or q1.shape[0] != len(joint_names):
        return ArmRrtPlanResult(False, [], planner, "dof_mismatch")
    if not np.isfinite(q0).all() or not np.isfinite(q1).all():
        return ArmRrtPlanResult(False, [], planner, "nonfinite_configuration")

    if max_time_s is not None and (not np.isfinite(max_time_s) or max_time_s <= 0):
        return ArmRrtPlanResult(False, [], planner, "invalid_planning_budget")
    deadline = time.monotonic() + max_time_s if max_time_s is not None else None
    space = make_arm_configuration_space(model, joint_names, step_size=step_size, rng=rng)
    if (not np.isfinite(start_bound_slack) or not 0 <= start_bound_slack <= .001
            or start_bound_slack and planner != 'linear'):
        return ArmRrtPlanResult(False, [], planner, 'invalid_start_bound_recovery')
    low, high = space.mins, space.maxs
    if start_bound_slack:
        if np.any(q0 < low-start_bound_slack) or np.any(q0 > high+start_bound_slack):
            return ArmRrtPlanResult(False, [], planner, 'invalid_start', detail='joint_bounds')
        # Only the measured start can extend the segment's allowed interval.
        # The goal retains nominal bounds, and linear interpolation moves each
        # overrun joint monotonically inward. Never clip the measured geometry.
        low, high = np.minimum(low, q0), np.maximum(high, q0)
    validate = make_arm_validate_fn(model, data, joint_names, collision, mins=low, maxs=high,
                                   deadline=deadline)

    start_reason = arm_config_violation(model, data, joint_names, q0, collision, mins=low, maxs=high)
    if start_reason is not None:
        if start_reason == 'collision' and departure is None and hasattr(collision, 'make_departure'):
            departure = collision.make_departure(model, data, joint_names, q0)
        if start_reason == 'collision' and departure is not None and departure is not False:
            return _plan_departure(
                model, data, joint_names, q0, q1, collision, departure,
                space=space, step_size=step_size, rng=rng,
            )
        return ArmRrtPlanResult(False, [], planner, "invalid_start", detail=start_reason)
    goal_reason = arm_config_violation(model, data, joint_names, q1, collision, mins=space.mins, maxs=space.maxs)
    if goal_reason is not None:
        return ArmRrtPlanResult(False, [], planner, "invalid_goal", detail=goal_reason)
    if deadline is not None and time.monotonic() >= deadline:
        return ArmRrtPlanResult(False, [], planner, "no_plan_within_budget")

    # Even a short move can cross an obstacle. Validate it before accepting.
    if float(np.linalg.norm(q1 - q0)) < float(goal_tolerance):
        path = [q0.copy(), *list(space.extend(q0, q1))]
        if not all(validate(q) for q in path):
            return ArmRrtPlanResult(False, [], planner, "invalid_short_path")
        return ArmRrtPlanResult(True, path, planner, None)

    algo = str(planner or "rrt_connect").strip().lower()
    if algo in ("rrt", "rrt_connect"):
        pl = get_planner(
            algo,
            space,
            validate,
            max_iter=int(max_iter),
            goal_tolerance=float(goal_tolerance),
            rng=rng,
        )
        if shortcut:
            pl = Shortcut(pl, shortcut_iter=int(shortcut_iter), rng=rng)
        res = pl.plan(q0, q1, verbose=verbose)
        if res.success and res.trajectory:
            wps = [np.asarray(n.state, dtype=np.float64).copy() for n in res.trajectory]
            # Include the exact requested endpoint; planners may stop within tolerance.
            if not np.array_equal(wps[-1], q1):
                wps.append(q1.copy())
            if all(validate(q) for a, b in zip(wps, wps[1:], strict=False) for q in space.extend(a, b)):
                return ArmRrtPlanResult(True, wps, algo, None)
        reason = getattr(res, "reason", None) or "rrt_failed"
        logger.debug(f"arm_rrt: {algo} failed ({reason}); linear_fallback={linear_fallback}")
        if not linear_fallback:
            return ArmRrtPlanResult(False, [], algo, reason)
    elif algo != "linear":
        return ArmRrtPlanResult(False, [], algo, "unknown_planner")

    # Linear (explicit request or RRT fallback)
    if deadline is not None and time.monotonic() >= deadline:
        return ArmRrtPlanResult(False, [], algo, "no_plan_within_budget")
    path = interpolate_arm_waypoints(q0, q1, n_steps=int(linear_steps))
    for index, (a, b) in enumerate(zip(path, path[1:], strict=False)):
        for q in space.extend(a, b):
            if deadline is not None and time.monotonic() >= deadline:
                return ArmRrtPlanResult(False, [], "linear", "no_plan_within_budget")
            violation = arm_config_violation(
                model, data, joint_names, q, collision, mins=low, maxs=high
            )
            if violation is not None:
                return ArmRrtPlanResult(False, [], "linear", f"linear_collision_at_{index}", detail=violation)
    return ArmRrtPlanResult(True, path, "linear", None)


def _plan_departure(model, data, joint_names, start, goal, collision, departure, *, space, step_size, rng):
    """Search forward from contact using an explicitly supplied edge policy.

    The policy owns the original contact baseline and physical travel bounds.
    Its edge check must inspect the full segment, including its endpoints.
    Ordinary collision checking resumes at the first clear configuration.
    """
    generator = np.random.default_rng(0) if rng is None else rng
    addresses = joint_qpos_addrs(model, joint_names)
    nodes, parents = [start.copy()], [-1]
    data.qpos[addresses] = start
    mujoco.mj_kinematics(model, data)
    joints = [model.joint(name).id for name in joint_names]
    slides = [i for i, jid in enumerate(joints) if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE]
    slides.sort(key=lambda i: -abs(float(data.xaxis[joints[i], 2])))
    probes = []
    for i in slides:
        direction = 1. if data.xaxis[joints[i], 2] >= 0 else -1.
        for sign in (direction, -direction):
            for distance in (.005, .01, .02):
                q = start.copy()
                q[i] += sign * distance
                probes.append(q)
    for route in getattr(departure, 'guide_configurations', ()):
        for target in route:
            target = np.asarray(target, float)
            if target.shape != start.shape:
                continue
            # Steering already bounds each proposal. Repeating sixteen
            # fractions of each guide starves gradient/random escape proposals.
            probes.append(target)
    for i in range(len(joints)):
        if i not in slides:
            for sign in (1., -1.):
                q = start.copy()
                q[i] += sign * .02
                probes.append(q)
    deadline = time.monotonic() + getattr(departure, 'search_timeout_s', 2.0)
    clear_candidates, last_connection_failure = 0, None
    for attempt in range(2000):
        if time.monotonic() >= deadline:
            break
        if attempt < len(probes):
            target = probes[attempt]
        elif attempt % 3 == 0:
            target = np.asarray(departure.sample_separating(generator), float)
        elif attempt % 3 == 1:
            target = goal
        else:
            target = generator.uniform(space.mins, space.maxs)
        if target.shape != start.shape or not np.isfinite(target).all():
            continue
        nearest = int(np.argmin([np.linalg.norm(q - target) for q in nodes]))
        origin = nodes[nearest]
        delta = target - origin
        candidate = origin + delta * min(1.0, step_size / max(np.linalg.norm(delta), 1e-12))
        if np.any(candidate < space.mins) or np.any(candidate > space.maxs):
            continue
        if not departure.validate_edge(origin, candidate):
            continue
        nodes.append(candidate.copy())
        parents.append(nearest)
        clearance_scope = getattr(departure, 'clearance_scope', nullcontext)
        with clearance_scope():
            violation = arm_config_violation(model, data, joint_names, candidate, collision,
                                            mins=space.mins, maxs=space.maxs)
            if violation is not None:
                continue
            clear_candidates += 1
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                break
            tail = plan_arm_joint_path(model, data, joint_names=joint_names, q_start=candidate,
                                      q_goal=goal, collision=collision, step_size=step_size, rng=generator,
                                      departure=False, max_iter=100, shortcut=False, max_time_s=remaining,
                                      via_configurations=getattr(departure, 'guide_configurations', ()))
        if not tail.success:
            last_connection_failure = {"reason": tail.reason, "detail": tail.detail,
                                       "collision_reason": getattr(collision, 'last_reason', None),
                                       "geometry": getattr(collision, 'last_blocked_geom', None),
                                       "contacts": list(getattr(getattr(collision, 'self_collision', None), 'last_contacts', ())),
                                       "surface": getattr(getattr(collision, 'local_scene', None), 'last_collision', None)}
            continue
        prefix, index = [], len(nodes) - 1
        while index >= 0:
            prefix.append(nodes[index])
            index = parents[index]
        data.qpos[addresses] = start
        mujoco.mj_kinematics(model, data)
        path = DeparturePath(list(reversed(prefix)) + tail.waypoints[1:], departure, len(prefix) - 1)
        path.clearance_scope = getattr(departure, 'clearance_scope', None)
        return ArmRrtPlanResult(True, path, 'departure_rrt', departure=departure)
    data.qpos[addresses] = start
    mujoco.mj_kinematics(model, data)
    if hasattr(collision, 'last_departure_diagnostics'):
        collision.last_departure_diagnostics = {
            **(collision.last_departure_diagnostics or {}),
            'reason': 'escape_search_exhausted',
            'rejections': dict(getattr(departure, 'rejections', {})),
            'last_rejection': getattr(departure, 'last_rejection', None),
            'accepted_nodes': len(nodes),
            'clear_candidates': clear_candidates,
            'last_connection_failure': last_connection_failure,
        }
    return ArmRrtPlanResult(False, [], 'departure_rrt', 'escape_search_exhausted')


def resolve_agent_manip_planner(*, config_mode: str | None = None) -> str:
    """Resolve ``rrt_connect`` | ``rrt`` | ``linear`` from env then config."""
    from emet.simulation.env_flags import env_manip_planner

    env_p = env_manip_planner()
    if env_p:
        return env_p
    mode = str(config_mode or "rrt_connect").strip().lower()
    if mode in ("rrt_connect", "rrt", "linear"):
        return mode
    return "rrt_connect"
