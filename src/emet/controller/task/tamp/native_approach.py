# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Unloaded, short straight base approaches on private native scene geometry.

Sweeps are sampled static checks, including an XY/yaw tracking envelope. They
are not continuous collision or dynamics certificates. No live qpos is written.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

import mujoco
import numpy as np

from emet.eval.physical_tamp import base_pose, make_scene_checker, write_offline_base_pose
from emet.motion.mujoco_collision import body_subtree

LINEAR_STEP_M = .005
ANGULAR_STEP_RAD = .025
XY_ENVELOPE_M = .01
YAW_ENVELOPE_RAD = .05


@dataclass(frozen=True)
class BaseApproach:
    start: tuple[float, float, float]
    goal: tuple[float, float, float]


@dataclass(frozen=True)
class ApproachResult:
    success: bool
    reason: str
    execution_started: bool = False


def approach_available(robot):
    from emet.simulation.env_flags import env_sim_nav_teleport
    session = robot.get_emet_session() or {}
    if not isinstance(session, dict):
        return False
    caps = session.get('capabilities') or {}
    return (session.get('is_simulation') is True and isinstance(caps, dict) and not env_sim_nav_teleport()
            and not caps.get('teleport_base')
            and all(callable(getattr(robot, name, None)) for name in ('move_base_to', 'cancel_navigation')))


def base_alternatives(model, data):
    """Bounded forward/reverse alternatives along the actual wheel drive axis."""
    start = base_pose(model, data)
    forward = np.array([np.cos(start[2]), np.sin(start[2]), 0.])
    yield None
    for offset in (.04, -.04, .08, -.08):
        yield BaseApproach(tuple(start), tuple(start + offset * forward))


def sweep_clear(model, measured, start, goal, *, deadline=None, envelope=True, event=lambda **row: None):
    """Check full robot geometry, forbidding even finger/target contact in transit.

    Private data and a private checker protect live contacts and payload state.
    The measured arm is held through the base sweep. A colliding held-arm posture
    rejects; this bounded implementation does not invent an unchecked tuck.
    """
    start, goal = np.asarray(start), np.asarray(goal)
    if start.shape != (3,) or goal.shape != (3,) or not np.isfinite([start, goal]).all():
        return False
    scratch = mujoco.MjData(model)
    mujoco.mj_copyData(scratch, model, measured)
    checker = make_scene_checker(model, {'object_body': 'base_link', 'gripper_bodies': ()})
    if checker.unsupported_links(model, ['base_link']):
        return False
    xy = np.arange(-XY_ENVELOPE_M, XY_ENVELOPE_M + 1e-9, LINEAR_STEP_M) if envelope else [0.]
    yaw = np.arange(-YAW_ENVELOPE_RAD, YAW_ENVELOPE_RAD + 1e-9, ANGULAR_STEP_RAD) if envelope else [0.]
    offsets = list(itertools.product(xy, xy, yaw))
    count = max(1, int(np.ceil(np.linalg.norm(goal[:2] - start[:2]) / LINEAR_STEP_M)))
    turn = np.arctan2(np.sin(goal[2]-start[2]), np.cos(goal[2]-start[2]))
    count = max(count, int(np.ceil(abs(turn) / ANGULAR_STEP_RAD)))
    for t in np.linspace(0, 1, count+1):
        pose = start + t * (goal-start)
        pose[2] = start[2] + t * turn
        for offset in offsets:
            if deadline is not None and time.monotonic() >= deadline:
                raise ValueError('no_plan_within_budget')
            sample = pose + offset
            if not write_offline_base_pose(model, scratch, base_body_name='base_link',
                                           x=sample[0], y=sample[1], theta=sample[2]):
                return False
            if checker.configuration_collides(model, scratch):
                event(phase='base_sweep_rejected', pose_world=sample.tolist(), contacts=checker.last_contacts)
                return False
    return True


def planning_starts(model, data, *, deadline, event=lambda **row: None):
    """Prefer aligned grasps across checked bases before stationary fallbacks."""
    before = data.qpos.copy()
    alternatives = list(base_alternatives(model, data))
    try:
        for approach in alternatives:
            data.qpos[:] = before
            mujoco.mj_kinematics(model, data)
            if approach is not None:
                if not sweep_clear(model, data, approach.start, approach.goal, deadline=deadline, event=event):
                    continue
                event(phase='base_approach_candidate', goal_world=list(approach.goal), accepted=True)
                if not write_offline_base_pose(model, data, base_body_name='base_link',
                        x=approach.goal[0], y=approach.goal[1], theta=approach.goal[2]):
                    continue
            yield approach, data.qpos.copy(), True
        yield None, before.copy(), False
    finally:
        data.qpos[:] = before
        mujoco.mj_kinematics(model, data)


def execute_approach(robot, executor, approach):
    """ACK is dispatch only; require fresh measured arrival, cancellation and quiet."""
    model, data = executor.model, executor.data
    if not approach_available(robot):
        return ApproachResult(False, 'base_approach_unavailable')
    executor.synchronize(data)
    start, goal = np.array(approach.start), np.array(approach.goal)
    current = base_pose(model, data)
    if np.linalg.norm(current[:2]-start[:2]) > .01 or abs(_angle(current[2]-start[2])) > .015:
        return ApproachResult(False, 'base_approach_invalidated')
    if not sweep_clear(model, data, current, goal, event=executor.event):
        return ApproachResult(False, 'base_approach_invalidated')
    arm = data.qpos[executor.qadr].copy()
    robot_ids = body_subtree(model, 'base_link')
    scene_addresses = [address for j in range(model.njnt) if model.jnt_bodyid[j] not in robot_ids
                       for address in range(int(model.jnt_qposadr[j]),
                                            int(model.jnt_qposadr[j+1]) if j+1 < model.njnt else model.nq)]
    scene_reference = data.qpos[scene_addresses].copy()
    direction = goal[:2]-start[:2]
    length = float(np.linalg.norm(direction))
    if length < 1e-8:
        return ApproachResult(False, 'base_approach_invalidated')
    direction /= length
    started_sim, started_wall = float(data.time), time.monotonic()
    cancelled, arrived, quiet, anchor = False, False, None, None
    result = ApproachResult(False, 'base_approach_failed', True)
    def follow():
        nonlocal cancelled, arrived, quiet, anchor, scene_reference
        returned = robot.move_base_to(goal, blocking=False, world_frame=True,
                                     navigation_policy='manipulation', timeout=30.)
        acknowledged = isinstance(returned, (bool, np.bool_)) and bool(returned)
        executor.event(phase='base_approach', goal_world=goal.tolist(), command_returned=acknowledged)
        if not acknowledged:
            return result
        while data.time-started_sim < 12. and time.monotonic()-started_wall < 45.:
            time.sleep(.05)
            executor.synchronize(data)
            current = base_pose(model, data)
            displacement = current[:2]-start[:2]
            along = float(displacement @ direction)
            cross = float(abs(displacement[0]*direction[1]-displacement[1]*direction[0]))
            delta = data.qpos[executor.qadr]-arm
            arm_error = float(np.max(np.abs(delta)))
            for group in getattr(executor, 'coupled_groups', ()):
                arm_error = max(arm_error, abs(sum(delta[executor.joint_names.index(n)] for n in group)))
            if (along < -XY_ENVELOPE_M or along > length+XY_ENVELOPE_M
                    or cross > XY_ENVELOPE_M or abs(_angle(current[2]-start[2])) > YAW_ENVELOPE_RAD
                    or arm_error > executor.joint_tolerance
                    or not sweep_clear(model, data, current, current, envelope=False)):
                executor.event(phase='base_approach_rejected', reason='tracking_or_collision',
                               measured_world=current.tolist(), arm_residual=arm_error)
                return result
            if scene_addresses and np.max(np.abs(data.qpos[scene_addresses]-scene_reference)) > .0001:
                if not sweep_clear(model, data, current, goal, event=executor.event):
                    executor.event(phase='base_approach_rejected', reason='scene_changed_collision')
                    return result
                scene_reference = data.qpos[scene_addresses].copy()
            in_goal = (np.linalg.norm(current[:2]-goal[:2]) <= .01
                       and abs(_angle(current[2]-goal[2])) <= .015)
            still = (anchor is not None and np.linalg.norm(current[:2]-anchor[:2]) <= .002
                     and abs(_angle(current[2]-anchor[2])) <= .002)
            if not in_goal or not still:
                quiet, anchor = float(data.time), current.copy()
            elif data.time-quiet >= .3:
                if not cancelled:
                    cancelled = _cancel(robot)
                    if not cancelled:
                        return ApproachResult(False, 'navigation_cancellation_unconfirmed', True)
                    quiet, anchor = float(data.time), current.copy()
                else:
                    arrived = True
                    break
        if arrived:
            executor.event(phase='base_approach_complete', measured_world=current.tolist(),
                           goal_world=goal.tolist(), stop_confirmed=True)
            return ApproachResult(True, 'ok', True)
        executor.event(phase='base_approach_rejected', reason='arrival_timeout', measured_world=current.tolist())
        return result

    try:
        result = follow()
    except TimeoutError:
        result = ApproachResult(False, 'command_outcome_unknown', True)
    except (RuntimeError, ValueError) as error:
        executor.event(phase='base_approach_rejected', reason='state_unavailable', detail=str(error))
        result = ApproachResult(False, 'base_approach_failed', True)
    finally:
        if not cancelled and not _cancel(robot):
            result = ApproachResult(False, 'navigation_cancellation_unconfirmed', True)
    return result


def _cancel(robot):
    try:
        value = robot.cancel_navigation()
        return isinstance(value, (bool, np.bool_)) and bool(value)
    except (RuntimeError, ValueError, TimeoutError):
        return False


def _angle(value):
    return np.arctan2(np.sin(value), np.cos(value))
