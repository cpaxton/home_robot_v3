# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Short straight base motion on private native scene geometry.

Sweeps are sampled static checks, including an XY/yaw tracking envelope. They
are not continuous collision or dynamics certificates. No live qpos is written.
"""
from __future__ import annotations

import itertools
import time
from copy import copy
from dataclasses import dataclass

import mujoco
import numpy as np

from emet.controller.manipulation.native_scene import qpos_difference
from emet.eval.physical_tamp import base_pose, make_scene_checker, write_offline_base_pose
from emet.motion.mujoco_collision import body_subtree
from emet.motion.release import held_checker, relative_pose

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


def sweep_clear(model, measured, start, goal, *, deadline=None, envelope=True, event=lambda **row: None,
                loaded_checker=None, scene_checker=None):
    """Check full robot geometry and, when supplied, the measured held payload.

    Unloaded motion forbids finger/target contact. Loaded motion preserves only
    the held checker's explicit grasp contact allowances and scene map hooks.

    Private data and a private checker protect live contacts and payload state.
    The measured arm is held through the base sweep. A colliding held-arm posture
    rejects; this bounded implementation does not invent an unchecked tuck.
    """
    start, goal = np.asarray(start), np.asarray(goal)
    if start.shape != (3,) or goal.shape != (3,) or not np.isfinite([start, goal]).all():
        return False
    scratch = mujoco.MjData(model)
    mujoco.mj_copyData(scratch, model, measured)
    checker = (held_checker(loaded_checker, relative_pose(scratch, loaded_checker.payload_body, loaded_checker.payload_parent))
               if loaded_checker is not None else copy(scene_checker) if scene_checker is not None
               else make_scene_checker(model, {'object_body': 'base_link', 'gripper_bodies': ()}))
    if checker.unsupported_links(model, ['base_link']):
        return False
    xy = np.arange(-XY_ENVELOPE_M, XY_ENVELOPE_M + 1e-9, LINEAR_STEP_M) if envelope else [0.]
    yaw = np.arange(-YAW_ENVELOPE_RAD, YAW_ENVELOPE_RAD + 1e-9, ANGULAR_STEP_RAD) if envelope else [0.]
    offsets = list(itertools.product(xy, xy, yaw))
    count = max(1, int(np.ceil(np.linalg.norm(goal[:2] - start[:2]) / LINEAR_STEP_M)))
    turn = np.arctan2(np.sin(goal[2]-start[2]), np.cos(goal[2]-start[2]))
    count = max(count, int(np.ceil(abs(turn) / ANGULAR_STEP_RAD)))
    if envelope and turn == 0 and np.any(start[:2] != goal[:2]):
        # A guard may prove a subset of its map checks over this whole straight
        # segment. Native contacts and sample order still run at every pose.
        checks = []
        for check in checker.additional_scene_checks:
            batch = getattr(getattr(check, '__self__', None), 'for_straight_base_sweep', None)
            checks.append(check if batch is None else batch(
                model, scratch, start, goal, xy_error=XY_ENVELOPE_M, yaw_error=YAW_ENVELOPE_RAD
            ))
        checker.additional_scene_checks = tuple(checks)
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
    """Try aligned stationary grasps, checked base alternatives, then all stationary grasps."""
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


def execute_approach(robot, executor, approach, *, loaded=False, scene_checker=None):
    """ACK is dispatch only; require fresh measured arrival, cancellation and quiet."""
    model, data = executor.model, executor.data
    if not approach_available(robot):
        return ApproachResult(False, 'base_approach_unavailable')
    if loaded and (not getattr(executor, '_loaded_binding_done', False)
                   or not executor.payload_body or executor.collision.payload_body != executor.payload_body):
        return ApproachResult(False, 'payload_not_retained')
    def check(measured, a, b, **kwargs):
        return sweep_clear(model, measured, a, b,
            **({'loaded_checker': executor.collision} if loaded else
               {'scene_checker': scene_checker} if scene_checker is not None else {}), **kwargs)
    executor.synchronize(data)
    if loaded and not executor.payload_retained():
        return ApproachResult(False, 'payload_not_retained')
    start, goal = np.array(approach.start), np.array(approach.goal)
    current = base_pose(model, data)
    if np.linalg.norm(current[:2]-start[:2]) > .01 or abs(_angle(current[2]-start[2])) > .015:
        return ApproachResult(False, 'base_approach_invalidated')
    before_check = data.qpos.copy()
    if not check(data, current, goal, event=executor.event):
        return ApproachResult(False, 'base_approach_invalidated')
    robot_ids = body_subtree(model, 'base_link')
    if loaded:
        robot_ids |= body_subtree(model, executor.payload_body)
    scene_addresses = [address for j in range(model.njnt) if model.jnt_bodyid[j] not in robot_ids
                       for address in range(int(model.jnt_qposadr[j]),
                                            int(model.jnt_qposadr[j+1]) if j+1 < model.njnt else model.nq)]
    executor.synchronize(data)
    if loaded and not executor.payload_retained():
        return ApproachResult(False, 'payload_not_retained')
    current = base_pose(model, data)
    if (np.linalg.norm(current[:2]-start[:2]) > .01 or abs(_angle(current[2]-start[2])) > .015
            or np.max(np.abs(data.qpos[executor.qadr]-before_check[executor.qadr])) > .002
            or (scene_addresses and np.max(np.abs(qpos_difference(model, data.qpos, before_check)[scene_addresses])) > .0001)):
        return ApproachResult(False, 'base_approach_invalidated')
    arm = data.qpos[executor.qadr].copy()
    scene_reference = data.qpos.copy()
    direction = goal[:2]-start[:2]
    length = float(np.linalg.norm(direction))
    if length < 1e-8:
        return ApproachResult(False, 'base_approach_invalidated')
    direction /= length
    started_sim, started_wall = float(data.time), time.monotonic()
    cancelled, cancel_attempted, arrived, quiet, anchor, arm_anchor = False, False, False, None, None, None
    result = ApproachResult(False, 'base_approach_failed', True)
    def cancel_once():
        nonlocal cancelled, cancel_attempted
        if not cancel_attempted:
            cancel_attempted = True
            cancelled = _cancel(robot)
        return cancelled

    def follow():
        nonlocal arrived, quiet, anchor, arm_anchor, scene_reference
        returned = robot.move_base_to(goal, blocking=False, world_frame=True,
                                     navigation_policy='manipulation', timeout=30.)
        acknowledged = isinstance(returned, (bool, np.bool_)) and bool(returned)
        executor.event(phase='base_approach', goal_world=goal.tolist(), command_returned=acknowledged)
        if not acknowledged:
            return result
        while data.time-started_sim < 12. and time.monotonic()-started_wall < 45.:
            time.sleep(.05)
            executor.synchronize(data)
            if loaded and not executor.payload_retained():
                return ApproachResult(False, 'payload_not_retained', True)
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
                    or not check(data, current, current, envelope=False)):
                executor.event(phase='base_approach_rejected', reason='tracking_or_collision',
                               measured_world=current.tolist(), arm_residual=arm_error)
                return result
            if scene_addresses and np.max(np.abs(qpos_difference(model, data.qpos, scene_reference)[scene_addresses])) > .0001:
                if not check(data, current, goal, event=executor.event):
                    executor.event(phase='base_approach_rejected', reason='scene_changed_collision')
                    return result
                scene_reference = data.qpos.copy()
            in_goal = (np.linalg.norm(current[:2]-goal[:2]) <= .01
                       and abs(_angle(current[2]-goal[2])) <= .015)
            still = (anchor is not None and np.linalg.norm(current[:2]-anchor[:2]) <= .002
                     and abs(_angle(current[2]-anchor[2])) <= .002)
            if arm_anchor is not None:
                drift = data.qpos[executor.qadr]-arm_anchor
                still &= np.max(np.abs(drift)) <= .002
                for group in getattr(executor, 'coupled_groups', ()):
                    still &= abs(sum(drift[executor.joint_names.index(n)] for n in group)) <= .002
            if not in_goal or not still:
                quiet, anchor = float(data.time), current.copy()
                arm_anchor = data.qpos[executor.qadr].copy()
            elif data.time-quiet >= .3:
                if not cancelled:
                    if not cancel_once():
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
        if not cancel_once():
            executor.event(phase='base_approach_stop_unconfirmed', primary_reason=result.reason)
            result = ApproachResult(False, 'navigation_cancellation_unconfirmed', True)
    return result


def verify_approach_ready(executor, approach):
    """Recheck a quiet measured base/arm immediately before the grasp skill."""
    started_wall, anchor, quiet = time.monotonic(), None, None
    goal = np.asarray(approach.goal)
    while time.monotonic()-started_wall < 3.:
        executor.synchronize(executor.data)
        current = base_pose(executor.model, executor.data)
        if np.linalg.norm(current[:2]-goal[:2]) > .01 or abs(_angle(current[2]-goal[2])) > .015:
            return False
        q = executor.data.qpos[executor.qadr].copy()
        state = np.r_[current, q]
        quiet_now = anchor is not None
        if anchor is not None:
            delta = state-anchor
            delta[2] = _angle(delta[2])
            quiet_now = np.max(np.abs(delta)) <= .002
            for group in getattr(executor, 'coupled_groups', ()):
                quiet_now &= abs(sum(delta[3+executor.joint_names.index(n)] for n in group)) <= .002
        if not quiet_now:
            anchor, quiet = state, float(executor.data.time)
        elif executor.data.time-quiet >= .3:
            return True
        time.sleep(.05)
    return False


def _cancel(robot):
    try:
        value = robot.cancel_navigation()
        return isinstance(value, (bool, np.bool_)) and bool(value)
    except (RuntimeError, ValueError, TimeoutError):
        return False


def _angle(value):
    return np.arctan2(np.sin(value), np.cos(value))
