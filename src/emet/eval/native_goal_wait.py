# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.
"""Bounded encoder-feedback correction for native simulation diagnostics."""
import time

import mujoco
import numpy as np


def wait_for_native_goal(arm, model, data, addresses, limits, goal, *, position_tolerance_m=.008,
                         timeout_s=4., rotation_tolerance_rad=.1, max_corrections=3, task_position=None, task_rotation=None, require_task_pose=False):
    """Correct persistent servo bias; success always uses measured nominal-goal error.

    This diagnostic does not infer contact or authorize hardware use. Corrections
    are bounded in size, count and time; joint limits remain enforced.
    """
    nominal = np.clip(np.asarray(goal, float), limits[:, 0], limits[:, 1])
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    scratch.qpos[addresses] = nominal
    mujoco.mj_kinematics(model, scratch)
    reference = scratch.body(arm.chain.ee_body)
    reference_position = reference.xpos.copy() if task_position is None else np.asarray(task_position, float)
    reference_rotation = reference.xmat.reshape(3, 3).copy() if task_rotation is None else np.asarray(task_rotation, float)
    command = nominal.copy()
    cap = np.r_[np.full(7, .12), .004]
    increment = np.r_[np.full(7, .04), .001]
    started = time.monotonic()
    next_correction = started+.6
    corrections = []
    arm.command_joints(command)
    while True:
        time.sleep(.02)
        arm.synchronize(data)
        measured = data.qpos[addresses].copy()
        error = nominal-measured
        body = data.body(arm.chain.ee_body)
        distance = float(np.linalg.norm(body.xpos-reference_position))
        angle = float(np.arccos(np.clip((np.trace(body.xmat.reshape(3, 3).T @
            reference_rotation)-1)/2, -1., 1.)))
        arrived = bool((np.max(np.abs(error[:7])) < .015 and abs(error[7]) < .001)
                       or (distance <= position_tolerance_m and angle <= rotation_tolerance_rad))
        if require_task_pose:
            arrived = bool(distance <= position_tolerance_m and angle <= rotation_tolerance_rad)
        now = time.monotonic()
        if arrived or now-started >= timeout_s:
            return {'arrived': arrived, 'joint_errors': np.abs(error).tolist(),
                    'tool_position_error_m': distance, 'tool_rotation_error_rad': angle,
                    'waypoint_m': reference_position.tolist(), 'measured_m': body.xpos.tolist(),
                    'corrections': corrections}
        if now >= next_correction and len(corrections) < max_corrections:
            command = np.clip(command+np.clip(error, -increment, increment), nominal-cap, nominal+cap)
            command = np.clip(command, limits[:, 0], limits[:, 1])
            arm.command_joints(command)
            corrections.append({'elapsed_s': now-started, 'offset': (command-nominal).tolist()})
            next_correction = now+.6
