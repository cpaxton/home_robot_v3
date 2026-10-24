"""Scratch-only differential-base transport with a fixed loaded arm posture.

Planning collision checks and static loaded support margins are admission
screens. Runtime must independently satisfy retention, tracking and wheel/force
contracts. No live simulator data or object poses are modified by this planner.
"""

from types import SimpleNamespace

import numpy as np

from emet.rl.manipulation.grasp_setup import support_margin
from emet.rl.manipulation.profiles import MARS
from emet.rl.manipulation.tracking import TaskFrameFK, TaskPoseTrajectory


def plan_transport(model, data, checker, payload, goal, *, base_target=None):
    """Return a path and its endpoint data; input data is private planner state."""
    base_id = model.body(MARS.base_body).id
    rotation = data.xmat[base_id].reshape(3, 3).copy()
    base = np.r_[data.xpos[base_id, :2], np.arctan2(rotation[1, 0], rotation[0, 0])]
    addresses = [np.array([model.joint(n).qposadr[0] for n in group]) for group in MARS.joint_groups]
    arm = np.array([data.qpos[group].sum() for group in addresses])
    limits = np.array([np.sum([model.joint(n).range for n in group], axis=0) for group in MARS.joint_groups])
    view = SimpleNamespace(
        profile=MARS, q=addresses, base=base_id, ee=model.body(MARS.ee_body).id, base_pose=lambda: base.copy()
    )
    fk = TaskFrameFK(model, data, view)
    target = base.copy()
    if base_target is not None:
        target[:] = base_target
    else:
        target[:2] += np.asarray(goal)[:2] - data.xpos[payload, :2]
    # Differential drive turns around the driven axle, not base_link. A path
    # that rotates base_link at fixed XY demands sideways axle motion and makes
    # the base servo fight the hand trajectory. Plan in axle coordinates, then
    # convert every sample back to the declared base_link target frame.
    axle_world = np.mean([data.body(name).xpos for name in MARS.wheel_bodies[:2]], axis=0)
    offset = (rotation.T @ (axle_world - data.xpos[base_id]))[:2]

    def rotate(yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([[c, -s], [s, c]])

    axle_start, axle_goal = base.copy(), target.copy()
    axle_start[:2] += rotate(base[2]) @ offset
    axle_goal[:2] += rotate(target[2]) @ offset
    delta = axle_goal[:2] - axle_start[:2]
    travel = np.arctan2(delta[1], delta[0]) if np.linalg.norm(delta) > 1e-8 else base[2]
    yaw_delta = np.arctan2(np.sin(travel - base[2]), np.cos(travel - base[2]))
    alternative = np.arctan2(np.sin(yaw_delta + np.pi), np.cos(yaw_delta + np.pi))
    options = sorted((yaw_delta, alternative), key=abs)
    yaw_delta = options[0]
    travel = base[2] + yaw_delta
    middle = axle_start.copy()
    middle[2] = travel
    arrived = axle_goal.copy()
    arrived[2] = travel
    times, bases, hands = [], [], []
    elapsed = 0.0
    minimum_margin = float("inf")
    for start, end in ((axle_start, middle), (middle, arrived), (arrived, axle_goal)):
        distance = np.linalg.norm(end[:2] - start[:2])
        angle = abs(end[2] - start[2])
        duration = max(
            0.2,
            1.875 * distance / (0.7 * MARS.base_speed),
            # The baseline's final-heading servo caps turn requests at
            # 0.2 rad/s. Respect that executable rate as well as hardware limits.
            1.875 * angle / (0.7 * min(0.2, MARS.base_yaw_speed)),
            np.sqrt(5.8 * distance / (0.7 * MARS.base_acceleration)),
            np.sqrt(5.8 * angle / (0.7 * MARS.base_yaw_acceleration)),
        )
        for t in np.linspace(0, duration, int(np.ceil(duration / MARS.dt)) + 1):
            if times and t == 0:
                continue
            u = t / duration
            smooth = 10 * u**3 - 15 * u**4 + 6 * u**5
            pose = start + smooth * (end - start)
            pose[:2] -= rotate(pose[2]) @ offset
            q = arm.copy()
            # Carry in a fixed robot-relative posture. Intermediate world hand
            # orientation follows the base and returns at the final heading;
            # grasp-relative retention is unchanged. Counter-rotating the
            # shoulder unnecessarily couples base lag into a five-DOF arm's
            # otherwise unreachable instantaneous six-DOF correction.
            if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
                return None, "transport_joint_limits"
            hand = fk(pose, q)
            if checker.configuration_collides(model, fk.data):
                return None, "transport_collision"
            margin = support_margin(model, fk.data, MARS, payload)["margin_m"]
            minimum_margin = min(minimum_margin, margin)
            if margin < 0.005:
                return None, "transport_static_support_margin"
            times.append(elapsed + float(t))
            bases.append(pose)
            hands.append(hand.copy())
        elapsed += duration
    path = TaskPoseTrajectory(
        times,
        bases,
        hands,
        np.full(len(times), MARS.gripper_closed if payload is not None else MARS.gripper_open),
        base_mask=[True] * 3,
    )
    data.qpos[:] = fk.data.qpos
    # The checker has moved the hypothetical payload with the scratch hand.
    import mujoco

    mujoco.mj_kinematics(model, data)
    return {"trajectory": path, "minimum_static_margin_m": minimum_margin}, None
