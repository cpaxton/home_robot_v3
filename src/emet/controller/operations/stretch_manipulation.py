# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Shared geometry handoff for Stretch's side-facing pick and place adapters."""

import numpy as np
from scipy.spatial.transform import Rotation

from emet.motion import constants


def world_delta_to_model_base(delta, ee_world_pose, ee_base_quaternion):
    """Use the common grasp frame, independent of episode-relative odometry yaw."""
    return Rotation.from_quat(ee_base_quaternion).as_matrix() @ ee_world_pose[:3, :3].T @ delta


def orient_arm_toward_target(robot, xyz):
    """Turn the -Y arm toward a world target, then wait for post-motion RGB-D."""
    from emet.controller.dynamem.look import wait_post_motion_obs

    pose = np.array(robot.get_base_pose_world(), dtype=float, copy=True)
    delta = np.asarray(xyz)[:2] - pose[:2]
    if not np.isfinite(delta).all() or np.linalg.norm(delta) < 1e-6:
        raise ValueError("Cannot orient manipulation toward an invalid target")
    pose[2] = np.arctan2(delta[1], delta[0]) + np.pi / 2
    robot.switch_to_navigation_mode()
    if not robot.move_base_to(pose, blocking=True, navigation_policy="precision", world_frame=True):
        raise RuntimeError("Manipulation orientation did not complete")
    robot.switch_to_manipulation_mode()
    robot.head_to(*constants.look_at_ee, blocking=True)
    wait_post_motion_obs(robot, timeout=2.0)
