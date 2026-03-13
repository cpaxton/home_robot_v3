# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc. All rights reserved.
# Slim pose utils for emet-core (no torch).

import numpy as np
import trimesh.transformations as tra
from scipy.spatial.transform import Rotation


def transform_to_list(transform):
    """Converts tf2 transform to position and rotation lists."""
    rot = transform.transform.rotation
    trans = transform.transform.translation
    rot = [rot.x, rot.y, rot.z, rot.w]
    trans = [trans.x, trans.y, trans.z]
    return trans, rot


def to_matrix(pos, rot, trimesh_format=False) -> np.ndarray:
    """Converts pos, quat to matrix format."""
    if trimesh_format:
        w, x, y, z = rot
    else:
        x, y, z, w = rot
    T = tra.quaternion_matrix([w, x, y, z])
    T[:3, 3] = pos
    return T


def to_pos_quat(matrix):
    """Utility to convert to (pos, quaternion) tuple in ROS quaternion format."""
    w, x, y, z = tra.quaternion_from_matrix(matrix)
    pos = matrix[:3, 3]
    return pos, np.array([x, y, z, w])


def get_l2_distance(x1, x2, y1, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def get_pose(position, rotation):
    x = -position[2]
    y = -position[0]
    euler_angles = Rotation.from_quat(rotation).as_euler()
    axis = euler_angles[0]
    if (axis % (2 * np.pi)) < 0.1 or (axis % (2 * np.pi)) > 2 * np.pi - 0.1:
        o = euler_angles[1]
    else:
        o = 2 * np.pi - euler_angles[1]
    if o > np.pi:
        o -= 2 * np.pi
    return x, y, o


def get_rel_pose_change(pos2, pos1):
    x1, y1, o1 = pos1
    x2, y2, o2 = pos2
    theta = np.arctan2(y2 - y1, x2 - x1) - o1
    dist = get_l2_distance(x1, x2, y1, y2)
    dx = dist * np.cos(theta)
    dy = dist * np.sin(theta)
    do = o2 - o1
    return dx, dy, do


def get_new_pose(pose, rel_pose_change):
    x, y, o = pose
    dx, dy, do = rel_pose_change
    global_dx = dx * np.sin(np.deg2rad(o)) + dy * np.cos(np.deg2rad(o))
    global_dy = dx * np.cos(np.deg2rad(o)) - dy * np.sin(np.deg2rad(o))
    x += global_dy
    y += global_dx
    o += np.rad2deg(do)
    if o > 180.0:
        o -= 360.0
    return x, y, o


def threshold_poses(coords, shape):
    coords[0] = min(max(0, coords[0]), shape[0] - 1)
    coords[1] = min(max(0, coords[1]), shape[1] - 1)
    return coords


def normalize_angle(angle_in_degrees):
    angle_in_degrees = angle_in_degrees % 360.0
    if angle_in_degrees > 180:
        angle_in_degrees -= 360
    return angle_in_degrees


def normalize_radians(angle_in_radians):
    angle_in_radians = angle_in_radians % (2 * np.pi)
    if angle_in_radians > np.pi:
        angle_in_radians -= 2 * np.pi
    return angle_in_radians


def convert_pose_habitat_to_opencv(hab_pose: np.ndarray) -> np.ndarray:
    """Update axis convention of habitat pose to match the real-world axis convention."""
    hab_pose = hab_pose.copy()
    hab_pose[[1, 2]] = hab_pose[[2, 1]]
    hab_pose[:, [1, 2]] = hab_pose[:, [2, 1]]
    hab_pose[0, 0] = -hab_pose[0, 0]
    hab_pose[1, 1] = -hab_pose[1, 1]
    hab_pose[0, 3] = -hab_pose[0, 3]
    return hab_pose
