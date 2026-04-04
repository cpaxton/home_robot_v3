# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

import logging

import numpy as np
import trimesh.transformations as tra
from geometry_msgs.msg import Point, Pose, Quaternion

log = logging.getLogger(__name__)


def matrix_from_pose_msg(msg):
    T = tra.quaternion_matrix([msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z])
    T[:3, 3] = np.array([msg.position.x, msg.position.y, msg.position.z])
    return T


def matrix_to_pose_msg(matrix):
    pose = Pose()
    w, x, y, z = tra.quaternion_from_matrix(matrix)
    pose.orientation = Quaternion(x=x * 1.0, y=y * 1.0, z=z * 1.0, w=w * 1.0)
    xyz = matrix[:3, 3].tolist()
    pose.position = Point(x=xyz[0], y=xyz[1], z=xyz[2])
    return pose


def transform_to_list(stamped_transform):
    """Extract translation and quaternion (x,y,z,w) from StampedTransform."""
    t = stamped_transform.transform.translation
    r = stamped_transform.transform.rotation
    trans = np.array([t.x, t.y, t.z])
    rot = np.array([r.x, r.y, r.z, r.w])
    return trans, rot


def to_matrix(trans, rot):
    """Build 4x4 matrix from translation and quaternion (x,y,z,w)."""
    T = tra.quaternion_matrix([rot[3], rot[0], rot[1], rot[2]])
    T[:3, 3] = trans
    return T
