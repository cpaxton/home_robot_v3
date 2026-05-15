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

"""MuJoCo-backed ZMQ sim :class:`~emet.robots.base.RobotSpec` for Stretch (MJCF-native cameras/actuators)."""

from __future__ import annotations

from emet.motion.constants import stretch_degrees_of_freedom
from emet.robots.base import RobotSpec
from emet.robots.footprint import Footprint
from emet.robots.stretch import STRETCH_JOINT_NAMES
from emet.utils.assets import get_mujoco_models_path

_STRETCH_XML = str((get_mujoco_models_path() / "stretch.xml").resolve())


def get_stretch_robosuite_mjcf_spec() -> RobotSpec:
    """Spec for :class:`~emet.simulation.stretch_robosuite_server.StretchRobosuiteZmqServer` (merged Stretch MJCF).

    ``joint_names`` follow the Stretch ZMQ vector (``HelloStretchIdx``); base entries are abstract
    (no matching MuJoCo hinge) and are filled from the free joint in
    :meth:`StretchRobosuiteZmqServer.get_joint_state`. ``actuator_names`` align by index for
    :func:`~emet.simulation.mujoco_ctrl_sync.sync_actuator_ctrl_from_joint_positions` (empty strings
    skip). Diff-drive wheel velocity actuators are zeroed in ``StretchRobosuiteZmqServer``.
    """
    actuator_names = [
        "",
        "",
        "",
        "lift",
        "arm",
        "gripper",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
        "head_pan",
        "head_tilt",
    ]
    return RobotSpec(
        name="stretch",
        dof=stretch_degrees_of_freedom,
        joint_names=list(STRETCH_JOINT_NAMES),
        camera_names=["d435i_camera_rgb", "d405_rgb", "d435i_camera_depth", "d405_depth"],
        urdf_path=None,
        mjcf_path=_STRETCH_XML,
        actuator_names=actuator_names,
        base_link_name="base_link",
        footprint=Footprint(width=0.34, length=0.33, width_offset=0.0, length_offset=-0.1),
        robosuite_rgb_depth_ops=("rot90_cw",),
    )
