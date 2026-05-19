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

"""Galaxea R1 robot — two-armed mobile manipulator with swerve drive.

Reference: https://github.com/userguide-galaxea/URDF
"""

from pathlib import Path

from emet.robots.base import RobotBackend, RobotSpec
from emet.robots.footprint import Footprint

_ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets" / "robot" / "galaxea_r1"
_MJCF_PATH = str(_ASSETS_DIR / "galaxea_r1.xml")

R1_JOINT_NAMES = [
    # Swerve base (3 steer + 3 wheel)
    "steer_motor_joint1",
    "wheel_motor_joint1",
    "steer_motor_joint2",
    "wheel_motor_joint2",
    "steer_motor_joint3",
    "wheel_motor_joint3",
    # Torso (4 DOF)
    "torso_joint1",
    "torso_joint2",
    "torso_joint3",
    "torso_joint4",
    # Left arm (6 DOF)
    "left_arm_joint1",
    "left_arm_joint2",
    "left_arm_joint3",
    "left_arm_joint4",
    "left_arm_joint5",
    "left_arm_joint6",
    # Left gripper (2 fingers)
    "left_gripper_finger_joint1",
    "left_gripper_finger_joint2",
    # Right arm (6 DOF)
    "right_arm_joint1",
    "right_arm_joint2",
    "right_arm_joint3",
    "right_arm_joint4",
    "right_arm_joint5",
    "right_arm_joint6",
    # Right gripper (2 fingers)
    "right_gripper_finger_joint1",
    "right_gripper_finger_joint2",
]

R1_ACTUATOR_NAMES = [
    "steer1",
    "wheel1",
    "steer2",
    "wheel2",
    "steer3",
    "wheel3",
    "torso1",
    "torso2",
    "torso3",
    "torso4",
    "left_arm1",
    "left_arm2",
    "left_arm3",
    "left_arm4",
    "left_arm5",
    "left_arm6",
    "left_gripper1",
    "left_gripper2",
    "right_arm1",
    "right_arm2",
    "right_arm3",
    "right_arm4",
    "right_arm5",
    "right_arm6",
    "right_gripper1",
    "right_gripper2",
]

R1_CAMERA_NAMES = ["zed_camera", "left_camera", "right_camera"]


class GalaxeaR1Backend(RobotBackend):
    """Galaxea R1 two-armed mobile manipulator backend."""

    def get_spec(self) -> RobotSpec:
        return RobotSpec(
            name="galaxea_r1",
            dof=26,
            joint_names=R1_JOINT_NAMES,
            camera_names=R1_CAMERA_NAMES,
            urdf_path=None,
            mjcf_path=_MJCF_PATH,
            actuator_names=R1_ACTUATOR_NAMES,
            base_link_name="base_link",
            footprint=Footprint(width=0.56, length=0.50, width_offset=0.0, length_offset=0.0),
        )

    def create_client(self, robot_ip: str, **kwargs):
        from emet.controller.generic_zmq_client import GenericZmqClient

        # Match StretchZmqClient: defer ZMQ recv until RobotAgent.start() so we do not block here twice
        # (and so a slow model load in DynamemTaskExecutor does not consume the wait before the sim is up).
        opts = dict(kwargs)
        opts.setdefault("start_immediately", False)
        return GenericZmqClient(robot_spec=self.get_spec(), robot_ip=robot_ip, **opts)

    def create_model(self, **kwargs):
        raise NotImplementedError(
            "Galaxea R1 kinematic model not yet implemented. Use MuJoCo-based IK or a third-party IK solver."
        )


__all__ = ["GalaxeaR1Backend"]
