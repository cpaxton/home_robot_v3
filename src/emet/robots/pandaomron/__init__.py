# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""PandaOmron as used by the native RoboCasa kitchen environment."""

from math import pi

from emet.robots.base import ArmChain, RobotBackend, RobotSpec
from emet.robots.footprint import Footprint

ARM_JOINTS = tuple(f"robot0_joint{i}" for i in range(1, 8))
TORSO_JOINT = "mobilebase0_joint_torso_height"
FINGER_JOINTS = tuple(f"gripper0_right_finger_joint{i}" for i in (1, 2))
CONTROL_JOINTS = (*ARM_JOINTS, TORSO_JOINT)
BASE_BODY = "mobilebase0_base"
GRASP_BODY = "gripper0_right_eef"
FRONT_CAMERA = "robot0_agentview_right"
WRIST_CAMERA = "robot0_eye_in_hand"
SCENE_CAMERAS = {
    side: f"robot0_agentview_{side}" for side in ("right", "left", "center")
}

PANDA_ARM_CHAIN = ArmChain(
    joint_names=CONTROL_JOINTS,
    ee_body=GRASP_BODY,
    actuator_names=CONTROL_JOINTS,
    gripper_bodies=("gripper0_right_right_gripper",),
    home_arm_q=(0., pi / 16, 0., -pi / 2 - pi / 3, 0., pi - .2, pi / 4, .16),
    # Keep the measured joints inside the soft mechanical stops when open.
    gripper_open_configuration=tuple(zip(FINGER_JOINTS, (0.039, -0.039), strict=True)),
)


class PandaOmronBackend(RobotBackend):
    """Simulation backend for RoboCasa's Panda arm and Omron mobile base."""

    def get_spec(self) -> RobotSpec:
        joints = (*CONTROL_JOINTS, *FINGER_JOINTS)
        return RobotSpec(
            name="pandaomron",
            dof=len(joints),
            joint_names=list(joints),
            actuator_names=list(CONTROL_JOINTS),
            camera_names=[FRONT_CAMERA, WRIST_CAMERA],
            urdf_path=None,
            base_link_name=BASE_BODY,
            arm_chain=PANDA_ARM_CHAIN,
            footprint=Footprint(width=0.60, length=0.90, width_offset=0.0, length_offset=-0.20),
            optional_uv_extras=("sim",),
        )

    def create_client(self, robot_ip: str, **kwargs):
        from emet.robots.pandaomron.client import PandaOmronClient

        return PandaOmronClient(robot_spec=self.get_spec(), robot_ip=robot_ip, **kwargs)

    def bind_observed_arm(self, robot, model):
        from emet.robots.pandaomron.observed import bind_pandaomron_arm

        return bind_pandaomron_arm(robot, model)

    def create_model(self, **kwargs):
        raise NotImplementedError("PandaOmron planning geometry comes from the native RoboCasa environment")
