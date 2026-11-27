# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""PandaOmron mobile manipulator backend for RoboCasa scenes.

The RoboCasa benchmark robot is a Panda arm on an Omron mobile base. RoboCasa
already puts this robot in the generated kitchen scene, so the backend does not
ship or merge an MJCF. ``robocasa_gen`` converts the Panda torque motors to
position servos for the kinematic TAMP executor; the mobile-base velocity
actuators stay velocity-controlled and are excluded from arm holds.
"""

from __future__ import annotations

import math

from emet.robots.base import ArmChain, RobotBackend, RobotSpec
from emet.robots.footprint import Footprint

PANDA_OMRON_BASE_JOINTS = (
    "mobilebase0_joint_mobile_forward",
    "mobilebase0_joint_mobile_side",
    "mobilebase0_joint_mobile_yaw",
)
PANDA_OMRON_BASE_ACTUATORS = (
    "mobilebase0_actuator_mobile_forward",
    "mobilebase0_actuator_mobile_side",
    "mobilebase0_actuator_mobile_yaw",
)
PANDA_OMRON_ARM_JOINTS = tuple(f"robot0_joint{i}" for i in range(1, 8))
PANDA_OMRON_ARM_ACTUATORS = tuple(f"robot0_joint{i}_act" for i in range(1, 8))
PANDA_OMRON_JOINT_NAMES = [
    *PANDA_OMRON_BASE_JOINTS,
    "mobilebase0_joint_torso_height",
    *PANDA_OMRON_ARM_JOINTS,
    "gripper0_right_finger_joint1",
    "gripper0_right_finger_joint2",
]
PANDA_OMRON_ACTUATOR_NAMES = [
    *PANDA_OMRON_BASE_ACTUATORS,
    "mobilebase0_actuator_torso_height",
    *PANDA_OMRON_ARM_ACTUATORS,
    "gripper0_right_gripper_finger_joint1",
    "gripper0_right_gripper_finger_joint2",
]


def _arm_chain() -> ArmChain:
    return ArmChain(
        joint_names=PANDA_OMRON_ARM_JOINTS,
        ee_body="gripper0_right_eef",
        actuator_names=PANDA_OMRON_ARM_ACTUATORS,
        link_bodies=tuple(f"robot0_link{i}" for i in range(1, 8)),
        gripper_bodies=(
            "robot0_right_hand",
            "gripper0_right_right_gripper",
            "gripper0_right_leftfinger",
            "gripper0_right_rightfinger",
        ),
        # PandaOmron.init_qpos from robosuite.
        home_arm_q=(
            0.0,
            math.pi / 16.0 - 0.2,
            0.0,
            -math.pi / 2.0 - math.pi / 3.0,
            0.0,
            math.pi - 0.4,
            math.pi / 4.0,
        ),
        gripper_open_configuration=(
            ("gripper0_right_finger_joint1", 0.04),
            ("gripper0_right_finger_joint2", -0.04),
        ),
        gripper_open=0.04,
        gripper_closed=0.0,
    )


class PandaOmronBackend(RobotBackend):
    """Panda arm + Omron planar base, using the robot embedded in a RoboCasa scene."""

    def get_spec(self) -> RobotSpec:
        return RobotSpec(
            name="panda_omron",
            dof=len(PANDA_OMRON_JOINT_NAMES),
            joint_names=list(PANDA_OMRON_JOINT_NAMES),
            camera_names=["agentview", "robot0_eye_in_hand"],
            urdf_path=None,
            mjcf_path=None,
            scene_provides_mjcf=True,
            actuator_names=list(PANDA_OMRON_ACTUATOR_NAMES),
            base_actuator_names=PANDA_OMRON_BASE_ACTUATORS,
            base_link_name="mobilebase0_base",
            footprint=Footprint(width=0.50, length=0.50, width_offset=0.0, length_offset=0.0),
            planar_base_joint_names=PANDA_OMRON_BASE_JOINTS,
            default_arm="right",
            live_ik_correction=False,
            grasp_lift_verify_tol_m=0.12,
            placement_retract_m=0.12,
            arm_chains={"right": _arm_chain()},
            advertise_kinematic_manip=True,
            tamp_approach="front",
        )

    def create_client(self, robot_ip: str, **kwargs):
        from emet.controller.generic_zmq_client import GenericZmqClient

        opts = dict(kwargs)
        opts.setdefault("start_immediately", False)
        return GenericZmqClient(robot_spec=self.get_spec(), robot_ip=robot_ip, **opts)

    def create_model(self, **kwargs):
        raise NotImplementedError(
            "PandaOmron uses the robot embedded in the RoboCasa scene; "
            "kinematic planning runs against that scene model."
        )


__all__ = ["PandaOmronBackend"]
