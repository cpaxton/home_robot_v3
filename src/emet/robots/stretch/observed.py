# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Stretch native observed-arm binding with the calibrated telescope coupling."""

from dataclasses import replace

import numpy as np

from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.stretch_encoder_kinematics import stretch_configuration_from_encoders
from emet.controller.task.tamp.observed_arm import ObservedArm
from emet.robots.stretch import STRETCH_ARM_CHAIN

_TELESCOPE = tuple(f"joint_arm_l{i}" for i in range(4))


def bind_stretch_arm(robot, model):
    chain = replace(STRETCH_ARM_CHAIN, joint_names=STRETCH_ARM_CHAIN.joint_names[:-1])

    def synchronize(data):
        observation = robot.get_observation()
        if observation is None:
            raise ObservationUnavailable("observation_unavailable")
        _, state = stretch_configuration_from_encoders(observation, model)
        data.qpos[:] = state.qpos

    def command_joints(target):
        # Hold the measured head and do not request any base displacement.
        observation = robot.get_observation()
        if observation is None:
            raise ObservationUnavailable("observation_unavailable")
        estimate, _ = stretch_configuration_from_encoders(observation, model)
        motors = estimate.motor_capture.as_dict()
        lift, *arm, yaw, pitch, roll = np.asarray(target, dtype=float)
        if not robot.in_manipulation_mode():
            robot.switch_to_manipulation_mode()
        return robot.arm_to(
            np.array([0.0, lift, sum(arm), roll, pitch, yaw]),
            head=np.array([motors["head_pan"], motors["head_tilt"]]),
            blocking=False,
            reliable=True,
        )

    def command_gripper_joints(target):
        robot.gripper_to(float(target[0]), blocking=False, reliable=True)
        return True

    return ObservedArm(
        model, "base_link", chain, synchronize, command_joints, command_gripper_joints,
        ("joint_gripper_slide",), coupled_groups=(_TELESCOPE,),
    )
