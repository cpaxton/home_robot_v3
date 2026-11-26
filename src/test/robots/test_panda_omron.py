# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from emet.motion.arm_manip_profile import ArmManipProfile
from emet.robots import get_robot_spec


def test_panda_omron_uses_the_scene_robot_and_right_arm():
    spec = get_robot_spec("panda_omron")
    assert spec is not None
    assert spec.scene_provides_mjcf is True
    assert spec.default_arm == "right"
    assert spec.base_link_name == "mobilebase0_base"
    assert spec.planar_base_joint_names == (
        "mobilebase0_joint_mobile_forward",
        "mobilebase0_joint_mobile_side",
        "mobilebase0_joint_mobile_yaw",
    )
    assert spec.arm_chain is None
    assert "right" in spec.arm_chains

    profile = ArmManipProfile.for_robot("panda_omron", arm="right")
    assert profile.ee_body == "gripper0_right_eef"
    assert profile.joint_names == tuple(f"robot0_joint{i}" for i in range(1, 8))
    assert profile.actuator_names == tuple(f"robot0_joint{i}_act" for i in range(1, 8))
    assert profile.gripper_open_configuration == (
        ("gripper0_right_finger_joint1", 0.04),
        ("gripper0_right_finger_joint2", -0.04),
    )
