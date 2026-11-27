# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import xml.etree.ElementTree as ET

import pytest

mujoco = pytest.importorskip("mujoco")

from emet.controller.manipulation.kinematic_pick_place import KinematicPickPlaceExecutor
from emet.motion.arm_manip_profile import ArmManipProfile
from emet.robots import get_robot_spec
from emet.simulation.stretch_mujoco.robocasa_gen import _panda_omron_position_servos


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
    assert spec.placement_retract_m == 0.12


def test_panda_gripper_configuration_resolves_to_actuators():
    """The config is joint-named; _set_gripper must map it to actuator names."""

    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="hand">
              <joint name="gripper0_right_finger_joint1" type="slide" range="-0.04 0.04"/>
              <geom name="finger1" type="box" size="0.01 0.01 0.01"/>
              <joint name="gripper0_right_finger_joint2" type="slide" range="-0.04 0.04"/>
              <geom name="finger2" type="box" size="0.01 0.01 0.01"/>
            </body>
          </worldbody>
          <actuator>
            <position name="gripper0_right_gripper_finger_joint1"
                      joint="gripper0_right_finger_joint1" kp="1"/>
            <position name="gripper0_right_gripper_finger_joint2"
                      joint="gripper0_right_finger_joint2" kp="1"/>
          </actuator>
        </mujoco>
        """
    )

    class _Robot:
        _spec = get_robot_spec("panda_omron")

        def __init__(self) -> None:
            self.sent: dict[str, float] = {}

        def get_joint_state(self, timeout=1.0):
            return None, None, None

        def set_actuator_positions(self, values):
            self.sent = dict(values)

    robot = _Robot()
    executor = object.__new__(KinematicPickPlaceExecutor)
    executor.robot = robot
    executor.profile = ArmManipProfile.for_robot("panda_omron", arm="right")
    executor._model = model
    executor._data = mujoco.MjData(model)
    executor.arm = "right"
    executor._set_gripper(open_=True)
    assert robot.sent == {
        "gripper0_right_gripper_finger_joint1": 0.04,
        "gripper0_right_gripper_finger_joint2": -0.04,
    }


def test_panda_torso_motor_becomes_a_position_servo():
    xml = """
    <mujoco>
      <worldbody>
        <body name="base">
          <joint name="mobilebase0_joint_torso_height" type="slide" range="0 0.34"/>
          <geom name="torso" type="box" size="0.1 0.1 0.1"/>
        </body>
      </worldbody>
      <actuator>
        <motor name="mobilebase0_actuator_torso_height"
               joint="mobilebase0_joint_torso_height" forcelimited="false"/>
      </actuator>
    </mujoco>
    """
    converted = ET.fromstring(_panda_omron_position_servos(xml))
    actuator = converted.find("actuator")
    assert actuator is not None
    torso = actuator.find("position[@name='mobilebase0_actuator_torso_height']")
    assert torso is not None
    assert torso.get("joint") == "mobilebase0_joint_torso_height"
    assert torso.get("ctrlrange") == "0 0.34"
    assert torso.get("forcelimited") == "false"


def test_panda_arm_servo_conversion_preserves_order_and_torque_authority():
    xml = """
    <mujoco>
      <worldbody>
        <body name="base">
          <joint name="robot0_joint1" type="hinge" range="-2.9 2.9"/>
          <joint name="robot0_joint2" type="hinge" range="-1.7 1.7"/>
          <geom name="arm" type="box" size="0.1 0.1 0.1"/>
        </body>
      </worldbody>
      <actuator>
        <motor name="robot0_torq_j1" joint="robot0_joint1" ctrlrange="-80 80"/>
        <position name="gripper" joint="robot0_joint2" kp="1"/>
        <motor name="robot0_torq_j2" joint="robot0_joint2" ctrlrange="-12 12"/>
      </actuator>
    </mujoco>
    """
    converted = ET.fromstring(_panda_omron_position_servos(xml))
    actuator = converted.find("actuator")
    assert actuator is not None
    children = list(actuator)
    assert [child.get("name") for child in children] == [
        "robot0_joint1_act",
        "gripper",
        "robot0_joint2_act",
    ]
    assert children[0].tag == "position"
    assert children[0].get("forcelimited") == "true"
    assert children[0].get("forcerange") == "-80 80"
    assert children[2].get("forcerange") == "-12 12"
    assert children[1].tag == "position"


def test_scene_model_cache_is_content_addressed_and_same_directory(tmp_path):
    from emet.simulation.mujoco_server import _write_scene_model_cache

    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body name='b'><geom size='.1'/></body></worldbody></mujoco>"
    )
    first = _write_scene_model_cache(model, tmp_path)
    second = _write_scene_model_cache(model, tmp_path)
    assert first == second
    assert first.parent == tmp_path
    assert first.is_file()
    assert not list(tmp_path.glob("emet_scene_*.mjb.tmp"))
    assert len(list(tmp_path.glob("emet_scene_*.mjb"))) == 1


def test_scene_model_load_failure_falls_back_without_raising(tmp_path):
    from emet.controller.manipulation.kinematic_pick_place import KinematicPickPlaceExecutor

    broken = tmp_path / "broken.mjb"
    broken.write_bytes(b"not a mujoco model")

    class _Spec:
        scene_provides_mjcf = True
        mjcf_path = None
        base_link_name = "mobilebase0_base"

    class _Robot:
        _spec = _Spec()

        def get_emet_session(self):
            return {"scene_model_path": str(broken)}

    executor = object.__new__(KinematicPickPlaceExecutor)
    executor.robot = _Robot()
    executor._model = None
    executor._data = None
    executor._scene_model_loaded = False
    executor.manip_collision = "none"
    executor.voxel_map = None
    assert executor._ensure_model() is False
    assert executor._model is None
