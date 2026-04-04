# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""Tests for multi-robot support: RobotSpec, GenericZmqClient, RobosuiteZmqServer."""


def test_galaxea_r1_spec():
    """GalaxeaR1Backend.get_spec() returns a valid RobotSpec."""
    from emet.robots.galaxea_r1 import GalaxeaR1Backend

    backend = GalaxeaR1Backend()
    spec = backend.get_spec()
    assert spec.name == "galaxea_r1"
    assert spec.dof == 26
    assert len(spec.joint_names) == 26
    assert len(spec.actuator_names) == 26
    assert len(spec.camera_names) == 3
    assert spec.mjcf_path is not None
    assert spec.base_link_name == "base_link"
    assert spec.footprint.width > 0


def test_stretch_spec():
    """StretchBackend.get_spec() returns a valid RobotSpec."""
    from emet.robots.stretch import StretchBackend

    backend = StretchBackend()
    spec = backend.get_spec()
    assert spec.name == "stretch"
    assert spec.dof > 0
    assert len(spec.joint_names) > 0
    assert spec.urdf_path is not None


def test_robot_registry():
    """ROBOT_REGISTRY contains expected robots."""
    from emet.robots import ROBOT_REGISTRY

    assert "stretch" in ROBOT_REGISTRY
    assert "galaxea_r1" in ROBOT_REGISTRY
    assert "rby1" in ROBOT_REGISTRY  # Rainbow RB-Y1 (Galaxea R1 family), MolmoSpaces id
    assert "rb_y1" in ROBOT_REGISTRY


def test_galaxea_r1_mjcf_loads():
    """The Galaxea R1 MJCF loads in MuJoCo and has correct structure."""
    import mujoco

    from emet.robots.galaxea_r1 import GalaxeaR1Backend

    spec = GalaxeaR1Backend().get_spec()
    model = mujoco.MjModel.from_xml_path(spec.mjcf_path)

    assert model.nq == 33  # 7 (freejoint) + 26 joints
    assert model.nu == 26

    for aname in spec.actuator_names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
        assert aid >= 0, f"Actuator '{aname}' not found in MJCF"

    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    assert data.time > 0


def test_rby1_spec():
    """Rby1Backend.get_spec() returns a valid RobotSpec (same structure as Galaxea R1)."""
    from emet.robots.rby1 import Rby1Backend

    backend = Rby1Backend()
    spec = backend.get_spec()
    assert spec.name == "rby1"
    assert spec.dof == 26
    assert len(spec.joint_names) == 26
    assert len(spec.actuator_names) == 26
    assert len(spec.camera_names) == 3
    assert spec.mjcf_path is not None
    assert spec.base_link_name == "base_link"
    assert spec.footprint.width > 0


def test_rby1_mjcf_loads():
    """The RB-Y1 MJCF loads in MuJoCo (same file as Galaxea R1)."""
    import mujoco

    from emet.robots.rby1 import Rby1Backend

    spec = Rby1Backend().get_spec()
    model = mujoco.MjModel.from_xml_path(spec.mjcf_path)

    assert model.nq == 33  # 7 (freejoint) + 26 joints
    assert model.nu == 26

    for aname in spec.actuator_names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
        assert aid >= 0, f"Actuator '{aname}' not found in MJCF"

    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    assert data.time > 0


def test_generic_zmq_client_import():
    """GenericZmqClient can be imported."""
    from emet.controller.generic_zmq_client import GenericZmqClient

    assert GenericZmqClient is not None


def test_stretch_zmq_client_backward_compat():
    """HomeRobotZmqClient alias still works."""
    from emet.controller import HomeRobotZmqClient, StretchZmqClient

    assert HomeRobotZmqClient is StretchZmqClient


def test_robosuite_server_import():
    """RobosuiteZmqServer can be imported."""
    from emet.simulation.robosuite_server import RobosuiteZmqServer

    assert RobosuiteZmqServer is not None


def test_robocasa_gen_robot_param():
    """model_generation_wizard accepts robot parameter (import-only test)."""
    import inspect

    from emet.simulation.stretch_mujoco.robocasa_gen import model_generation_wizard

    sig = inspect.signature(model_generation_wizard)
    assert "robot" in sig.parameters


def test_robot_spec_new_fields():
    """RobotSpec has mjcf_path, actuator_names, base_link_name fields."""
    from emet.robots.base import RobotSpec
    from emet.robots.footprint import Footprint

    spec = RobotSpec(
        name="test",
        dof=6,
        joint_names=["j1", "j2", "j3", "j4", "j5", "j6"],
        camera_names=["cam"],
        urdf_path=None,
        footprint=Footprint(width=0.5, length=0.5),
        mjcf_path="/tmp/test.xml",
        actuator_names=["a1", "a2"],
        base_link_name="base",
    )
    assert spec.mjcf_path == "/tmp/test.xml"
    assert spec.actuator_names == ["a1", "a2"]
    assert spec.base_link_name == "base"


def test_robocasa_gen_robosuite_mapping():
    """_robosuite_robot_for maps names correctly."""
    from emet.simulation.stretch_mujoco.robocasa_gen import _robosuite_robot_for

    assert _robosuite_robot_for("stretch") == "PandaMobile"
    assert _robosuite_robot_for("PandaOmron") == "PandaOmron"
    assert _robosuite_robot_for("Tiago") == "Tiago"
    assert _robosuite_robot_for("GR1") == "GR1"
