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


def test_generic_zmq_mapping_depth_copy_on_store():
    """Fused depth is copied on store so voxel/Rerun threads do not share a mutable buffer."""
    from threading import Lock

    import numpy as np

    from emet.controller.generic_zmq_client import GenericZmqClient

    c = GenericZmqClient.__new__(GenericZmqClient)
    c._mapping_depth_lock = Lock()
    c._mapping_depth_for_rerun = None
    d = np.ones((2, 4), dtype=np.float32)
    c.set_mapping_depth_for_rerun(d)
    d[0, 0] = 7.0
    out = c.peek_mapping_depth_for_rerun()
    assert out is not None and float(out[0, 0]) == 1.0
    c.set_mapping_depth_for_rerun(None)
    assert c.peek_mapping_depth_for_rerun() is None


def test_stretch_zmq_mapping_depth_copy_on_store():
    from threading import Lock

    import numpy as np

    from emet.controller.zmq_client import StretchZmqClient

    c = StretchZmqClient.__new__(StretchZmqClient)
    c._mapping_depth_lock = Lock()
    c._mapping_depth_for_rerun = None
    d = np.ones((3, 2), dtype=np.float32)
    c.set_mapping_depth_for_rerun(d)
    d[1, 1] = 8.0
    out = c.peek_mapping_depth_for_rerun()
    assert out is not None and float(out[1, 1]) == 1.0


def test_stretch_zmq_client_init_before_mapping_depth_lock_regression():
    """Regression: __init__ must not call reset() before _mapping_depth_lock exists."""
    from unittest.mock import MagicMock, patch

    from emet.controller.zmq_client import StretchZmqClient

    with patch.object(StretchZmqClient, "_create_recv_socket", return_value=MagicMock()):
        client = StretchZmqClient(
            robot_ip="127.0.0.1",
            start_immediately=False,
            enable_rerun_server=False,
        )
    client.reset()
    assert client._mapping_depth_lock is not None
    assert client.peek_mapping_depth_for_rerun() is None


def test_stretch_zmq_client_backward_compat():
    """HomeRobotZmqClient alias still works."""
    from emet.controller import HomeRobotZmqClient, StretchZmqClient

    assert HomeRobotZmqClient is StretchZmqClient


def test_robosuite_server_import():
    """RobosuiteZmqServer can be imported."""
    from emet.simulation.robosuite_server import RobosuiteZmqServer

    assert RobosuiteZmqServer is not None


def test_want_robocasa_planar_autoplace_innate_mars():
    from emet.robots.innate_mars import InnateMarsBackend
    from emet.simulation import scene_base_spawn

    spec = InnateMarsBackend().get_spec()
    assert scene_base_spawn.want_robocasa_planar_autoplace(
        environment={"kind": "robocasa", "task": "PickPlaceCounterToCabinet"},
        robot_spec=spec,
    )
    assert not scene_base_spawn.want_robocasa_planar_autoplace(
        environment={"kind": "default_table"},
        robot_spec=spec,
    )


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


def test_xml_remove_all_tags_strips_key():
    """xml_remove_all_tags removes every ``<key>`` (used after Robocasa strip-replace)."""
    from emet.simulation.stretch_mujoco.utils import xml_remove_all_tags

    xml = (
        '<mujoco model="t"><key name="k" qpos="0 1 2"/><worldbody><body name="b"/></worldbody>'
        '<key qpos="3 4 5"/></mujoco>'
    )
    out = xml_remove_all_tags(xml, "key")
    assert "<key" not in out
    assert "worldbody" in out


def test_robocasa_gen_robosuite_mapping():
    """_robosuite_robot_for maps names correctly."""
    from emet.simulation.stretch_mujoco.robocasa_gen import _robosuite_robot_for

    assert _robosuite_robot_for("stretch") == "PandaMobile"
    assert _robosuite_robot_for("PandaOmron") == "PandaOmron"
    assert _robosuite_robot_for("Tiago") == "Tiago"
    assert _robosuite_robot_for("GR1") == "GR1"
