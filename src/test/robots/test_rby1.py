# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""Tests for RB-Y1 robot backend (same MJCF as Galaxea R1). Run without simulation conftest."""

import pytest


def test_rby1_spec():
    """Rby1Backend.get_spec() returns a valid RobotSpec."""
    pytest.importorskip("emet.robots.rby1")
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
    pytest.importorskip("emet.robots.rby1")
    import mujoco

    from emet.robots.rby1 import Rby1Backend

    spec = Rby1Backend().get_spec()
    model = mujoco.MjModel.from_xml_path(spec.mjcf_path)

    assert model.nq == 33  # 7 (freejoint) + 26 joints
    assert model.nu == 26

    for aname in spec.actuator_names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
        assert aid >= 0, f"Actuator '{aname}' not found in MJCF"

    for cname in spec.camera_names:
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cname)
        assert cid >= 0, f"MuJoCo camera '{cname}' missing (RobotSpec names must be mjOBJ_CAMERA for ZMQ render)"

    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    assert data.time > 0


def test_rby1_create_client():
    """Rby1Backend.create_client returns GenericZmqClient."""
    pytest.importorskip("emet.robots.rby1")
    from emet.controller.generic_zmq_client import GenericZmqClient
    from emet.robots.rby1 import Rby1Backend

    backend = Rby1Backend()
    client = backend.create_client("127.0.0.1", start_immediately=False, enable_rerun_server=False)
    assert isinstance(client, GenericZmqClient)
    assert client._spec.name == "rby1"


def test_rby1_create_client_defers_zmq_start_by_default():
    """Avoid blocking ZMQ wait in __init__ (RobotAgent.start() connects once after model setup)."""
    pytest.importorskip("emet.robots.rby1")
    from emet.robots.rby1 import Rby1Backend

    client = Rby1Backend().create_client("127.0.0.1", enable_rerun_server=False)
    assert not client._recv_threads_started
    assert not client._started


def test_decode_servo_message_robosuite_format():
    """Servo ZMQ dict from RobosuiteZmqServer decodes to Observations for Rerun."""
    import numpy as np

    import emet.utils.compression as compression
    from emet.controller.generic_zmq_client import _decode_servo_message_to_observations

    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    depth_u16 = (np.ones((48, 64), dtype=np.float32) * 0.5 * 1000).astype(np.uint16)
    msg = {
        "head_color_image": compression.to_jpg(rgb),
        "head_depth_image": compression.to_jp2(depth_u16),
        "head_camera_K": np.eye(3, dtype=np.float64),
        "joint_positions": np.zeros(5, dtype=np.float64),
        "base_pose": np.array([1.0, 2.0, 0.5], dtype=np.float64),
    }
    obs = _decode_servo_message_to_observations(msg, None, None)
    assert obs is not None
    assert obs.rgb.shape[0] == 48 and obs.rgb.shape[2] == 3
    assert obs.depth is not None and obs.depth.shape == (48, 64)
    assert obs.joint is not None and obs.joint.shape == (5,)
    assert abs(float(obs.gps[0]) - 1.0) < 1e-5
    assert obs.camera_pose is None

    T = np.eye(4, dtype=np.float64)
    T[0, 3] = 3.0
    msg_pose = {**msg, "camera_pose": T}
    obs2 = _decode_servo_message_to_observations(msg_pose, None, None)
    assert obs2 is not None and obs2.camera_pose is not None
    assert abs(float(obs2.camera_pose[0, 3]) - 3.0) < 1e-8

    full = {"camera_pose": np.diag([2.0, 2.0, 2.0, 1.0])}
    obs3 = _decode_servo_message_to_observations(msg, None, full)
    assert obs3 is not None and obs3.camera_pose is not None
    assert float(obs3.camera_pose[0, 0]) == 2.0


def test_default_scene_with_rby1_loads_and_robot_can_be_commanded():
    """Load default scene with rby1; apply joint commands via server handle_action and step; assert sim advances and state changes."""
    import numpy as np
    import pytest

    pytest.importorskip("mujoco")
    import mujoco

    from emet.robots.rby1 import Rby1Backend
    from emet.simulation.mujoco_server import _load_default_scene_with_robot
    from emet.simulation.robosuite_server import RobosuiteZmqServer

    model = _load_default_scene_with_robot("rby1")
    if model is None:
        pytest.skip("scene_environment.xml or rby1 MJCF not found (run from repo with assets)")

    backend = Rby1Backend()
    spec = backend.get_spec()
    server = RobosuiteZmqServer(
        robot_spec=spec,
        scene_model=model,
        send_port=0,
        recv_port=0,
        send_state_port=0,
        send_servo_port=0,
        mujoco_stationary_control=backend.create_mujoco_stationary_control(),
    )
    server._load_model()
    server._stabilize_physics_state_after_load()
    server._initial_xyt = server.get_base_xyt()

    # Get initial joint state and base position
    q0, _, _ = server.get_joint_state()
    xyt0 = server.get_base_xyt().copy()

    # Command arm/torso actuators (indices 6-9 torso, 10-15 left arm) so something moves
    joint_cmd = np.zeros(spec.dof)
    joint_cmd[6:10] = 0.1  # torso
    joint_cmd[10:16] = 0.05  # left arm
    server.handle_action({"joint": joint_cmd.tolist()})

    # Step simulation (server's internal loop would do this; we step directly)
    for _ in range(100):
        mujoco.mj_step(server._mjmodel, server._mjdata)

    q1, _, _ = server.get_joint_state()
    xyt1 = server.get_base_xyt()

    assert server._mjdata.time > 0, "Simulation time should advance"
    # Joint commands should change joint state (or base if we had moved it)
    q_changed = np.linalg.norm(np.array(q1) - np.array(q0)) > 1e-5
    base_changed = np.linalg.norm(xyt1[:2] - xyt0[:2]) > 1e-5
    assert q_changed or base_changed, (
        "Commanding joints and stepping should change state; "
        f"|q1-q0|={np.linalg.norm(np.array(q1) - np.array(q0)):.4f}, "
        f"base_xy_delta={np.linalg.norm(xyt1[:2] - xyt0[:2]):.4f}"
    )
