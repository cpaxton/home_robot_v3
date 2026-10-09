# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Physics-process diagnostics survive the Stretch IPC and ZMQ boundary."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from emet.simulation.mujoco_server_stretch import MujocoZmqServer
from emet.simulation.stretch_mujoco.datamodels.status_stretch_joints import StatusStretchJoints
from emet.simulation.stretch_mujoco.mujoco_server import MujocoServer


def test_physics_status_copies_pose_and_named_controls():
    controls = np.array([0.25, -0.4])
    position = np.array([1.0, 2.0, 0.1])
    rotation = np.eye(3)
    rotation[2, 2] = 0.8
    actuator = SimpleNamespace(length=[0.1], velocity=[0.0])
    proxy = Mock()
    physics = SimpleNamespace(
        mjdata=SimpleNamespace(
            time=12.0,
            ctrl=controls,
            actuator=lambda _: actuator,
            body=lambda _: SimpleNamespace(xpos=position, xmat=rotation),
        ),
        mjmodel=SimpleNamespace(nu=2, actuator=lambda i: SimpleNamespace(name=["lift", "head_tilt"][i])),
        physics_fps_counter=SimpleNamespace(fps=100, sim_to_real_time_ratio_msg="1", sim_to_real_ratio=1.0),
        base_controller=SimpleNamespace(get_base_pose=lambda: [1.0, 2.0, 0.0]),
        _to_real_gripper_range=lambda x: x,
        data_proxies=proxy,
    )
    MujocoServer.pull_status(physics)
    status = proxy.set_status.call_args.args[0]
    controls[:] = 9
    position[:] = 9
    restored = StatusStretchJoints.from_dict(status.to_dict())
    assert restored.base_xyz == [1.0, 2.0, 0.1]
    assert restored.base_up_dot_world_z == 0.8
    assert restored.actuator_targets == [0.25, -0.4]
    assert restored.actuator_names == ["lift", "head_tilt"]


def test_zmq_state_forwards_diagnostics_and_preserves_missing_legacy_values():
    status = StatusStretchJoints.default()
    server = SimpleNamespace(
        _status=status,
        _last_step=1,
        _stretch_sim_publish_ok=lambda: True,
        get_joint_state=lambda: ([], [], []),
        get_base_pose=lambda: [0, 0, 0],
        get_ee_pose=lambda: np.eye(4),
        get_control_mode=lambda: "navigation",
        base_controller_at_goal=lambda: True,
        get_robot_spec=lambda: SimpleNamespace(name="stretch"),
        _attach_emet_session=lambda msg: msg,
    )
    assert MujocoZmqServer.get_state_message(server)["base_up_dot_world_z"] is None
    status.base_xyz = [1, 2, 0.1]
    status.base_up_dot_world_z = 0.8
    status.actuator_targets = [0.2]
    status.actuator_names = ["lift"]
    message = MujocoZmqServer.get_state_message(server)
    for key in ("base_xyz", "base_up_dot_world_z", "actuator_targets", "actuator_names"):
        assert message[key] == getattr(status, key)
