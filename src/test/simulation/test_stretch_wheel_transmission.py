# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from emet.simulation.stretch_mujoco import utils
from emet.simulation.stretch_mujoco.mujoco_server import BaseController, MujocoServer


def wheel_model(gear):
    model = mujoco.MjModel.from_xml_string(f"""
    <mujoco>
      <worldbody>
        <body><joint name="joint_left_wheel" type="hinge"/>
          <geom type="sphere" size="0.05"/></body>
        <body pos="1 0 0"><joint name="joint_right_wheel" type="hinge"/>
          <geom type="sphere" size="0.05"/></body>
      </worldbody>
      <actuator>
        <velocity name="left_wheel_vel" joint="joint_left_wheel" gear="{gear}" kv="20"/>
        <velocity name="right_wheel_vel" joint="joint_right_wheel" gear="{gear}" kv="20"/>
      </actuator>
    </mujoco>
    """)
    return model, mujoco.MjData(model)


@pytest.mark.parametrize("gear", [1, 3, -2])
@pytest.mark.parametrize("velocity", [(0.05, 0.0), (0.0, 0.5), (0.0, -0.5), (0.0, 0.0)])
def test_wheel_commands_match_transmission_velocity(gear, velocity):
    model, data = wheel_model(gear)
    controller = BaseController(SimpleNamespace(mjmodel=model, mjdata=data))
    controller._set_base_velocity(*velocity)
    wheels = np.asarray(utils.diff_drive_inv_kinematics(*velocity))
    np.testing.assert_allclose(data.ctrl, gear * wheels)
    # At the requested joint speeds, the velocity servo must exert no error force.
    data.qvel[:] = wheels
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.actuator_force, 0, atol=1e-12)


@pytest.mark.parametrize("gear", [1, 3, -2])
def test_base_velocity_telemetry_uses_joint_not_actuator_speed(gear):
    model, data = wheel_model(gear)
    data.qvel[:] = [1.2, 0.4]
    mujoco.mj_forward(model, data)
    server = object.__new__(MujocoServer)
    # Other status fields are unrelated to the two actual MuJoCo wheel joints.
    server.mjdata = SimpleNamespace(
        time=0.0,
        joint=data.joint,
        actuator=lambda name: SimpleNamespace(length=[0.0], velocity=[0.0]),
    )
    server.physics_fps_counter = SimpleNamespace(fps=100, sim_to_real_time_ratio_msg="1", sim_to_real_ratio=1.0)
    server.base_controller = Mock()
    server.base_controller.get_base_pose.return_value = np.zeros(3)
    server.data_proxies = Mock()
    server.pull_status()
    status = server.data_proxies.set_status.call_args.args[0]
    np.testing.assert_allclose([status.base.x_vel, status.base.theta_vel], utils.diff_drive_fwd_kinematics(1.2, 0.4))
