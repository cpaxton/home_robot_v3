# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from emet.simulation.stretch_mujoco import config, utils
from emet.simulation.stretch_mujoco.datamodels.status_command import CommandBaseVelocity, StatusCommand
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


def settle_reference(controller, velocity):
    # Target conversion/curvature assertions apply after the acceleration ramp.
    for _ in range(10000):
        previous = controller.mujoco_server.mjdata.ctrl.copy()
        controller._set_base_velocity(*velocity)
        if np.array_equal(previous, controller.mujoco_server.mjdata.ctrl):
            break


@pytest.mark.parametrize("gear", [1, 3, -2])
@pytest.mark.parametrize("velocity", [(0.05, 0.0), (0.0, 0.5), (0.0, -0.5), (0.0, 0.0)])
def test_wheel_commands_match_transmission_velocity(gear, velocity):
    model, data = wheel_model(gear)
    controller = BaseController(SimpleNamespace(mjmodel=model, mjdata=data))
    settle_reference(controller, velocity)
    wheels = np.asarray(utils.diff_drive_inv_kinematics(*velocity))
    np.testing.assert_allclose(data.ctrl, gear * wheels)
    # At the requested joint speeds, the velocity servo must exert no error force.
    data.qvel[:] = wheels
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.actuator_force, 0, atol=1e-12)


@pytest.mark.parametrize("velocity", [(0.3, 0.5), (-0.3, 0.5), (0.3, -0.5), (0, 2)])
@pytest.mark.parametrize("gear", [3, -2])
def test_saturated_wheels_preserve_commanded_curvature(velocity, gear):
    model, data = wheel_model(gear)
    model.actuator_ctrllimited[:] = 1
    model.actuator_ctrlrange[:] = [-3, 6]
    controller = BaseController(SimpleNamespace(mjmodel=model, mjdata=data))
    settle_reference(controller, velocity)
    assert np.all(data.ctrl >= -3) and np.all(data.ctrl <= 6)
    actual_twist = np.asarray(utils.diff_drive_fwd_kinematics(*(data.ctrl / gear)))
    desired = np.asarray(velocity)
    fraction = np.dot(actual_twist, desired) / np.dot(desired, desired)
    assert 0 < fraction < 1
    np.testing.assert_allclose(actual_twist, fraction * desired, atol=1e-12)


def test_common_wheel_limit_does_not_scale_an_in_range_command():
    model, data = wheel_model(3)
    model.actuator_ctrllimited[:] = 1
    model.actuator_ctrlrange[:] = [-6, 6]
    controller = BaseController(SimpleNamespace(mjmodel=model, mjdata=data))
    settle_reference(controller, (0.01, 0.02))
    np.testing.assert_allclose(data.ctrl, 3 * np.asarray(utils.diff_drive_inv_kinematics(0.01, 0.02)))


@pytest.mark.parametrize("gear", [1, 3, -2])
@pytest.mark.parametrize("dt", [0.002, 0.01])
def test_wheel_acceleration_is_in_joint_units_and_physics_time(gear, dt):
    model, data = wheel_model(gear)
    model.opt.timestep = dt
    controller = BaseController(SimpleNamespace(mjmodel=model, mjdata=data))
    for velocity in [(0.1, 0), (0, 1), (0, -1), (0, 0)]:
        previous = data.ctrl.copy()
        controller._set_base_velocity(*velocity)
        assert np.max(np.abs((data.ctrl - previous) / gear)) <= config.wheel_reference_acceleration * dt + 1e-12
    data.ctrl[:] = 10 * gear * config.wheel_reference_acceleration * dt
    controller._set_base_velocity(0, 0)
    assert np.any(data.ctrl != 0), "Ordinary zero commands must brake through the profile"
    for _ in range(100):
        controller._set_base_velocity(0, 0)
    np.testing.assert_array_equal(data.ctrl, 0)


def test_explicit_cancel_clears_reference_immediately():
    model, data = wheel_model(3)
    controller = BaseController(SimpleNamespace(mjmodel=model, mjdata=data))
    data.ctrl[:] = [6, -6]
    controller.last_command = object()
    command = StatusCommand()
    command.set_base_velocity(CommandBaseVelocity(0, 0, True, stop=True))
    # Exercise the same command serialization and consumption as cancellation.
    controller.push_command(StatusCommand.from_dict(command.to_dict()).base_velocity)
    assert controller.last_command is None
    np.testing.assert_array_equal(data.ctrl, 0)
    controller.update()
    np.testing.assert_array_equal(data.ctrl, 0)


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
