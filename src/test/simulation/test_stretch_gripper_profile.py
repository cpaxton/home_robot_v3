# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from unittest.mock import Mock

import mujoco
import pytest

from emet.simulation.stretch_mujoco.config import joint_position_rates
from emet.simulation.stretch_mujoco.datamodels.status_command import CommandMove, StatusCommand
from emet.simulation.stretch_mujoco.mujoco_server import MujocoServer
from emet.simulation.stretch_mujoco.position_targets import PositionTargets


@pytest.mark.parametrize("target", [-0.3, 0.56])
def test_bridge_sends_one_exact_gripper_target_without_sleep(target, monkeypatch):
    from emet.simulation import mujoco_server_stretch as bridge

    server = object.__new__(bridge.MujocoZmqServer)
    server.robot_sim = Mock()
    server.stop = Mock()  # This contract-only fixture starts no server threads.
    monkeypatch.setattr(bridge.time, "sleep", Mock(side_effect=AssertionError("wall-clock stepping")))
    server.handle_action({"gripper": target})
    server.robot_sim.move_to.assert_called_once_with("gripper", target)


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("dt", [0.002, 0.01])
def test_native_gripper_commands_share_clamped_physics_time_profile(relative, dt):
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody><body><joint name="slide" type="slide"/>
      <geom type="sphere" size="0.01"/></body></worldbody>
      <actuator><position name="gripper" joint="slide" ctrlrange="-0.02 0.04"/></actuator>
    </mujoco>""")
    model.opt.timestep = dt
    data = mujoco.MjData(model)
    server = object.__new__(MujocoServer)
    server.mjmodel, server.mjdata = model, data
    server.position_targets = PositionTargets(model, data, joint_position_rates)
    server.base_controller = Mock()
    server.data_proxies = Mock()
    command = StatusCommand()
    setter = command.set_move_by if relative else command.set_move_to
    setter(CommandMove("gripper", True, 100.0))
    server.push_command(command)
    assert server.position_targets.pending["gripper"] == 0.04
    step = joint_position_rates["gripper"] * dt
    assert data.ctrl[0] == pytest.approx(step)
    # A physics tick without a new message continues the profile.
    server.push_command(command)
    assert data.ctrl[0] == pytest.approx(2 * step)
    setter(CommandMove("gripper", True, -100.0))
    server.push_command(command)
    assert server.position_targets.pending["gripper"] == -0.02
    assert data.ctrl[0] == pytest.approx(step)


def test_full_gripper_reference_travel_fits_existing_client_deadline():
    # No increased timeout to hide a slow profile. Measured completion remains
    # the client's responsibility and may fail on obstruction.
    assert 0.06 / joint_position_rates["gripper"] < 10.0
