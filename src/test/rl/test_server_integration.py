import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from emet.core.command_client import command_receipt, send_command, send_reactive_goal, update_reactive_goal
from emet.core.server import BaseZmqServer
from emet.rl.model import build_model
from emet.robots.innate_mars import InnateMarsBackend
from emet.simulation.robosuite_server import RobosuiteZmqServer

pytestmark = pytest.mark.sim


@pytest.fixture
def server(monkeypatch):
    # Exercise the actual sim server with in-process transport, without binding
    # robot ports or starting cameras/audio during a physics regression test.
    def initialize(self, *args, **kwargs):
        self.initialize_commands()
        self._last_step = -1
        self.verbose = False
        self._done = False

    monkeypatch.setattr(BaseZmqServer, "__init__", initialize)
    result = RobosuiteZmqServer(InnateMarsBackend().get_spec(), scene_model=build_model("table"), controller="baseline")
    result._load_model()
    result._sync_actuator_ctrl_from_joint_positions()
    result._init_reactive_controller()
    yield result
    result._running = False


def connection(server):
    client = SimpleNamespace(_act_lock=threading.Lock(), _iter=0, _state=server.command_message({}))

    def send(action):
        server.dispatch_command(action)
        client._state = server.command_message({})

    client.send_message = send
    return client


def test_table_sim_command_to_measured_completion_without_legacy_writes(server, monkeypatch):
    client = connection(server)
    action = send_reactive_goal(client, base=[0.12, 0, 0], arm={"joint1": 0.15})
    update_reactive_goal(client, action, {"base": [0.18, 0, 0], "arm": {"joint1": 0.2}})
    # Legacy state pinning would defeat physics and must never run during control.
    monkeypatch.setattr(server, "_hold_stationary_base_if_idle", lambda: pytest.fail("legacy writer ran"))
    for _ in range(2000):
        server._step_controlled_physics_once()
        if server._reactive_controller.outcome is not None:
            break
    assert server._reactive_controller.outcome[0] == "succeeded", server._reactive_controller.outcome
    server.poll_navigation_command()
    client._state = server.command_message({})
    receipt = command_receipt(client, action)
    assert receipt["status"] == "succeeded"
    assert receipt["result"]["stop_confirmed"]
    assert server.get_base_xyt()[0] == pytest.approx(0.18, abs=0.05)
    assert server._mjdata.qpos[server._mjmodel.joint("joint1").qposadr][0] == pytest.approx(0.2, abs=0.05)


def test_stale_physics_fails_and_cancellation_requires_measured_stop(server):
    client = connection(server)
    action = send_reactive_goal(client, base=[1, 0, 0])
    server._reactive_controller.last_step_wall = time.monotonic() - 1
    server.poll_navigation_command()
    client._state = server.command_message({})
    receipt = command_receipt(client, action)
    assert receipt["status"] == "failed"
    assert receipt["result"]["failure"] == "stale physics state"
    assert receipt["result"]["stop_confirmed"]
    assert not np.any(server._mjdata.ctrl[server._reactive_controller.control.base_act])


def test_policy_can_run_past_one_control_interval_but_misses_overall_budget(server):
    def slightly_slow(obs):
        time.sleep(0.06)
        return np.ones(7)

    server._reactive_controller.policy = slightly_slow
    client = connection(server)
    action = send_reactive_goal(client, base=[1, 0, 0])
    server._step_controlled_physics_once()
    server.poll_navigation_command()
    client._state = server.command_message({})
    assert server._reactive_controller.outcome is None
    assert np.any(server._mjdata.ctrl[server._reactive_controller.control.base_act])

    def too_slow(obs):
        time.sleep(0.12)
        return np.ones(7)

    server._reactive_controller.policy = too_slow
    server._reactive_controller.next_tick = server._mjdata.time
    server._step_controlled_physics_once()
    server.poll_navigation_command()
    client._state = server.command_message({})
    receipt = command_receipt(client, action)
    assert receipt["status"] == "failed"
    assert "deadline" in receipt["result"]["failure"]


def test_legacy_goal_execution_remains_available(server):
    client = connection(server)
    action = send_command(client, {"xyt": [0.12, 0, 0], "nav_world": True})
    for _ in range(2000):
        server._step_controlled_physics_once()
        server.poll_navigation_command()
        if server._navigation_command is None:
            break
    client._state = server.command_message({})
    assert command_receipt(client, action)["status"] == "succeeded"
    assert not server._reactive_controller.active


def test_base_only_server_completes_and_holds_arm(server):
    server._reactive_controller_kind = "base"
    server._init_reactive_controller()
    c = server._reactive_controller.control
    held = c.arm_positions().copy()
    client = connection(server)
    action = send_reactive_goal(client, base=[0.12, 0, 0])
    for _ in range(2500):
        server._step_controlled_physics_once()
        if server._reactive_controller.outcome is not None:
            break
    assert server._reactive_controller.outcome[0] == "succeeded", server._reactive_controller.outcome
    server.poll_navigation_command()
    client._state = server.command_message({})
    assert command_receipt(client, action)["status"] == "succeeded"
    assert server.get_base_xyt()[0] == pytest.approx(0.12, abs=0.02)
    np.testing.assert_allclose(c.arm_positions(), held, atol=0.02)
