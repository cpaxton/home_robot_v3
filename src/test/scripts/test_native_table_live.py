# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Live scoring reads measured snapshots and never steps the simulator itself."""
from types import SimpleNamespace

import mujoco
import pytest

from emet.eval import native_table_live as live


def test_live_wait_requires_new_physics_and_rejects_restarts(monkeypatch):
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint name="j"/><geom size=".1"/></body></worldbody></mujoco>')
    state = {'qpos': [.1], 'qvel': [0.], 'sim_time': 0.}
    boot = ['boot-1']
    monkeypatch.setattr(live, 'read_snapshot', lambda *a: (boot[0], state.copy()))
    monkeypatch.setattr(mujoco, 'mj_step', lambda *a: pytest.fail('must not step live physics'))
    robot = live.LiveTableRobot(SimpleNamespace(marker=42), model)
    assert robot.marker == 42
    assert robot.data.qpos[0] == .1
    def advance(_):
        state['sim_time'] += .02
    monkeypatch.setattr(live.time, 'sleep', advance)
    robot.step(.05)
    assert robot.data.time >= .05
    assert state['qpos'] == [.1]
    boot[0] = 'boot-2'
    with pytest.raises(RuntimeError, match='restarted'):
        robot.refresh()


def test_live_wait_has_wall_watchdog(monkeypatch):
    robot = object.__new__(live.LiveTableRobot)
    robot.data = SimpleNamespace(time=0.)
    robot.refresh = lambda: None
    ticks = iter([0., 31.])
    monkeypatch.setattr(live.time, 'monotonic', lambda: next(ticks))
    with pytest.raises(TimeoutError, match='stopped advancing'):
        robot.step(.05)


@pytest.mark.parametrize('opt_in,simulation,cameras,expected', [
    (True, True, 0, True), (False, True, 0, False),
    (True, False, 0, False), (True, True, 2, False), (True, True, None, False),
])
def test_state_only_start_requires_explicit_camera_free_simulation(opt_in, simulation, cameras, expected):
    from emet.controller.zmq_client import StretchZmqClient
    from emet.core.zmq_protocol import EMET_ZMQ_SESSION_KEY
    client = object.__new__(StretchZmqClient)
    client._state_only = opt_in
    client._state = {EMET_ZMQ_SESSION_KEY: {'is_simulation': simulation, 'capabilities': {'num_cameras': cameras}}}
    assert client._state_only_startup_ready() is expected


def test_read_only_observer_never_claims_command_ownership(monkeypatch, tmp_path):
    from unittest.mock import Mock

    from emet.config import sim_launch_config
    from emet.controller import zmq_client
    from emet.eval import sim_eval_session

    client = Mock()
    client.start.return_value = True
    monkeypatch.setattr(zmq_client, 'StretchZmqClient', lambda **kw: client)
    monkeypatch.setattr(sim_launch_config, 'load_sim_launch_config_from_path',
                        lambda path: SimpleNamespace(no_cameras=True))
    server = SimpleNamespace(port_offset=360)
    monkeypatch.setattr(sim_eval_session, 'launch_benchmark_sim_server', lambda *a, **kw: server)
    cleanup = Mock()
    monkeypatch.setattr(sim_eval_session, 'terminate_benchmark_sim_server', cleanup)
    monkeypatch.setattr(live, 'LiveTableRobot', lambda client, model: client)
    with live.live_table_robot(tmp_path / 'sim.yaml', tmp_path, None, initialize=False) as observer:
        assert observer is client
        client.move_to_nav_posture.assert_not_called()
        client.set_velocity.assert_not_called()
    client.stop.assert_called_once()
    cleanup.assert_called_once_with(server)
