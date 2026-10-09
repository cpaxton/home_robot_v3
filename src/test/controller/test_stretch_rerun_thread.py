"""Disabled visualization must not spawn a tight loop that starves motion planning."""
from threading import Lock
from types import SimpleNamespace

import pytest

from emet.controller.zmq_client import StretchZmqClient
from emet.visualization.null_visualizer import NullVisualizer


@pytest.mark.parametrize('enabled', [False, True])
def test_start_creates_viewer_thread_only_when_enabled(monkeypatch, enabled):
    started = []
    monkeypatch.setattr('emet.controller.zmq_client.threading.Thread',
                        lambda target, daemon: SimpleNamespace(start=lambda: started.append(target)))
    recv, state, servo, viewer = (object() for _ in range(4))
    client = SimpleNamespace(
        _started=False, _zmq_closed=False, _rerun=SimpleNamespace(enabled=True) if enabled else NullVisualizer(),
        blocking_spin=recv, blocking_spin_state=state, blocking_spin_servo=servo, blocking_spin_rerun=viewer,
        _obs={}, _state={}, _servo={}, _obs_lock=Lock(), _state_lock=Lock(),
        _verify_emet_robot_id_stretch=lambda: True, _note_emet_session_from_zmq_dict=lambda _: None,
        is_homed=True, is_runstopped=False,
    )
    assert StretchZmqClient.start(client)
    assert started == [recv, state, servo] + ([viewer] if enabled else [])


@pytest.mark.parametrize('enabled', [False, True])
def test_visualization_loop_yields_with_existing_observations(monkeypatch, enabled):
    sleeps = []
    calls = iter([True, False])
    client = SimpleNamespace(
        _finish=False, _wait_if_streams_paused=lambda: next(calls),
        _rerun=SimpleNamespace(enabled=True, step=lambda *a, **k: None) if enabled else NullVisualizer(),
        _obs=object(), _servo=object(), _rerun_debug=False, peek_mapping_depth_for_rerun=lambda: None,
    )
    monkeypatch.setattr('emet.controller.zmq_client.time.monotonic', lambda: 0.0)
    monkeypatch.setattr('emet.controller.zmq_client.time.sleep', sleeps.append)
    StretchZmqClient.blocking_spin_rerun(client)
    assert sleeps == [pytest.approx(1 / 30)]
