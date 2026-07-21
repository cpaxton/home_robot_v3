# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

from __future__ import annotations

from types import SimpleNamespace

import pytest

import emet.simulation.sim_subprocess as ssub
from emet.config.sim_launch_config import SimLaunchDefaultMujoco


@pytest.fixture(autouse=True)
def _clean_sim_proc():
    ssub.shutdown_mujoco_server_subprocess()
    yield
    ssub.shutdown_mujoco_server_subprocess()


def test_spawn_terminates_subprocess_when_wait_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeProc:
        pid = 99999

        def poll(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            calls.append("wait")
            return 0

    fake = FakeProc()

    def fake_popen_session(*_a: object, **kw: object) -> FakeProc:
        calls.append("popen")
        assert kw.get("stdout") == ssub.subprocess.DEVNULL
        assert kw.get("stderr") == ssub.subprocess.DEVNULL
        return fake

    def fake_terminate(proc: object, *, grace_s: float = 15.0) -> None:
        calls.append("terminate")
        assert proc is fake
        assert grace_s == 8.0

    monkeypatch.setattr("emet.utils.process_tree.popen_session", fake_popen_session)
    monkeypatch.setattr("emet.utils.process_tree.terminate_process_tree", fake_terminate)

    def boom(*_a: object, **_kw: object) -> None:
        raise TimeoutError("no port")

    monkeypatch.setattr(ssub, "wait_for_sim_tcp_port", boom)
    monkeypatch.setattr(
        "emet.utils.port_utils.get_ports",
        lambda _offset: SimpleNamespace(send=4402),
    )

    cfg = SimLaunchDefaultMujoco(robot="stretch", headless=True)
    with pytest.raises(TimeoutError, match="no port"):
        ssub.spawn_mujoco_server_subprocess(cfg)
    assert "popen" in calls
    assert "terminate" in calls
    assert ssub._SIM_PROC is None
