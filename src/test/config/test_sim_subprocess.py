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

        def terminate(self) -> None:
            calls.append("terminate")

        def wait(self, timeout: float | None = None) -> int:
            calls.append("wait")
            return 0

        def kill(self) -> None:
            calls.append("kill")

    fake = FakeProc()

    def fake_popen(*_a: object, **_kw: object) -> FakeProc:
        calls.append("popen")
        return fake

    monkeypatch.setattr(ssub.subprocess, "Popen", fake_popen)

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
