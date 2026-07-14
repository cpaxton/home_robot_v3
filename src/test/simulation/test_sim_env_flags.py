# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

from __future__ import annotations

import importlib

import pytest


@pytest.fixture(autouse=True)
def _reset_sim_nav_warn_state(monkeypatch):
    import emet.simulation.env_flags as ef

    monkeypatch.setattr(ef, "_warned_sim_nav_env", False)
    monkeypatch.delenv("EMET_SIM_NAV_DEBUG", raising=False)
    monkeypatch.delenv("EMET_SIM_NAV_TELEPORT", raising=False)
    yield
    importlib.reload(ef)


def test_env_sim_nav_debug_truthy(monkeypatch):
    import emet.simulation.env_flags as ef

    monkeypatch.setenv("EMET_SIM_NAV_DEBUG", "1")
    assert ef.env_sim_nav_debug() is True
    monkeypatch.setenv("EMET_SIM_NAV_DEBUG", "off")
    assert ef.env_sim_nav_debug() is False


def test_warn_sim_nav_env_flags_once(monkeypatch, capsys):
    import emet.simulation.env_flags as ef

    monkeypatch.setenv("EMET_SIM_NAV_DEBUG", "yes")
    ef.warn_sim_nav_env_flags()
    ef.warn_sim_nav_env_flags()
    err = capsys.readouterr().err
    assert err.count("EMET_SIM_NAV_DEBUG") == 1


def test_pure_yaw_relative_skips_teleport_on_planar_robots_without_flag():
    import numpy as np

    from emet.simulation.robosuite_server import RobosuiteZmqServer

    raw = np.array([0.0, 0.0, np.pi / 4.0])
    action = {"nav_relative": True}
    assert RobosuiteZmqServer._is_pure_yaw_relative(action, raw)
    srv = object.__new__(RobosuiteZmqServer)
    srv._planar_base_joint_names = lambda: ("base_x", "base_y", "base_yaw")
    srv._is_molmospaces_session = lambda: False
    assert not srv._resolve_nav_teleport(action, raw)


def test_explicit_nav_teleport_wins_over_planar_pure_yaw():
    import numpy as np

    from emet.simulation.robosuite_server import RobosuiteZmqServer

    raw = np.array([0.0, 0.0, np.pi / 4.0])
    action = {"nav_relative": True, "nav_teleport": True}
    srv = object.__new__(RobosuiteZmqServer)
    srv._planar_base_joint_names = lambda: ("base_x", "base_y", "base_yaw")
    srv._is_molmospaces_session = lambda: False
    assert srv._resolve_nav_teleport(action, raw)
    assert srv._resolve_nav_teleport(
        {"nav_relative": True, "nav_teleport": True},
        np.array([1.0, 0.0, 0.0]),
    )
