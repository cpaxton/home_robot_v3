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
