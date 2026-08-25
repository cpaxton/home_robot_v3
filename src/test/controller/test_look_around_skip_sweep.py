# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""look_around skips head pans on non-Stretch robots (rby1 OVMM gate)."""

from __future__ import annotations

from unittest.mock import MagicMock

from emet.controller.controller_dynamem import DynamemController
from emet.controller.generic_zmq_client import GenericZmqClient
from emet.controller.zmq_client import StretchZmqClient


def _look_around_on(robot) -> DynamemController:
    # Avoid DynamemController.__init__ (heavy); only need look_around behavior.
    agent = DynamemController.__new__(DynamemController)
    agent.robot = robot
    agent._fast_explore_lookaround = True
    agent.announce_action = MagicMock()
    agent.announce_motion_progress = MagicMock()
    agent.update = MagicMock()
    agent._head_to_sweep = MagicMock()
    return agent


def test_look_around_skips_head_sweep_for_rby1(monkeypatch):
    monkeypatch.delenv("EMET_FORCE_HEAD_SWEEP", raising=False)
    monkeypatch.delenv("EMET_SKIP_HEAD_SWEEP", raising=False)
    robot = MagicMock(spec=GenericZmqClient)
    agent = _look_around_on(robot)
    agent.look_around()
    agent.update.assert_called_once()
    agent._head_to_sweep.assert_not_called()


def test_look_around_sweeps_for_stretch(monkeypatch):
    monkeypatch.delenv("EMET_FORCE_HEAD_SWEEP", raising=False)
    monkeypatch.delenv("EMET_SKIP_HEAD_SWEEP", raising=False)
    monkeypatch.setenv("EMET_SIM_NAV_TELEPORT", "1")
    monkeypatch.setattr("emet.controller.controller_dynamem.time.sleep", lambda *_a, **_k: None)
    robot = MagicMock(spec=StretchZmqClient)
    agent = _look_around_on(robot)
    agent.look_around()
    assert agent._head_to_sweep.call_count >= 2
    assert agent.update.call_count >= 2


def test_look_around_force_sweep_overrides_rby1(monkeypatch):
    monkeypatch.setenv("EMET_FORCE_HEAD_SWEEP", "1")
    monkeypatch.setenv("EMET_SIM_NAV_TELEPORT", "1")
    monkeypatch.setattr("emet.controller.controller_dynamem.time.sleep", lambda *_a, **_k: None)
    robot = MagicMock(spec=GenericZmqClient)
    agent = _look_around_on(robot)
    agent.look_around()
    assert agent._head_to_sweep.call_count >= 2
