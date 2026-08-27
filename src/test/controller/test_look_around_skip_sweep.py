# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""look_around skips head pans on non-Stretch robots (rby1 OVMM gate)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from emet.controller.controller_dynamem import (
    DynamemController,
    default_table_mapping_relative_yaws,
)
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


def _rotate_agent(robot) -> DynamemController:
    agent = _look_around_on(robot)
    agent.save_rerun = False
    agent._realtime_updates = False
    agent.rerun_iter = 0
    agent.log = ""
    agent._maybe_emit_navgrid_ascii = MagicMock()
    agent._find_phase_nav_timeout = lambda: 10.0
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
    monkeypatch.setattr("emet.controller.dynamem.look.time.sleep", lambda *_a, **_k: None)
    robot = MagicMock(spec=StretchZmqClient)
    agent = _look_around_on(robot)
    agent.look_around()
    assert agent._head_to_sweep.call_count >= 2
    assert agent.update.call_count >= 2


def test_look_around_force_sweep_overrides_rby1(monkeypatch):
    monkeypatch.setenv("EMET_FORCE_HEAD_SWEEP", "1")
    monkeypatch.setenv("EMET_SIM_NAV_TELEPORT", "1")
    monkeypatch.setattr("emet.controller.dynamem.look.time.sleep", lambda *_a, **_k: None)
    robot = MagicMock(spec=GenericZmqClient)
    agent = _look_around_on(robot)
    agent.look_around()
    assert agent._head_to_sweep.call_count >= 2


def test_default_table_mapping_relative_yaws_four_view_scan():
    yaws = default_table_mapping_relative_yaws(3)
    np.testing.assert_allclose(np.rad2deg(yaws), [25.0, -50.0, 25.0], atol=1e-6)


def _relative_yaws(robot: MagicMock) -> list[float]:
    out: list[float] = []
    for call in robot.move_base_to.call_args_list:
        args, kwargs = call
        if not kwargs.get("relative"):
            continue
        goal = np.asarray(args[0], dtype=float).reshape(-1)
        out.append(float(goal[2]))
    return out


def test_rotate_in_place_captures_heading_before_table_yaw(monkeypatch):
    monkeypatch.setattr("emet.controller.dynamem.look.time.sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "emet.eval.ovmm_find_phase._prepare_default_table_rby1_mapping_view",
        lambda _agent: True,
    )
    robot = MagicMock(spec=GenericZmqClient)
    agent = _rotate_agent(robot)
    agent.rotate_in_place(n_steps=4)
    assert agent.update.call_count == 4
    np.testing.assert_allclose(np.rad2deg(_relative_yaws(robot)), [25.0, -50.0, 25.0], atol=1e-6)
    robot.look_front.assert_not_called()


def test_rotate_in_place_non_table_still_uses_45deg_after_capture(monkeypatch):
    monkeypatch.setattr("emet.controller.dynamem.look.time.sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "emet.eval.ovmm_find_phase._prepare_default_table_rby1_mapping_view",
        lambda _agent: False,
    )
    robot = MagicMock(spec=GenericZmqClient)
    agent = _rotate_agent(robot)
    agent.rotate_in_place(n_steps=4)
    assert agent.update.call_count == 4
    np.testing.assert_allclose(_relative_yaws(robot), [np.pi / 4.0] * 3, atol=1e-6)
    robot.look_front.assert_called_once()


def test_rotate_in_place_updates_even_when_realtime(monkeypatch):
    monkeypatch.setattr("emet.controller.dynamem.look.time.sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "emet.eval.ovmm_find_phase._prepare_default_table_rby1_mapping_view",
        lambda _agent: True,
    )
    robot = MagicMock(spec=GenericZmqClient)
    agent = _rotate_agent(robot)
    agent._realtime_updates = True
    agent.rotate_in_place(n_steps=4)
    assert agent.update.call_count == 4
