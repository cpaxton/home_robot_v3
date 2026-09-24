# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from emet.controller.operations import payload_verification as payload


def carrying_agent():
    robot = Mock()
    robot.get_pan_tilt.return_value = (0.2, -0.5)
    robot.head_to.return_value = robot.look_at_ee.return_value = True
    return SimpleNamespace(
        robot=robot,
        _carried_object=payload.CarriedObject(
            "can",
            np.array([1.0, 2.0, 0.9]),
            np.array([0.0, 0.0, 0.02]),
            0.05,
        ),
    )


def test_no_payload_means_no_robot_calls():
    agent = SimpleNamespace(robot=Mock())
    payload.verify_carried_object(agent)
    assert not agent.robot.mock_calls


@pytest.mark.parametrize("reason", [None, "target absent", "object moved relative to gripper"])
def test_payload_check_restores_head_and_latches_uncertainty(monkeypatch, reason):
    agent = carrying_agent()
    observe = Mock(side_effect=ValueError(reason) if reason else None)
    monkeypatch.setattr(payload, "observe_lifted_query", observe)
    if reason:
        with pytest.raises(payload.PayloadVerificationError, match=reason):
            payload.verify_carried_object(agent)
        with pytest.raises(payload.PayloadVerificationError, match=reason):
            payload.verify_carried_object(agent)
    else:
        payload.verify_carried_object(agent)
    observe.assert_called_once()
    agent.robot.head_to.assert_called_once_with(0.2, -0.5, blocking=True)
    agent.robot.move_base_to.assert_not_called()
    agent.robot.open_gripper.assert_not_called()


def test_failed_head_restore_prevents_navigation(monkeypatch):
    agent = carrying_agent()
    agent.robot.head_to.return_value = False
    monkeypatch.setattr(payload, "observe_lifted_query", Mock())
    with pytest.raises(payload.PayloadVerificationError, match="restoration"):
        payload.verify_carried_object(agent)


def test_payload_failure_after_trajectory_prevents_arrival_and_next_hop(monkeypatch):
    from emet.controller.dynamem.navigation import execute_action

    check = Mock(side_effect=[None, payload.PayloadVerificationError("payload unverified")])
    monkeypatch.setattr(payload, "verify_carried_object", check)
    monkeypatch.setattr("emet.controller.dynamem.navigation.time.sleep", lambda _: None)
    monkeypatch.setattr("emet.controller.nav_confirm.confirm_navigation_plan", lambda *a, **kw: True)
    agent = SimpleNamespace(
        _realtime_updates=True,
        robot=Mock(),
        query_driven_memory=True,
        _current_planning_xyt=Mock(return_value=np.zeros(3)),
        process_text=Mock(return_value=[np.zeros(3), np.ones(3)]),
        _last_nav_plan={},
        _record_nav_plan_fields=Mock(),
        announce_action=Mock(),
        _find_phase_nav_timeout=Mock(return_value=1),
        _navigation_origin_xyt=Mock(return_value=np.zeros(3)),
        pos_err_threshold=0.1,
        rot_err_threshold=0.1,
        update=Mock(),
        verify_query_arrival=Mock(),
    )
    with pytest.raises(payload.PayloadVerificationError):
        execute_action(agent, "countertop")
    agent.robot.execute_trajectory.assert_called_once()
    agent.process_text.assert_called_once()
    agent.verify_query_arrival.assert_not_called()
    agent.update.assert_not_called()
