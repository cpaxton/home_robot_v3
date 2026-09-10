# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Query find must verify arrival, never return the retrieval anchor as success."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from emet.controller.dynamem.navigation import execute_action


def test_real_lazy_controller_scan_forwards_verification(monkeypatch):
    from emet.controller.controller_lazy_graph import LazyGraphController

    monkeypatch.setattr("emet.controller.controller_graph_eqa.is_habitat_robot_client", lambda _: False)
    monkeypatch.setenv("EMET_SKIP_HEAD_SWEEP", "1")
    monkeypatch.delenv("EMET_FORCE_HEAD_SWEEP", raising=False)
    agent = object.__new__(LazyGraphController)
    agent.robot = Mock()
    agent.parameters = {}
    agent.announce_action = Mock()
    agent.update = Mock()
    verifier = Mock(return_value=True)
    assert agent.look_around(on_observation=verifier) is True
    agent.update.assert_called_once_with(full_perception=True)
    verifier.assert_called_once()


def test_habitat_scan_preserves_verified_view():
    from emet.controller.habitat_nav import habitat_body_scan

    robot = SimpleNamespace(_sim=Mock(), _sync_pose_from_sim=Mock())
    update = Mock()
    verifier = Mock(side_effect=[False, True])
    assert habitat_body_scan(robot, on_step=update, on_observation=verifier) is True
    assert robot._sim.step.call_count == 2
    assert update.call_count == 2


@pytest.mark.parametrize("accepted", [True, False])
def test_query_arrival_returns_only_verified_geometry(monkeypatch, accepted):
    monkeypatch.setattr("emet.controller.dynamem.navigation.time.sleep", lambda _: None)
    monkeypatch.setattr("emet.controller.nav_confirm.confirm_navigation_plan", lambda *a, **kw: True)
    anchor = np.array([9, 9, 9])
    verified = np.array([1, 2, 3]) if accepted else None
    agent = SimpleNamespace(
        _realtime_updates=True,
        robot=Mock(),
        query_driven_memory=True,
        _current_planning_xyt=Mock(return_value=np.zeros(3)),
        process_text=Mock(return_value=[[np.nan] * 3, anchor]),
        _last_nav_plan={"mode": "navigation", "query_candidate_handle": 7},
        _record_nav_plan_fields=Mock(),
        announce_action=Mock(),
        _find_phase_nav_timeout=Mock(return_value=1),
        _navigation_origin_xyt=Mock(return_value=np.zeros(3)),
        verify_query_arrival=Mock(return_value=verified),
        query_candidates=Mock(),
        voxel_map=SimpleNamespace(observations=[0, 1]),
    )
    status, point = execute_action(agent, "cup")
    assert status is accepted
    assert point is verified
    agent.verify_query_arrival.assert_called_once_with("cup", candidate_handle=7)
    if accepted:
        agent.query_candidates.reject.assert_not_called()
    else:
        agent.query_candidates.reject.assert_called_once()
