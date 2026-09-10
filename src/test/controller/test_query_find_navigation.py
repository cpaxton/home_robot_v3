# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Query find must verify arrival, never return the retrieval anchor as success."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from emet.controller.dynamem.navigation import execute_action


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
