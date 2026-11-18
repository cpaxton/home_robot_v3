# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""The clearance hint must query the same world frame as navigation."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from emet.agent.tools import format_base_clearance_hint
from emet.utils.geometry import nav_xyt_to_world_xyt


@pytest.mark.parametrize("origin", [[1.4, 1.6, 0.0], [10.0, -3.0, 1.2]])
def test_clearance_hint_converts_local_robot_pose_to_planning_world(origin):
    local = np.array([0.2, 0.1, -0.3])
    session = {"navigation_origin_xyt": origin}
    world = nav_xyt_to_world_xyt(local, session)

    def clearance(xy):
        np.testing.assert_allclose(xy, world[:2])
        return 0.45

    planner = SimpleNamespace(_clearance_m=object(), clearance_at_xy=Mock(side_effect=clearance))
    agent = SimpleNamespace(
        robot=SimpleNamespace(get_base_pose=lambda: local.copy()),
        planner=planner,
        _planning_base_xyt=lambda pose: nav_xyt_to_world_xyt(pose, session),
        _min_clearance_m=0.33,
    )
    assert format_base_clearance_hint(agent) == "Base clearance ≈ 0.45m (min required 0.33m)."
    planner.clearance_at_xy.assert_called_once()


def test_unavailable_frame_conversion_does_not_emit_a_fabricated_clearance():
    agent = SimpleNamespace(
        robot=SimpleNamespace(get_base_pose=lambda: [0, 0, 0]),
        planner=SimpleNamespace(clearance_at_xy=Mock()),
        _planning_base_xyt=Mock(side_effect=RuntimeError("origin unavailable")),
    )
    assert format_base_clearance_hint(agent) == ""
    agent.planner.clearance_at_xy.assert_not_called()
