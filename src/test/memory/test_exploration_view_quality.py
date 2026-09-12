# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from emet.memory.graph_eqa.agentic.view_quality import exploration_view_quality, recover_exploration_view
from emet.memory.graph_eqa.agentic.views import CapturedView


def test_close_geometry_not_color_or_distant_wall_triggers_recovery():
    assert exploration_view_quality(np.full((20, 20), 0.2))["obstructed"]
    assert not exploration_view_quality(np.full((20, 20), 2.0))["obstructed"]
    assert not exploration_view_quality(None)["obstructed"]


def test_head_recovery_is_bounded_and_requires_current_capture():
    robot = SimpleNamespace(head_to=Mock(), wait_for_obs=Mock())
    vm = SimpleNamespace(observations=[])
    ex = SimpleNamespace(agent=SimpleNamespace(robot=robot, voxel_map=vm), _captured_views={}, _append_trace=Mock())

    def capture():
        vm.observations.append(SimpleNamespace(depth=np.full((20, 20), 0.2)))
        oid = len(vm.observations)
        ex._captured_views[oid] = CapturedView(oid, oid, np.zeros((20, 20, 3), dtype=np.uint8), None)
        return {"ok": True, "obs_id": oid}

    ex._tool_capture_and_update = capture
    result = recover_exploration_view(ex, capture())
    assert not result["ok"] and result["status"] == "OBSTRUCTED_VIEW"
    assert robot.head_to.call_count == 2
    assert len(vm.observations) == 3


def test_open_view_does_not_move_head():
    robot = SimpleNamespace(head_to=Mock())
    ex = SimpleNamespace(
        agent=SimpleNamespace(
            robot=robot, voxel_map=SimpleNamespace(observations=[SimpleNamespace(depth=np.ones((20, 20)))])
        ),
        _captured_views={1: CapturedView(1, 1, np.zeros((20, 20, 3), dtype=np.uint8), None)},
        _append_trace=Mock(),
    )
    cap = {"ok": True, "obs_id": 1}
    assert recover_exploration_view(ex, cap) == cap
    robot.head_to.assert_not_called()
