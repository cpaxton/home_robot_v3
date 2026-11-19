# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Release refinements share retained plan budgets and preserve failure state."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from emet.controller.manipulation.physical_pick_place import PhysicalMotionResult
from emet.controller.task.tamp import native_adapter, native_placement


@pytest.mark.parametrize("limit", ["ik", "time", "geometry", "success"])
def test_release_refinement_charges_retained_budget_and_restores_settings(monkeypatch, limit):
    clock = [10.0]
    timer = SimpleNamespace(monotonic=lambda: clock[0])
    monkeypatch.setattr(native_adapter, "time", timer)
    monkeypatch.setattr(native_placement, "time", timer)
    adapter = native_adapter.NativeTableAdapter.__new__(native_adapter.NativeTableAdapter)
    adapter.max_planning_ik_calls, adapter._planning_deadline = 99, 999.0
    adapter.planning_ik_calls = 88
    adapter.release_limits = object()
    collision = object()
    old_planner = Mock()
    ex = SimpleNamespace(
        collision=collision,
        release_pose_planner=old_planner,
        release_lateral_shift=np.array([-0.02, 0.0]),
        command_joints=Mock(),
    )
    goal = np.array([0.2, -0.6, 0.54])
    plan = SimpleNamespace(
        executor=ex,
        planning_ik_calls=3,
        planning_ik_limit=4 if limit == "ik" else 20,
        planning_seconds_remaining=2.0,
        goal=goal.copy(),
        support_body="table",
        placement_region=None,
    )

    def solve(*args, **kwargs):
        clock[0] += 1.25 if limit == "time" else 0.1
        return [np.zeros(1)], None

    probe = SimpleNamespace(plan_pose=Mock(side_effect=solve))

    def refine(executor, **kwargs):
        with executor.release_planning_budget():
            for _ in range(1 if limit == "geometry" else 2):
                executor.release_pose_planner(probe, np.zeros(3), np.eye(3))
            if limit == "geometry":
                clock[0] += 3.0  # Dense opening/fall checks after cheap IK.
        return PhysicalMotionResult(True, "ok", "release")

    monkeypatch.setattr(native_placement, "refine_release", refine)
    result = native_placement.refine_release_placement(adapter, plan)
    assert result.success is (limit == "success")
    if limit != "success":
        assert result.message == "no_plan_within_budget" and result.phase == "release"
        np.testing.assert_array_equal(plan.goal, goal)
    else:
        np.testing.assert_allclose(plan.goal, goal + [-0.02, 0, 0])
    assert probe.plan_pose.call_count == (1 if limit in {"ik", "geometry"} else 2)
    assert adapter.planning_ik_calls == plan.planning_ik_calls == 3 + probe.plan_pose.call_count
    assert adapter.max_planning_ik_calls == 99 and adapter._planning_deadline == 999.0
    assert plan.planning_seconds_remaining == pytest.approx(
        {"ik": 1.9, "time": 0.0, "geometry": 0.0, "success": 1.8}[limit]
    )
    assert ex.release_pose_planner is old_planner and ex.collision is collision
    assert ex.release_planning_budget is None
    ex.command_joints.assert_not_called()
