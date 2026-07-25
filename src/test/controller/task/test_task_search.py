# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for deterministic TAMP task search (no sim)."""

from __future__ import annotations

import numpy as np

from emet.controller.task.tamp.task_search import (
    TaskPlan,
    TaskPlanStep,
    approach_pose_for_object_xy,
    plan_pick_place,
)


class _FakeRobot:
    def __init__(self, placements: dict):
        self._pl = placements
        self.moved: list[np.ndarray] = []

    def get_emet_session(self):
        return {
            "is_simulation": True,
            "emet_robot_id": "rby1",
            "sim_object_placements": self._pl,
            "capabilities": {"kinematic_manip": True, "sim_set_body_pose": True},
        }

    def move_base_to(self, xyt, blocking=True, world_frame=True):
        self.moved.append(np.asarray(xyt, dtype=np.float64).copy())


class _FakeGrasp:
    def __init__(self, xyz):
        T = np.eye(4)
        T[:3, 3] = np.asarray(xyz, dtype=np.float64)
        self.T_world = T
        self.position = T[:3, 3].copy()


def test_approach_pose_faces_minus_y():
    p = approach_pose_for_object_xy([1.0, 2.0], standoff=0.4)
    assert abs(p[0] - 1.0) < 1e-9
    assert abs(p[1] - 2.4) < 1e-9
    assert abs(p[2] + np.pi / 2) < 1e-9


def test_plan_pick_place_without_executor():
    pl = {
        "obj_a": {"cat": "red cylinder", "pos": [0.1, -0.5, 0.8]},
        "cube_b": {"cat": "blue cube", "pos": [0.2, -0.4, 0.75]},
    }
    robot = _FakeRobot(pl)
    grasps = [_FakeGrasp([0.1, -0.5, 0.85]), _FakeGrasp([0.12, -0.5, 0.85])]
    plan = plan_pick_place(
        robot,
        object_query="red cylinder",
        receptacle_query="blue cube",
        grasp_poses=grasps,
        object_gt_body="obj_a",
        receptacle_gt_body="cube_b",
        executor=None,
    )
    assert plan.success
    assert plan.chosen_grasp_index == 0
    assert [s.op for s in plan.steps] == ["approach", "grasp", "place"]
    assert any("approach@" in n for n in plan.expanded_nodes)


def test_plan_missing_object():
    robot = _FakeRobot({})
    plan = plan_pick_place(
        robot,
        object_query="x",
        receptacle_query="y",
        grasp_poses=[],
        object_gt_body="missing",
        executor=None,
    )
    assert not plan.success
    assert plan.message == "object_not_in_gt"


def test_task_plan_dataclass():
    p = TaskPlan(steps=[TaskPlanStep("approach", {"xyt": [0, 0, 0]})], object_body="a", receptacle_body=None)
    assert p.steps[0].op == "approach"
