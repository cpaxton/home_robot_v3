# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Collision-aware place-approach selection (graded conservatism)."""

from __future__ import annotations

import numpy as np

from emet.controller.manipulation.kinematic_pick_place import KinematicPickPlaceExecutor


class _Robot:
    def __init__(self, clear_flags):
        self._spec = type("Spec", (), {"tamp_approach": "front"})()
        self.clear_flags = clear_flags
        self.checked: list[list] = []
        self.moved: list[np.ndarray] = []

    def check_base_poses(self, poses):
        self.checked.append(list(poses))
        flags = [self.clear_flags[i] if i < len(self.clear_flags) else False for i in range(len(poses))]
        return {"poses": poses, "clear": flags}

    def move_base_to(self, xyt, **kwargs):
        self.moved.append(np.asarray(xyt, dtype=np.float64))
        return True


def _executor(clear_flags):
    ex = object.__new__(KinematicPickPlaceExecutor)
    ex.arm = "left"
    ex._sleep = lambda *a, **k: None
    ex.robot = _Robot(clear_flags)
    return ex, ex.robot


def _ring(standoff):
    from emet.controller.task.tamp.task_search import approach_candidates_for_object_xy

    return approach_candidates_for_object_xy(np.array([0.0, 0.0]), standoff=standoff, mode="front", arm="left", count=16)


def test_base_pose_clearance_filters_blocked_candidates():
    ex, robot = _executor([True, False, True])
    cands = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([-1.0, 0.0, 0.0])]
    assert ex._base_pose_clearance(cands) == [True, False, True]
    assert len(robot.checked[0]) == 3


def test_approach_xy_picks_first_collision_free_candidate():
    ex, robot = _executor([False, False, True])
    assert ex._approach_xy(np.array([0.0, 0.0])) is True
    assert len(robot.moved) == 1
    np.testing.assert_allclose(robot.moved[0][:2], _ring(0.55)[2][:2], atol=1e-9)


def test_approach_xy_no_collision_free_candidate_fails_without_motion():
    ex, robot = _executor([False] * 20)
    assert ex._approach_xy(np.array([0.0, 0.0])) is False
    assert robot.moved == []


def test_approach_xy_bounds_aware_standoff():
    ex, robot = _executor([True] * 20)
    # A deep receptacle (planar radius 0.5 m) pushes the standoff to 0.5 + 0.45 = 0.95 m.
    assert ex._approach_xy(np.array([0.0, 0.0]), recep_radius_m=0.5) is True
    assert len(robot.moved) == 1
    np.testing.assert_allclose(robot.moved[0][:2], _ring(0.95)[0][:2], atol=1e-9)


def test_approach_xy_falls_back_when_check_base_poses_missing():
    ex = object.__new__(KinematicPickPlaceExecutor)
    ex.arm = "left"
    ex._sleep = lambda *a, **k: None

    class NoCheck:
        def __init__(self):
            self._spec = type("Spec", (), {"tamp_approach": "front"})()
            self.moved = []

        def move_base_to(self, xyt, **kwargs):
            self.moved.append(xyt)
            return True

    ex.robot = NoCheck()
    ex._world_base_xyt = lambda: np.array([0.0, -1.0, 0.0])
    assert ex._approach_xy(np.array([0.0, 0.0])) is True
    assert len(ex.robot.moved) == 1
