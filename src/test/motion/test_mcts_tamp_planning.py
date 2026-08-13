# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""MCTS pick-place planning tests (no server, no LLM)."""

from __future__ import annotations

import numpy as np

from emet.controller.task.tamp.task_search import plan_pick_place_mcts, policy_rollout
from emet.motion.agent_mcts import MCTSAction


def _cand(obj: str, recep: str) -> dict:
    return {
        "object_query": obj,
        "receptacle_query": recep,
        "object_gt_body": f"{obj}_gt",
        "receptacle_gt_body": f"{recep}_gt",
    }


class _FakeRobot:
    """Robot whose session placements define scene geometry for the MCTS planner."""

    def __init__(self, placements: dict):
        self._placements = placements

    def get_emet_session(self):
        return {"sim_object_placements": self._placements}


def _placement(pos: np.ndarray, cat: str) -> dict:
    return {"pos": pos.reshape(3).tolist(), "cat": cat}


def test_policy_rollout_move_then_pickup_then_place():
    state = {
        "robot": np.array([0.0, 0.0]),
        "object": np.array([2.0, 0.0]),
        "carrying": False,
        "receptacle": np.array([5.0, 0.0]),
    }
    # move to object
    s1, r1, done1 = policy_rollout(state, MCTSAction("move_to", {"xy": [2.0, 0.0]}, cost=1.0))
    assert not done1 and r1 > 0.0
    assert np.allclose(s1["robot"], [2.0, 0.0])
    # pickup
    s2, r2, done2 = policy_rollout(s1, MCTSAction("pickup", {"object": "t"}, cost=0.1))
    assert s2["carrying"] is True and not done2
    # move to receptacle
    s3, r3, done3 = policy_rollout(s2, MCTSAction("move_to", {"xy": [5.0, 0.0]}, cost=1.0))
    assert not done3 and np.allclose(s3["robot"], [5.0, 0.0])
    # place
    s4, r4, done4 = policy_rollout(s3, MCTSAction("place", {"receptacle": "t"}, cost=0.1))
    assert done4 and s4["carrying"] is False
    assert np.allclose(s4["object"], [5.0, 0.0])
    assert r4 > 0.0


def test_pickup_requires_proximity():
    state = {
        "robot": np.array([0.0, 0.0]),
        "object": np.array([2.0, 0.0]),
        "carrying": False,
        "receptacle": np.array([5.0, 0.0]),
    }
    s, r, done = policy_rollout(state, MCTSAction("pickup", {"object": "t"}, cost=0.1))
    assert done and r < 0.0  # far pickup fails


def _dummy_grasp_pose() -> np.ndarray:
    T = np.eye(4)
    T[2, 3] = 1.0
    return T


def test_plan_pick_place_mcts_finds_reachable_task():
    # Bowl near origin; microwave across the room. MCTS should pick bowl -> microwave.
    placements = {
        "bowl_gt": _placement(np.array([0.3, -0.5, 1.0]), "Bowl"),
        "microwave_gt": _placement(np.array([-0.2, -2.5, 0.9]), "Microwave"),
        "counter_gt": _placement(np.array([0.0, -0.5, 0.8]), "Counter"),
    }
    robot = _FakeRobot(placements)
    candidates = [
        _cand("bowl", "microwave"),
        _cand("bowl", "counter"),
    ]

    plan = plan_pick_place_mcts(
        robot,
        candidates=candidates,
        grasp_poses=[_dummy_grasp_pose()],
        executor=None,
        mcts_iterations=40,
        seed=1,
    )
    assert plan.object_body == "bowl_gt"
    assert plan.success
    names = [s.op for s in plan.steps]
    assert names == ["approach", "grasp", "place"]


def test_plan_pick_place_mcts_skips_missing_gt():
    robot = _FakeRobot({})  # no placements
    plan = plan_pick_place_mcts(
        robot,
        candidates=[_cand("bowl", "microwave")],
        grasp_poses=[],
        executor=None,
    )
    assert not plan.success
    assert plan.message == "no_gt_candidates"
