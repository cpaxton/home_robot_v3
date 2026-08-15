# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Agent-call-wrapping MCTS tests (heuristic distance policy, deterministic sim).

No LLM, no MuJoCo: ``simulate`` is a tiny functional pick-place step so the
search can be asserted end-to-end (find a pick -> place sequence).
"""

from __future__ import annotations

import numpy as np

from emet.motion.agent_mcts import (
    AgentMCTSPlanner,
    MCTSAction,
    MCTSConfig,
    PickPlaceDistancePolicy,
)


def make_state(*, robot=(0.0, 0.0), obj=(2.0, 0.0), carrying=False, receptacle=(5.0, 0.0)) -> dict:
    return {
        "robot": np.asarray(robot, dtype=float),
        "object": np.asarray(obj, dtype=float),
        "carrying": bool(carrying),
        "receptacle": np.asarray(receptacle, dtype=float),
    }


def simulate(state: dict, action: MCTSAction) -> tuple[dict, float, bool]:
    """Deterministic functional pick-place step (no physics)."""
    next_state = {k: np.asarray(v, dtype=float) if isinstance(v, np.ndarray) else v for k, v in state.items()}
    next_state["carrying"] = bool(state["carrying"])
    obj = np.asarray(next_state["object"], dtype=float)
    rec = np.asarray(next_state["receptacle"], dtype=float)
    robot = np.asarray(next_state["robot"], dtype=float)
    reward = -float(action.cost)

    if action.name == "move_to":
        target = np.asarray(action.args["xy"], dtype=float)
        next_state["robot"] = target.copy()
        # reward = negative distance travelled + progress toward goal (0 if far)
        progress = max(0.0, float(np.linalg.norm(obj - rec) - np.linalg.norm(obj - target) if False else 0.0))
        return next_state, reward + progress, False

    if action.name == "pickup":
        if float(np.linalg.norm(robot - obj)) <= 0.25 and not bool(state["carrying"]):
            next_state["carrying"] = True
            return next_state, -0.1, False
        return state, -1.0, True

    if action.name == "place":
        if bool(state["carrying"]) and float(np.linalg.norm(robot - rec)) <= 0.30:
            next_state["carrying"] = False
            next_state["object"] = rec.copy()
            return next_state, 10.0, True  # goal achieved
        return state, -1.0, True

    return state, -float(action.cost), True


# ---------------------------------------------------------------------------


def test_policy_proposes_move_then_pickup_then_place():
    policy = PickPlaceDistancePolicy(seed=0)

    # far from the object -> only "move_to" is valid (no pickup within reach)
    cands = policy._propose_all(make_state(robot=(0.0, 0.0), obj=(2.0, 0.0), receptacle=(5.0, 0.0)), None)
    names = [c.action.name for c in cands]
    assert names == ["move_to"]

    # next to the object -> pickup appears
    cands = policy._propose_all(make_state(robot=(1.9, 0.0), obj=(2.0, 0.0), receptacle=(5.0, 0.0)), None)
    assert {c.action.name for c in cands} == {"move_to", "pickup"}

    # carrying, far from receptacle -> move only
    cands = policy._propose_all(
        make_state(robot=(2.0, 0.0), obj=(2.0, 0.0), carrying=True, receptacle=(5.0, 0.0)), None
    )
    assert [c.action.name for c in cands] == ["move_to"]

    # carrying, at receptacle -> place available
    cands = policy._propose_all(
        make_state(robot=(4.9, 0.0), obj=(2.0, 0.0), carrying=True, receptacle=(5.0, 0.0)), None
    )
    assert {c.action.name for c in cands} == {"move_to", "place"}


def test_policy_sampling_weighted_toward_goal():
    policy = PickPlaceDistancePolicy(seed=1)
    # carrying: place reduces distance-to-goal to 0 -> must be highest prior
    cands = policy._propose_all(
        make_state(robot=(4.9, 0.0), obj=(2.0, 0.0), carrying=True, receptacle=(5.0, 0.0)), None
    )
    prior = {c.action.name: c.prior for c in cands}
    assert prior["place"] > prior["move_to"]

    samples = [
        policy(make_state(robot=(4.9, 0.0), obj=(2.0, 0.0), carrying=True, receptacle=(5.0, 0.0)), None, 2)
        for _ in range(50)
    ]
    assert all(any(c.action.name == "place" for c in s) for s in samples)


def test_sampling_is_stochastic_within_distribution():
    a = PickPlaceDistancePolicy(seed=0)
    b = PickPlaceDistancePolicy(seed=1)
    state = make_state(robot=(1.9, 0.0), obj=(2.0, 0.0), receptacle=(5.0, 0.0))
    first = [c.action.name for c in a(state, None, 2)]
    second = [c.action.name for c in b(state, None, 2)]
    assert isinstance(first, list) and all(x in {"move_to", "pickup"} for x in first)
    assert isinstance(second, list)


def test_mcts_finds_pick_place_sequence():
    policy = PickPlaceDistancePolicy(seed=3)
    config = MCTSConfig(n_iterations=120, expansion_breadth=4, depth_limit=5, uct_c=1.2, seed=3)
    planner = AgentMCTSPlanner(policy=policy, simulate=simulate, config=config)

    start = make_state(robot=(0.0, 0.0), obj=(2.0, 0.0), carrying=False, receptacle=(5.0, 0.0))
    goal = np.asarray([5.0, 0.0], dtype=float)
    plan = planner.search(start, goal)

    assert plan, "MCTS must return a non-empty plan"
    names = [a.name for a in plan]
    assert "pickup" in names and "place" in names
    assert names.index("pickup") < names.index("place")

    # execute the plan -> object must end up at the receptacle
    state = start
    for action in plan:
        state, _, _ = simulate(state, action)
    assert float(np.linalg.norm(np.asarray(state["object"]) - goal)) <= 1e-6
    assert state["carrying"] is False


def test_mcts_greedy_goal_reached_directly_when_trivial():
    policy = PickPlaceDistancePolicy(seed=0)
    config = MCTSConfig(n_iterations=20, expansion_breadth=4, depth_limit=3, uct_c=0.6, seed=0)
    planner = AgentMCTSPlanner(policy=policy, simulate=simulate, config=config)

    start = make_state(robot=(4.9, 0.0), obj=(2.0, 0.0), carrying=True, receptacle=(5.0, 0.0))
    goal = np.asarray([5.0, 0.0], dtype=float)
    plan = planner.search(start, goal)
    assert plan
    assert plan[0].name == "place"
    state = start
    for action in plan:
        state, _, _ = simulate(state, action)
    assert float(np.linalg.norm(np.asarray(state["object"]) - goal)) <= 1e-6


def test_search_preserves_state_functionality():
    """simulate must not mutate the caller's state (functional step)."""
    state = make_state(robot=(1.9, 0.0), obj=(2.0, 0.0))
    before = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in state.items()}
    next_state, _, _ = simulate(state, MCTSAction("pickup", {"object": "target"}))
    for key in state:
        if isinstance(before[key], np.ndarray):
            assert np.allclose(state[key], before[key]), f"{key} mutated"
        else:
            assert state[key] == before[key], f"{key} mutated"
    assert next_state["carrying"] is True
