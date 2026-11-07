# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Run the real dispatch loop through legacy action-rewrite triggers."""

from types import SimpleNamespace

import pytest

from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor


@pytest.mark.parametrize("policy", ["legacy", "model_final"])
@pytest.mark.parametrize("trigger", ["close_map", "prefer_explore", "camera_place", "explore_streak", "rejected"])
def test_model_final_dispatches_the_selected_tool_and_arguments(policy, trigger):
    agent = SimpleNamespace(parameters={"eqa": {"agentic_decision_policy": policy}}, graph_memory=None, robot=None)
    ex = AgenticEQAExecutor(agent, "Where is the rug?", router=False, max_rounds=2, max_nav_steps=4)
    selected = ("explore_frontier", {"toward": "bathroom"})
    if trigger in {"prefer_explore", "camera_place", "rejected"}:
        selected = ("investigate", {"obs_id": 1, "approach_index": 2})
    ex._extract_vlm_target = lambda: None
    ex._frontier_pick_out_dir = lambda: None
    ex._hold_detections_before_explore = lambda: False
    ex._frontier_count = lambda: 3
    ex._close_map_stay_hypothesis = lambda: SimpleNamespace(obs_id=7) if trigger == "close_map" else None
    ex._unused_detection_hypothesis = lambda: SimpleNamespace(obs_id=8) if trigger == "camera_place" else None
    ex._hypothesis_for_obs_id = lambda oid: SimpleNamespace(obs_id=oid)
    ex._hypothesis_is_camera_pose_place = lambda hyp: trigger == "camera_place"
    ex._next_untried_hypothesis = lambda: SimpleNamespace(obs_id=9)
    ex._append_trace = lambda row: None

    def route():
        ex._prefer_explore = trigger == "prefer_explore"
        ex._n_consecutive_explore = 99 if trigger == "explore_streak" else 0
        return [(selected[0], dict(selected[1]))], "vlm", {}

    if trigger == "rejected":
        agent.query_driven_memory = True
        agent.query_candidates = SimpleNamespace(records={1: SimpleNamespace(rejected_revision=1)})
    ex._route_tool_calls = route
    dispatched = []

    class Dispatched(RuntimeError):
        pass

    def dispatch(name, args):
        if name == "inspect_graph":
            return {"ok": True}
        dispatched.append((name, args))
        raise Dispatched()

    ex.handle_tool = dispatch
    with pytest.raises(Dispatched):
        ex.run()
    assert (dispatched[0] == selected) is (policy == "model_final")
