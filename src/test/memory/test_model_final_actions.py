# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Run the real dispatch loop through legacy action-rewrite triggers."""

from types import SimpleNamespace

import pytest

from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor


@pytest.mark.parametrize("policy", ["legacy", "model_final"])
def test_per_round_state_respects_decision_policy(policy):
    from emet.memory.graph_eqa.agentic.tools import build_state_message

    agent = SimpleNamespace(parameters={"eqa": {"agentic_decision_policy": policy}}, graph_memory=None, robot=None)
    ex = AgenticEQAExecutor(agent, "Where is the microwave?", router=False)
    ex._recent_actions = ["measured failed motion"]
    ex._last_capture_status = "OK"
    ex._close_map_stay_hypothesis = lambda: SimpleNamespace(obs_id=3)
    ex._close_look_required = True
    ex._prefer_explore = True
    text = build_state_message(ex)
    assert "Recent actions: measured failed motion" in text and "Last capture: OK" in text
    for instruction in [
        "do NOT submit_answer",
        "Choose: inspect_graph",
        "do not explore away yet",
        "Prefer explore_frontier once",
        "Question needs a CLOSE LOOK",
    ]:
        assert (instruction in text) is (policy == "legacy")


@pytest.mark.parametrize("policy", ["legacy", "model_final"])
@pytest.mark.parametrize("mode", ["answer", "explore"])
def test_model_final_can_stop_before_legacy_verification_or_coverage_gate(policy, mode):
    from unittest.mock import Mock

    agent = SimpleNamespace(parameters={"eqa": {"agentic_decision_policy": policy}}, graph_memory=None, robot=None)
    ex = AgenticEQAExecutor(agent, "Where is the rug?" if mode == "answer" else None, router=False)
    ex._do_submit_answer = Mock(return_value={"ok": True, "answer": "model answer"})
    ex._do_finish = Mock(return_value={"ok": True, "answer": "model summary"})
    ex._explore_done = lambda: False
    out = ex._tool_submit_answer("model answer") if mode == "answer" else ex._tool_finish("model summary")
    handler = ex._do_submit_answer if mode == "answer" else ex._do_finish
    assert bool(out["ok"]) is (policy == "model_final")
    assert handler.called is (policy == "model_final")


def test_model_final_prompt_allows_observation_recovery_without_a_fixed_tool_sequence():
    from emet.memory.graph_eqa.agentic.tools import build_graph_eqa_system_prompt

    prompt = build_graph_eqa_system_prompt([], decision_policy="model_final")
    assert "You decide the final answer" in prompt
    assert "not proof that an object is absent" in prompt
    assert "no required INVESTIGATE/EXPLORE sequence" in prompt
    assert "Each turn: investigate" not in prompt
    assert "Prefer source=voxel" not in prompt
    legacy = build_graph_eqa_system_prompt([], decision_policy="legacy")
    assert "Each turn: investigate" in legacy


def test_explore_arrival_preserves_task_gaze(monkeypatch):
    """A successful route must not silently level the head before capture."""
    from unittest.mock import Mock

    import numpy as np

    from emet.controller.habitat_nav import NavOutcome
    from emet.memory.graph_eqa.agentic import view_quality

    robot = SimpleNamespace(look_ahead=Mock(), head_to=Mock())
    agent = SimpleNamespace(
        parameters={"eqa": {"agentic_decision_policy": "model_final"}},
        graph_memory=None,
        robot=robot,
        navigate_to_target_pose=Mock(return_value=NavOutcome.REACHED),
    )
    ex = AgenticEQAExecutor(agent, None, router=False, max_nav_steps=4)
    ex._unused_detection_hypothesis = lambda: None
    ex._escape_min_travel_m = lambda: 0
    ex._robot_xyt_world = lambda: np.zeros(3)
    for name in (
        "_begin_policy_approach",
        "_refresh_room_after_motion",
        "_policy_approached",
        "_retire_visited_frontier",
        "_attach_gt",
        "_append_trace",
        "_save_frontier_pick_panel",
    ):
        setattr(ex, name, Mock(return_value=None))
    capture = {"ok": True, "obs_id": 1}
    ex._tool_capture_and_update = Mock(return_value=capture)
    monkeypatch.setattr(
        "emet.controller.habitat_nav.pick_uncovered_explore_target", lambda *a, **kw: np.array([1, 0, 1])
    )
    monkeypatch.setattr(view_quality, "recover_exploration_view", lambda executor, captured: captured)
    result = ex._tool_explore_frontier()
    assert result["ok"] and result["capture"] == capture
    agent.navigate_to_target_pose.assert_called_once()
    ex._tool_capture_and_update.assert_called_once()
    robot.look_ahead.assert_not_called()
    robot.head_to.assert_not_called()


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
