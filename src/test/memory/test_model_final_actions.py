# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Run the real dispatch loop through legacy action-rewrite triggers."""

from types import SimpleNamespace

import pytest

from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor


@pytest.mark.parametrize(
    "status",
    [
        "OBS_NOT_IN_EVIDENCE",
        "NOT_INVESTIGATE_CARD",
        "CAMERA_POSE_PLACE",
        "NAV_LOOP_BLOCKED",
        "STATION_OBS_NOT_PLACE",
        "DETECTIONS_REMAIN",
    ],
)
def test_model_final_returns_rejections_without_unselected_navigation(status):
    from unittest.mock import Mock

    agent = SimpleNamespace(
        parameters={"eqa": {"agentic_decision_policy": "model_final"}}, graph_memory=None, robot=None
    )
    ex = AgenticEQAExecutor(agent, "Find the target", router=False)
    ex.handle_tool = Mock(side_effect=AssertionError("Unselected tool executed"))
    tool = "explore_frontier" if status == "DETECTIONS_REMAIN" else "investigate"
    assert not ex._recover_failed_router_motion(tool=tool, out={"ok": False, "status": status, "listed_obs_ids": [13]})
    ex.handle_tool.assert_not_called()


@pytest.mark.parametrize("value", [-1, 4, True, 1.5, "0"])
def test_investigate_rejects_invalid_explicit_approach_before_navigation(value):
    from unittest.mock import Mock

    agent = SimpleNamespace(
        parameters={"eqa": {"agentic_decision_policy": "model_final"}},
        graph_memory=None,
        robot=None,
        navigate_to_target_pose=Mock(),
    )
    ex = AgenticEQAExecutor(agent, "Find the target", router=False)
    out = ex._tool_investigate(13, approach_index=value)
    assert not out["ok"] and out["status"] == "invalid_tool_arguments"
    agent.navigate_to_target_pose.assert_not_called()


@pytest.mark.parametrize(
    "reply",
    [
        "{}",
        '{"tool_calls":[{"name":"invented","arguments":{}}]}',
        RuntimeError("server unavailable"),
        TypeError("inference bug"),
        '{"tool_calls":[{"name":"look_around","arguments":{}}]}',
    ],
)
def test_model_final_router_never_falls_back_or_drops_dialogue(monkeypatch, reply):
    from unittest.mock import Mock

    from emet.memory.graph_eqa.agentic import router

    client = Mock(side_effect=reply) if isinstance(reply, Exception) else Mock(return_value=reply)
    gm = SimpleNamespace(eqa_client=client)
    agent = SimpleNamespace(parameters={"eqa": {"agentic_decision_policy": "model_final"}}, graph_memory=gm, robot=None)
    ex = AgenticEQAExecutor(agent, "Find the target", router=True)
    ex._unchanged_inspections = 2
    ex._ensure_router_prompt = lambda: None
    ex._system_prompt = "system"
    ex._tool_names = {"look_around"}
    ex._refresh_graph_room_estimate = lambda: None
    ex._room_visual_pack = lambda: ([], [], "")
    ex._robot_xyt_world = lambda: None
    ex._append_trace = lambda row: None
    ex._fallback_tool = Mock(side_effect=AssertionError("Unselected fallback action"))
    monkeypatch.setattr(router, "build_state_message", lambda executor: "actual state")
    if isinstance(reply, Exception):
        with pytest.raises(type(reply), match=str(reply)):
            ex._route_tool_calls()
    else:
        calls, selected_by, meta = ex._route_tool_calls()
        if "look_around" in reply:
            assert calls == [("look_around", {})]
        else:
            assert calls == [] and selected_by == "model_action_rejected"
            assert "NO_VALID_TOOL_CALLS" in str(ex._conversation.messages[-1])
    ex._fallback_tool.assert_not_called()
    client.assert_called_once()
    assert client.call_args.kwargs["conversation"] is ex._conversation


@pytest.mark.parametrize("policy", ["legacy", "model_final"])
def test_positive_assessment_does_not_stop_model_final_coordinator(policy):
    from unittest.mock import Mock

    from emet.memory.graph_eqa.agentic.policy import AgenticState

    agent = SimpleNamespace(parameters={"eqa": {"agentic_decision_policy": policy}}, graph_memory=None, robot=None)
    ex = AgenticEQAExecutor(agent, "Where is the bowl?", router=False, max_rounds=4, max_nav_steps=0)
    ex._voxel_eqa_client = SimpleNamespace()
    ex._extract_vlm_target = lambda: None
    ex._frontier_pick_out_dir = lambda: None
    ex._frontier_count = lambda: 0
    ex._hold_detections_before_explore = lambda: False
    ex._unused_detection_hypothesis = lambda: None
    ex._close_map_stay_hypothesis = lambda: None
    ex._append_trace = lambda row: None
    routes = iter([[("look_around", {})], [("look_around", {})], [("submit_answer", {"answer": "model answer"})]])
    ex._route_tool_calls = lambda: (next(routes), "vlm", {})
    final = {"ok": True, "answer": "model answer", "confidence": True}
    ex._do_submit_answer = Mock(return_value=final)
    dispatched = []

    def dispatch(name, args):
        if name == "inspect_graph":
            return {"ok": True}
        dispatched.append(name)
        if name == "submit_answer":
            return final
        ex._verified = True
        ex._evidence_policy.state = AgenticState.ANSWER
        return {"ok": True}

    ex.handle_tool = dispatch
    result = ex.run()
    assert result.answer == "model answer"
    # Explicit completion need not be budget-forced, even when no navigation
    # allowance remains. The distinction must survive the final summary flush.
    assert not result.budget_hit
    status = agent._agentic_eqa_summary["budget_status"]
    assert status["navigation_exhausted"] and status["navigation_limit"] == 0
    assert status["navigation_used"] == 0 and status["round_limit"] == 4
    if policy == "model_final":
        assert dispatched == ["look_around", "look_around", "submit_answer"]
        ex._do_submit_answer.assert_not_called()  # No automatic submission outside selected dispatch.
    else:
        assert dispatched == ["look_around"]
        ex._do_submit_answer.assert_called_once()


@pytest.mark.parametrize("capture_state", ["fresh", "unchanged", "capture_error", "assessment_error"])
def test_failed_investigation_reports_current_evidence_without_claiming_arrival(capture_state):
    from unittest.mock import Mock

    import numpy as np

    from emet.controller.habitat_nav import NavOutcome
    from emet.memory.graph_eqa.graph_memory import NavHypothesis

    agent = SimpleNamespace(
        parameters={"eqa": {"agentic_decision_policy": "model_final"}},
        graph_memory=None,
        robot=None,
        navigate_to_target_pose=Mock(return_value=NavOutcome.NO_TARGET),
    )
    ex = AgenticEQAExecutor(agent, "Where is the bowl?", router=False)
    ex._target_phrase = "bowl"
    ex._hypotheses = [NavHypothesis(phrase="bowl", obs_id=-3_000_000, xyz=np.ones(3), score=1, source="voxel")]
    ex._investigate_hypotheses = lambda: ex._hypotheses
    ex._hypothesis_nav_blocked = lambda oid: False
    ex._next_approach_index = Mock(return_value=2)
    ex._unused_detection_hypothesis = lambda: None
    ex._investigate_target_xyz = Mock(return_value=np.ones(3))
    ex._investigate_arrival_look_at_xy = lambda *a: (1, 1)
    ex._robot_xyt = lambda: np.zeros(3)
    ex._robot_xyt_world = lambda: np.zeros(3)
    for name in ("_begin_policy_approach", "_mark_approach_tried", "_attach_gt", "_append_trace", "_policy_approached"):
        setattr(ex, name, Mock())
    ex._apply_close_map_after_approach = Mock(return_value=SimpleNamespace(as_dict=lambda: {}))
    cap = {"ok": False, "status": "NO_NEW_OBS"} if capture_state == "unchanged" else {"ok": True, "obs_id": 42}
    ex._tool_capture_and_update = Mock(return_value=cap)
    if capture_state == "capture_error":
        ex._tool_capture_and_update.side_effect = RuntimeError("camera disconnected")
    ex.handle_tool = Mock(return_value={"ok": True, "visual": {"present": True}, "geometry": {"status": "admitted"}})
    if capture_state == "assessment_error":
        ex.handle_tool.side_effect = RuntimeError("model unavailable")
    out = ex._tool_investigate(-3_000_000, approach_index=0)
    ex._next_approach_index.assert_not_called()
    ex._investigate_target_xyz.assert_called_once_with(-3_000_000, 0)
    assert not out["ok"] and not out["nav_progress"] and not out["nav_finished"]
    assert out["nav_outcome"] == "no_target"
    ex._tool_capture_and_update.assert_called_once()
    if capture_state in {"fresh", "assessment_error"}:
        ex.handle_tool.assert_called_once_with("verify_siglip", {"phrase": "bowl", "obs_id": 42})
        assert out["capture"] == cap
        if capture_state == "assessment_error":
            assert out["verify"]["status"] == "assessment_failed"
        else:
            assert out["verify"]["visual"]["present"]
    else:
        ex.handle_tool.assert_not_called()
        assert out["verify"] is None
        assert not out["capture"]["ok"]


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
