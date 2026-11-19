# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Regression coverage for CHAT result, capture, and capability simplifications."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from emet.agent.loop import _dispatch_tool_calls
from emet.agent.tamp_dialogue import TampDialogueContext
from emet.agent.tamp_session import tamp_followup
from emet.agent.tool import Tool
from emet.agent.tool_outcome import ToolOutcome
from emet.agent.tools import PENDING_DISCORD_IMAGE_KEY, get_tools
from emet.controller.task.tamp.api import response


@pytest.mark.parametrize("text", ["Done.", "No image available.", "Scan interrupted or failed."])
def test_prose_never_establishes_action_success(text):
    outcome = ToolOutcome.coerce("scan_environment", text)
    assert not outcome.ok and outcome.status == "unverified_result"


def test_explicit_capture_and_scan_failures_are_typed():
    tools = {tool.name: tool for tool in get_tools({"executor": lambda calls: False})}
    for name in ("send_image", "scan_environment", "take_picture"):
        outcome = tools[name].func()
        assert isinstance(outcome, ToolOutcome)
        assert not outcome.ok and outcome.tool == name


@pytest.mark.parametrize("task_ok", [True, False])
def test_scan_keeps_executor_liveness_separate_from_task_success(task_ok):
    class Executor:
        _last_exec_ok = task_ok

        def __call__(self, commands):
            assert commands == [("rotate_in_place", "")]
            return True

    tools = {tool.name: tool for tool in get_tools({"executor": Executor()})}
    assert tools["scan_environment"].func().ok is task_ok


@pytest.mark.parametrize("partial", [False, True])
def test_tamp_receipts_reach_continuation_and_history_without_reparsing(monkeypatch, partial):
    envelope = response(
        "execute_task_plan",
        status="partial" if partial else "ok",
        code="command_outcome_unknown" if partial else "ok",
        data={"completed_steps": 1, "plan_ref": "opaque-plan", "failed_step_execution_started": None},
    )
    tool = Tool("execute_task_plan", "test", {}, lambda: envelope, returns_info=True)
    dispatched = _dispatch_tool_calls([{"name": tool.name}], {tool.name: tool}, Mock())
    assert dispatched.outcomes[0].payload == envelope
    # The runtime consumers must not decode their own rendered log messages.
    monkeypatch.setattr(ToolOutcome, "coerce", Mock(side_effect=AssertionError("receipt reparsed")))
    followup = tamp_followup("Move the objects", dispatched.outcomes)
    assert (followup is None) is partial
    if followup:
        assert "opaque-plan" in followup
    context = TampDialogueContext()
    context.record_turn("Move the objects", dispatched.outcomes)
    historical = context.prompt("Explain")
    assert '"completed_steps": 1' in historical
    assert "opaque-plan" not in historical
    assert '"failed_step_execution_started": null' in historical


def test_scene_caption_and_attachment_use_one_owned_observation():
    from emet.controller.dynamem import describe

    pixels = np.full((4, 4, 3), 22, dtype=np.uint8)
    depth = np.ones((4, 4), dtype=np.float32)
    robot = SimpleNamespace(get_observation=Mock(return_value=SimpleNamespace(rgb=pixels, depth=depth)))

    class Agent:
        parameters = {}
        _describe_scene_capture_rgb = describe._describe_scene_capture_rgb
        describe_head_camera_scene_text = describe.describe_head_camera_scene_text

        def _detection_cfg(self):
            return {}

        def _describe_scene_try_sources(self, rgb, captured_depth, config, **kwargs):
            assert not np.shares_memory(rgb, pixels)
            assert not np.shares_memory(captured_depth, depth)
            # Simulate the decoded live buffers changing during caption inference.
            pixels[:] = 99
            depth[:] = 9
            assert np.all(rgb == 22) and np.all(captured_depth == 1)
            return "Caption of the 22-valued frame"

    agent = Agent()
    agent.robot = robot
    context = {"robot": robot, "executor": SimpleNamespace(agent=agent)}
    tools = {tool.name: tool for tool in get_tools(context)}
    assert "22-valued" in tools["describe_scene"].func()
    robot.get_observation.assert_called_once()
    assert np.all(context[PENDING_DISCORD_IMAGE_KEY] == 22)


def test_graph_queries_share_success_path_but_keep_distinct_fallbacks():
    graph = Mock()
    graph.query_answer.return_value = ("reason", "graph answer", True, "confident", None, [])
    voxel = Mock()
    voxel.query_answer.return_value = ("reason", "voxel answer", True, "confident", None, [])
    scene_graph = SimpleNamespace(num_objects=1, to_string=lambda: "open-vocabulary snapshot")
    agent = SimpleNamespace(get_voxel_map=lambda: SimpleNamespace(get_scene_graph=lambda: scene_graph))
    context = {"graph_memory_backend": graph, "memory_backend": voxel, "executor": SimpleNamespace(agent=agent)}
    tools = {tool.name: tool for tool in get_tools(context)}
    for name in ("query_memory", "query_scene_graph"):
        assert "graph answer" in tools[name].func(question="where?")
    assert graph.query_answer.call_count == 2
    voxel.query_answer.assert_not_called()
    graph.query_answer.side_effect = RuntimeError("unavailable")
    assert "voxel answer" in tools["query_memory"].func(question="where?")
    assert "open-vocabulary snapshot" in tools["query_scene_graph"].func(question="where?")
    assert voxel.query_answer.call_count == 1


def test_rotate_only_policy_matches_advertised_refusals_and_prompt(monkeypatch):
    from emet.agent.prompt import build_agent_system_prompt
    from emet.agent.tool_policy import ROTATE_ONLY_BLOCKED_TOOLS

    monkeypatch.setenv("EMET_BASE_ROTATE_ONLY", "1")
    tools = {tool.name: tool for tool in get_tools({})}
    prompt = build_agent_system_prompt(list(tools.values()))
    restriction = prompt.split("# Base motion safety", 1)[1].split("Do NOT call ", 1)[1].split(". ", 1)[0]
    assert set(restriction.split(", ")) == set(ROTATE_ONLY_BLOCKED_TOOLS)
    for name in ROTATE_ONLY_BLOCKED_TOOLS:
        outcome = tools[name].func()
        assert outcome.ok is False and outcome.status == "base_rotate_only"
    # Missing provider-gated names use the same restriction at dispatch.
    result = _dispatch_tool_calls([{"name": "execute_task_plan"}], {}, Mock())
    assert result.has_info and not result.outcomes[0].ok
    assert "can't drive" in result.messages[0]


@pytest.mark.parametrize("payload", [{}, {"schema_version": 1, "data": 123}, response("plan_task")])
def test_typed_wrapper_cannot_bypass_tamp_envelope_validation(payload):
    outcome = ToolOutcome(True, tool="scene_tasks", payload=payload)
    later = Mock()
    tools = {
        "scene_tasks": Tool("scene_tasks", "test", {}, lambda: outcome, returns_info=True),
        "plan_task": Tool("plan_task", "test", {}, later, returns_info=True),
    }
    dispatched = _dispatch_tool_calls([{"name": "scene_tasks"}, {"name": "plan_task"}], tools, Mock())
    assert dispatched.outcomes[0].status == "invalid_tool_result"
    later.assert_not_called()
    assert tamp_followup("Continue", [outcome]) is None
    # Malformed data cannot enter a historical context as an admitted receipt.
    context = TampDialogueContext()
    context.record_turn("Discover", [outcome])
    assert "invalid_tool_result" in context.prompt("Explain")


@pytest.mark.parametrize("result", [None, "Done.", ToolOutcome(False, status="failed", note="Refused")])
def test_unverified_or_failed_action_requests_feedback_instead_of_silent_ack(result):
    tool = Tool("custom_action", "test", {}, lambda: result, returns_info=False)
    dispatched = _dispatch_tool_calls([{"name": tool.name}], {tool.name: tool}, Mock())
    assert dispatched.has_info
    assert not dispatched.outcomes[0].ok


def test_successful_action_only_command_does_not_force_extra_model_round():
    tool = Tool("wave", "test", {}, lambda: None, executor_commands=lambda args: [("wave", "")])
    dispatched = _dispatch_tool_calls([{"name": tool.name}], {tool.name: tool}, lambda commands: True)
    assert dispatched.continue_running and not dispatched.has_info
    assert dispatched.outcomes[0].ok


@pytest.mark.parametrize(
    "claimed_ok,status,code",
    [(True, "partial", "command_outcome_unknown"), (True, "error", "controller_failed"), (False, "ok", "ok")],
)
def test_typed_result_disagreement_never_permits_continuation(claimed_ok, status, code):
    envelope = response("execute_task_plan", status=status, code=code)
    wrapped = ToolOutcome(claimed_ok, tool="execute_task_plan", payload=envelope)
    normalized = ToolOutcome.coerce(wrapped.tool, wrapped)
    assert not normalized.ok and normalized.status == "invalid_tool_result"
    assert tamp_followup("Continue the transfer", [wrapped]) is None
    later = Mock()
    tools = {
        wrapped.tool: Tool(wrapped.tool, "test", {}, lambda: wrapped, returns_info=True),
        "scene_tasks": Tool("scene_tasks", "test", {}, later, returns_info=True),
    }
    dispatched = _dispatch_tool_calls([{"name": wrapped.tool}, {"name": "scene_tasks"}], tools, Mock())
    assert not dispatched.outcomes[0].ok
    later.assert_not_called()


@pytest.mark.parametrize("wrapped", [False, True])
def test_versioned_receipt_owns_nested_provider_data(wrapped):
    envelope = response("execute_task_plan", data={"completed_steps": 1, "completed_actions": [{"object_name": "cup"}]})
    original = ToolOutcome(True, tool="execute_task_plan", payload=envelope) if wrapped else envelope
    tool = Tool("execute_task_plan", "test", {}, lambda: original, returns_info=True)
    dispatched = _dispatch_tool_calls([{"name": tool.name}], {tool.name: tool}, Mock())
    envelope["data"]["completed_steps"] = 99
    envelope["data"]["completed_actions"][0]["object_name"] = "later object"
    data = dispatched.outcomes[0].payload["data"]
    assert data["completed_steps"] == 1
    assert data["completed_actions"] == [{"object_name": "cup"}]
    context = TampDialogueContext()
    context.record_turn("Move the cup", dispatched.outcomes)
    assert '"completed_steps": 1' in context.prompt("Explain")
    assert "later object" not in context.prompt("Explain")


def test_rejected_exploration_never_includes_success_summary(monkeypatch):
    import emet.agent.tools as tools_module

    monkeypatch.setattr(tools_module, "snapshot_from_voxel_map", lambda *args: (None, {"summary_lines": []}, None))

    class Executor:
        agent = SimpleNamespace(_last_nav_plan={"outcome": "rejected_low_clearance"}, get_voxel_map=lambda: None)
        _last_exec_ok = True

        def __call__(self, commands):
            return True

    tools = {tool.name: tool for tool in get_tools({"executor": Executor()})}
    outcome = tools["explore"].func()
    assert not outcome.ok
    assert "reported success" not in outcome.note


@pytest.mark.parametrize("name", ["execute_task_plan", "execute_pick_place_plan", "pick_place", "move_forward"])
def test_rotate_only_refusal_keeps_its_machine_reason_through_dispatch(monkeypatch, name):
    monkeypatch.setenv("EMET_BASE_ROTATE_ONLY", "1")
    tools = {tool.name: tool for tool in get_tools({})}
    executor = Mock()
    dispatched = _dispatch_tool_calls([{"name": name}], tools, executor)
    outcome = dispatched.outcomes[0]
    assert not outcome.ok and outcome.status == "base_rotate_only"
    assert outcome.payload["recovery"] == "none"
    assert "tethered" in outcome.note
    executor.assert_not_called()
