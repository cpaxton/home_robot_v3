# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from emet.controller.dynamem.relative_motion import execute_relative_motion


def agent_fixture(rejection=None, arrived=True):
    pose = np.array([1.0, 2.0, np.pi / 2])

    def execute(goals, timeout):
        if arrived:
            pose[:] = goals[-1]
        return arrived, None if arrived else "aborted_waypoint_execution"

    return SimpleNamespace(
        space=SimpleNamespace(obstacle_map_mode="physical"),
        planner=SimpleNamespace(reset=MagicMock()),
        _current_planning_xyt=lambda: pose.copy(),
        _filter_unsafe_nav_traj=MagicMock(return_value=([], rejection, None)),
        _execute_validated_waypoints=MagicMock(side_effect=execute),
        _find_phase_nav_timeout=lambda: 5,
    )


@pytest.mark.parametrize(
    "tool,value,delta",
    [
        ("move_forward", 0.5, [0.5, 0, 0]),
        ("rotate_base", -90, [0, 0, -np.pi / 2]),
    ],
)
def test_relative_motion_is_body_frame_and_validated(tool, value, delta):
    agent = agent_fixture()
    out = execute_relative_motion(agent, tool, value)
    assert out["code"] == "ok"
    np.testing.assert_allclose(out["data"]["measured_body_delta_xyt"], delta, atol=1e-8)
    assert agent._filter_unsafe_nav_traj.call_args.kwargs["explore_goal"] is False


def test_unknown_floor_rejection_never_executes():
    agent = agent_fixture(rejection="rejected_swept_footprint:unobserved_footprint")
    out = execute_relative_motion(agent, "move_forward", 0.2)
    assert out["code"] == "motion_rejected"
    assert out["data"]["execution_started"] is False
    agent._execute_validated_waypoints.assert_not_called()


@pytest.mark.parametrize("value", [True, None, "0.5", float("nan"), float("inf"), -0.1, 1.6])
def test_bad_arguments_never_execute(value):
    agent = agent_fixture()
    assert execute_relative_motion(agent, "move_forward", value)["status"] == "error"
    agent._execute_validated_waypoints.assert_not_called()


def test_failed_motion_retains_actual_displacement():
    out = execute_relative_motion(agent_fixture(arrived=False), "move_forward", 0.2)
    assert out["code"] == "aborted_waypoint_execution"
    assert out["data"]["measured_body_delta_xyt"] == [0.0, 0.0, 0.0]


def test_missing_capability_fails_closed():
    assert execute_relative_motion(SimpleNamespace(), "rotate_base", 90)["code"] == "validated_motion_unavailable"


def test_eqa_and_chat_share_motion_schemas_only_when_capable():
    from emet.agent.skills.eqa import bind_eqa_episode_tools
    from emet.agent.skills.specs import CHAT_SKILL_SPECS

    executor = SimpleNamespace(agent=agent_fixture(), mode="answer")
    tools = {tool.name: tool for tool in bind_eqa_episode_tools(executor)}
    specs = {spec.name: spec for spec in CHAT_SKILL_SPECS}
    for name in ("move_forward", "rotate_base"):
        assert tools[name].parameters == specs[name].parameters
    assert "plan_task" not in tools
    executor.agent = SimpleNamespace()
    names = {tool.name for tool in bind_eqa_episode_tools(executor)}
    assert not names.intersection({"move_forward", "rotate_base", "plan_task"})


def test_eqa_motion_budget_prevents_execution():
    from emet.memory.graph_eqa.agentic.executor_init import _handle_relative_motion

    executor = SimpleNamespace(_n_nav=2, max_nav_steps=2, _append_trace=MagicMock())
    out = _handle_relative_motion(executor, {"_tool_name": "move_forward", "meters": 0.2})
    assert out["code"] == "navigation_budget_exhausted"
    assert out["ok"] is False


def test_eqa_failed_motion_still_captures_actual_view():
    from emet.memory.graph_eqa.agentic.executor_init import _handle_relative_motion

    executor = SimpleNamespace(
        agent=agent_fixture(arrived=False),
        _n_nav=0,
        max_nav_steps=2,
        query_text="microwave",
        mode="answer",
        _begin_policy_approach=MagicMock(return_value="motion-view"),
        _policy_approached=MagicMock(),
        _tool_capture_and_update=MagicMock(return_value={"ok": True, "obs_id": 123}),
        _verify_after_motion=MagicMock(return_value={"present": True}),
        _append_trace=MagicMock(),
    )
    out = _handle_relative_motion(executor, {"_tool_name": "move_forward", "meters": 0.2})
    assert out["ok"] is False
    assert out["capture"]["obs_id"] == 123
    assert out["verify"]["present"] is True
    assert executor._n_nav == 1


def test_rejected_motion_preserves_navigation_budget():
    from emet.memory.graph_eqa.agentic.executor_init import _handle_relative_motion

    executor = SimpleNamespace(
        agent=agent_fixture(rejection="unknown_floor"),
        _n_nav=0,
        max_nav_steps=2,
        _append_trace=MagicMock(),
    )
    assert not _handle_relative_motion(executor, {"_tool_name": "move_forward", "meters": 0.2})["ok"]
    assert executor._n_nav == 0
