# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Coverage recovery remains observation-only until fresh mapping succeeds."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from emet.agent.loop import _dispatch_tool_calls
from emet.agent.tools import Tool, get_tools
from emet.controller.dynamem import look


@pytest.mark.parametrize(
    "raw",
    [
        '{"tool_calls": [{"name": "observe_floor", "arguments": {}], "message": ""}',
        '{"tool_calls": [{"name": "wave", "message": "Goodbye!"}',
        '```json\n{"tool_calls": [{"name": "observe_floor", "arguments": {}]\n```',
        '{"tool_calls": "observe_floor", "message": ""}',
    ],
)
def test_malformed_tool_envelope_is_an_error_not_a_completed_turn(raw):
    from emet.agent.prompt import parse_tool_calls_response

    result = parse_tool_calls_response(raw)
    assert result["tool_calls"] == []
    assert result["format_error"] == "invalid_tool_call_json"


@pytest.mark.parametrize("raises", [False, True])
def test_shared_perception_does_not_replace_agent_conversation(raises):
    from emet.llms.base import AbstractLLMClient

    class Client(AbstractLLMClient):
        def __call__(self, *args, **kwargs):
            return ""

    client = Client("agent tools")
    original = [{"role": "system", "content": "agent tools"}, {"role": "user", "content": "put cup on table"}]
    client.conversation_history = original.copy()
    client._iterations = 2
    try:
        with client.preserve_conversation():
            client.reset()
            client.add_history({"role": "system", "content": "list object labels only"})
            if raises:
                raise RuntimeError("perception failed")
    except RuntimeError:
        pass
    assert client.get_history() == original
    assert client.steps == 2


def test_system_prompt_allows_only_explicit_observation_recovery():
    from emet.agent.prompt import build_agent_system_prompt

    prompt = build_agent_system_prompt(tools=get_tools({}), name="Robot")
    assert "explicitly lists recovery_tools" in prompt
    assert "uncertain payload" in prompt
    assert "without claiming success or attempting recovery" not in prompt


@pytest.mark.parametrize("disconnected", [False, True])
@pytest.mark.parametrize("phase", ["pre_grasp", "navigation"])
@pytest.mark.parametrize(
    "status",
    [
        "insufficient_floor_coverage",
        "no_reachable_workspace",
        "workspace_obstructed",
        "rejected_swept_footprint:unobserved_footprint",
        "rejected_swept_footprint:obstacle",
    ],
)
@pytest.mark.parametrize("payload_state", ["empty", "unknown"])
def test_pick_failure_exposes_reason_and_only_safe_recovery(status, payload_state, disconnected, phase):
    executor = Mock(return_value=True)
    executor._last_exec_ok = False
    executor.visual_servo = True
    executor.agent = SimpleNamespace(query_driven_memory=True)
    executor.last_query_manipulation = {
        "reason": "workspace rejected",
        "phase": phase,
        "payload_state": payload_state,
        "observed_after_action": True,
        "navigation": {
            "status": status,
            "reachable_cells": 28,
            "in_range_center_cells": {"disconnected": 12 if disconnected else 0},
        },
    }
    tools = {tool.name: tool for tool in get_tools({"executor": executor})}
    later = Mock()
    tools["later"] = Tool("later", "Must not run", {}, later)
    recovery = []
    ok, results, failed = _dispatch_tool_calls(
        [{"name": "pick_place", "arguments": {"object_name": "cup", "receptacle_name": "table"}}, {"name": "later"}],
        tools,
        executor,
        recovery_tools=recovery,
    )
    assert ok and failed
    later.assert_not_called()
    assert status in results[0] and '"reachable_cells": 28' in results[0]
    permitted = payload_state == "empty" and (
        (phase == "pre_grasp" and status in {"insufficient_floor_coverage", "no_reachable_workspace"})
        or (phase == "navigation" and status == "rejected_swept_footprint:unobserved_footprint")
    )
    assert recovery == (["observe_floor"] if permitted else [])
    if disconnected and recovery:
        assert "adjacent/lateral" in results[0]


@pytest.mark.parametrize("pan_rad", [None, -0.8, 0.8])
@pytest.mark.parametrize("failure", [None, "stale", "pose", "depth", "map", "unsupported", "sequence"])
def test_floor_observation_requires_fresh_measured_capture(monkeypatch, failure, pan_rad):
    robot = SimpleNamespace(_seq_id=1, head_to=Mock(return_value=True))
    expected_pan = 0.2 if pan_rad is None else pan_rad
    robot.get_pan_tilt = Mock(side_effect=[(0.2, -0.5), (expected_pan, 0 if failure == "pose" else -1)])
    robot.get_observation = Mock(
        return_value=SimpleNamespace(
            rgb=np.ones((2, 2, 3)),
            depth=None if failure == "depth" else np.ones((2, 2)),
            get_xyz_in_world_frame=lambda: np.ones((2, 2, 3)),
        )
    )
    monkeypatch.delenv("EMET_EQA_EPISODE_DIR", raising=False)
    agent = SimpleNamespace(
        robot=robot,
        voxel_map=SimpleNamespace(
            observations=[],
            get_2d_map=lambda: (torch.zeros((2, 2), dtype=torch.bool), torch.ones((2, 2), dtype=torch.bool)),
        ),
    )
    agent.update = Mock(side_effect=lambda **kw: None if failure == "map" else agent.voxel_map.observations.append(1))

    def receive(robot, timeout):
        if failure != "stale":
            robot._seq_id += 1

    monkeypatch.setattr(look, "wait_post_motion_obs", receive)
    if failure == "unsupported":
        robot.head_to = None
    if failure == "sequence":
        robot._seq_id = None
    result = look.observe_floor(agent, pan_rad=pan_rad)
    assert result["ok"] is (failure is None)
    if failure not in (None, "map"):
        agent.update.assert_not_called()
    if failure is None:
        robot.head_to.assert_called_once_with(expected_pan, -1.0, blocking=True)
        agent.update.assert_called_once_with(full_perception=True)


@pytest.mark.parametrize("pan_rad", [float("nan"), float("inf"), -1.01, 1.01, True, "left"])
def test_floor_observation_rejects_invalid_pan_before_motion(pan_rad):
    assert look.observe_floor(SimpleNamespace(), pan_rad=pan_rad) == {"ok": False, "status": "invalid_head_pan"}
