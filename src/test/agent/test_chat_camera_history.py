# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Verify session evidence at the actual CHAT/HTTP boundary."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from emet.agent import loop
from emet.agent.tools import Tool
from emet.llms.openai_client import OpenaiClient


def setup_chat_loop(monkeypatch):
    robot = MagicMock()
    robot.get_observation.return_value = None
    executor = MagicMock()
    executor.robot = robot
    executor.agent.robot = robot
    executor.agent.graph_memory = None
    executor._last_memory_save_path = None
    monkeypatch.setattr(loop, "StretchZmqClient", lambda **kw: robot)
    monkeypatch.setattr(loop, "DynamemTaskExecutor", lambda *a, **kw: executor)
    monkeypatch.setattr(loop, "get_memory_backend", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(loop, "ChatLog", MagicMock())
    monkeypatch.setattr(loop, "print_memory_view_help_on_quit", lambda *a: None)
    monkeypatch.setattr(loop, "print_embodied_model_report", lambda **kw: None)
    monkeypatch.setattr(loop, "get_llm_client", lambda *a, **kw: MagicMock())
    return robot


def test_live_chat_session_preserves_earlier_user_turn_and_camera(monkeypatch):
    """Actual two-turn CHAT loop and HTTP client; robot and transport are controlled."""
    import copy
    from types import SimpleNamespace

    import numpy as np

    from emet.llms.openai_client import OpenaiClient

    robot = setup_chat_loop(monkeypatch)
    robot.get_observation.return_value = SimpleNamespace(rgb=np.full((4, 6, 3), 80, dtype=np.uint8))
    requests = []

    def create(**kwargs):
        requests.append(copy.deepcopy(kwargs["messages"]))
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"tool_calls": [], "message": "observed"}', tool_calls=None)
                )
            ]
        )

    client = OpenaiClient("sys", model="emet", base_url="http://127.0.0.1:9/v1")
    client._openai = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(loop, "get_llm_client", lambda *a, **kw: client)
    monkeypatch.setattr(loop, "get_tools", lambda ctx: [])
    turns = iter(["Observe this", "What was visible?", None])
    factory = MagicMock(side_effect=AssertionError("must reuse the existing owner"))
    monkeypatch.setattr(loop, "StretchZmqClient", factory)
    loop.run_agent_with_robot(
        discord=False,
        use_llm=True,
        robot_client=robot,
        input_reader=lambda _prompt: next(turns),
        device="cpu",
        vl_include_camera=True,
        thinking_status=False,
    )
    assert len(requests) == 2
    factory.assert_not_called()
    robot.stop.assert_called_once()
    assert [m["role"] for m in requests[1]] == ["system", "user", "assistant", "user"]
    assert requests[1][1]["content"][0]["text"] == "Observe this"
    assert requests[1][1]["content"][1]["type"] == "image_url"


def test_actual_chat_continues_sensing_then_motion_within_budget(monkeypatch):
    """An info result cannot silently terminate an unfinished authorized task."""
    robot = setup_chat_loop(monkeypatch)
    actions, requests = [], []
    replies = iter(
        [
            {"tool_calls": [{"name": "scan_environment", "arguments": {}}], "message": ""},
            {"tool_calls": [{"name": "move_forward", "arguments": {"meters": 0.2}}], "message": ""},
            {"tool_calls": [], "message": "Moved after inspecting the floor."},
        ]
    )

    def create(**kwargs):
        requests.append(kwargs["messages"][-1]["content"])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(next(replies)), tool_calls=None))]
        )

    client = OpenaiClient("sys", model="emet", base_url="http://127.0.0.1:9/v1")
    client._openai = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(loop, "get_llm_client", lambda *a, **kw: client)

    def scan():
        actions.append("scan")
        return "Floor observed; replan before moving."

    def move(meters):
        actions.append(("move", meters))
        return {
            "schema_version": 1,
            "tool": "move_forward",
            "status": "ok",
            "code": "ok",
            "message": "measured motion",
            "data": {"distance_m": meters},
        }

    tools = [
        Tool(name=name, description=name, parameters={"type": "object", "properties": {}}, func=func, returns_info=True)
        for name, func in [("scan_environment", scan), ("move_forward", move)]
    ]
    monkeypatch.setattr(loop, "get_tools", lambda _ctx: tools)
    goal = "Inspect the floor, then move forward 0.2 meters safely."
    loop.run_agent_with_robot(
        discord=False,
        use_llm=True,
        commands=[goal],
        device="cpu",
        robot_client=robot,
        max_tool_rounds=3,
        thinking_status=False,
    )
    assert actions == ["scan", ("move", 0.2)]
    assert len(requests) == 3
    assert goal in requests[1] and "Do not call any more tools" not in requests[1]
