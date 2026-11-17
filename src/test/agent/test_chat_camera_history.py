# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Verify session evidence at the actual CHAT/HTTP boundary."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

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


@pytest.mark.parametrize("task_ok", [True, False])
def test_chat_find_result_reaches_next_model_request(monkeypatch, task_ok):
    """Production FIND registration must reach the next real CHAT/HTTP request."""
    robot = setup_chat_loop(monkeypatch)
    executor = loop.DynamemTaskExecutor()
    executor.agent.planner = None
    executor.agent._last_nav_plan = {
        "outcome": "ok" if task_ok else "rejected_swept_footprint:unobserved_footprint",
        "footprint": {"coverage": 0.75},
    }

    def execute(cmds):
        assert cmds == [("find", "microwave")]
        executor._last_exec_ok = task_ok
        return True

    executor.side_effect = execute
    requests = []
    replies = iter(
        [
            {"tool_calls": [{"name": "find_objects", "arguments": {"text": "microwave"}}], "message": ""},
            {"tool_calls": [], "message": "Reporting the measured find result."},
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
    loop.run_agent_with_robot(
        discord=False,
        use_llm=True,
        commands=["Find the microwave and report its measured location."],
        device="cpu",
        robot_client=robot,
        max_tool_rounds=3,
        thinking_status=False,
    )
    executor.assert_called_once_with([("find", "microwave")])
    assert len(requests) == 2
    assert ("[find_objects] ok" if task_ok else "[find_objects] failed") in requests[1]
    assert '"coverage": 0.75' in requests[1]
    assert "Do not call any more tools" not in requests[1]
    if not task_ok:
        assert "unobserved_footprint" in requests[1]


@pytest.mark.parametrize("admitted", [True, False])
def test_chat_grounding_reaches_model_without_motion_or_answer_override(monkeypatch, admitted):
    """Exercise production registration, dispatch, dialogue retention and HTTP serialization."""
    robot = setup_chat_loop(monkeypatch)
    executor = loop.DynamemTaskExecutor()
    executor.agent.ground_current_view.return_value = {
        "ok": admitted,
        "status": "grounded" if admitted else "not_grounded",
        "reason": None if admitted else "no supported target surface",
        "geometry": {"status": "admitted", "xyz": [1, 2, 3]} if admitted else {"status": "not_admitted"},
        "observation": {"source_obs_id": 42, "view_id": "voxel-frame:42"},
    }
    requests = []
    replies = iter(
        [
            {"tool_calls": [{"name": "ground_object", "arguments": {"query": "microwave"}}], "message": ""},
            {"tool_calls": [], "message": "Reporting the grounding result."},
        ]
    )

    def create(**kwargs):
        requests.append(json.dumps(kwargs["messages"]))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(next(replies)), tool_calls=None))]
        )

    client = OpenaiClient("sys", model="emet", base_url="http://127.0.0.1:9/v1")
    client._openai = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(loop, "get_llm_client", lambda *a, **kw: client)
    goal = "Localize the visible microwave; do not move."
    loop.run_agent_with_robot(
        discord=False,
        use_llm=True,
        commands=[goal],
        device="cpu",
        robot_client=robot,
        max_tool_rounds=3,
        thinking_status=False,
    )
    assert len(requests) == 2 and goal in requests[1]
    assert "voxel-frame:42" in requests[1]
    assert (
        "[ground_object] ok status=grounded" if admitted else "[ground_object] failed status=not_grounded"
    ) in requests[1]
    executor.agent.ground_current_view.assert_called_once_with("microwave")
    executor.assert_not_called()


def test_chat_grounds_the_decision_rgbd_when_camera_changes_during_inference(monkeypatch):
    """A newer clipped frame must not replace the frame the coordinator chose."""
    import base64
    import copy
    from hashlib import sha256
    from io import BytesIO

    import numpy as np
    from PIL import Image

    robot = setup_chat_loop(monkeypatch)
    live = SimpleNamespace(
        rgb=np.full((4, 6, 3), 80, dtype=np.uint8),
        depth=np.ones((4, 6), dtype=np.float32),
        camera_K=np.eye(3),
        camera_pose=np.eye(4),
        gps=np.array([1.0, 2.0]),
        compass=np.array([0.3]),
        seq_id=17,
    )
    robot.get_observation.return_value = live
    executor = loop.DynamemTaskExecutor()
    captured = []

    def ground(query, *, observation, decision_view):
        assert query == "microwave"
        assert np.all(observation.rgb == 80) and np.all(observation.depth == 1)
        assert observation.camera_pose[0, 3] == 0
        assert observation.camera_K[0, 0] == 1
        assert decision_view["rgb_sha256"] == sha256(observation.rgb.tobytes()).hexdigest()
        captured.append(decision_view.copy())
        return {"ok": True, "status": "grounded", "geometry": {"xyz": [1, 2, 3]}, "observation": decision_view}

    executor.agent.ground_current_view.side_effect = ground
    requests = []
    replies = iter(
        [
            {"tool_calls": [{"name": "ground_object", "arguments": {"query": "microwave"}}], "message": ""},
            {"tool_calls": [], "message": "Reporting the observed position."},
        ]
    )

    def create(**kwargs):
        requests.append(copy.deepcopy(kwargs["messages"]))
        # The transport advances while the coordinator reasons over its image.
        live.rgb[:] = 0
        live.depth[:] = 9
        live.camera_pose[0, 3] = 99
        live.camera_K[0, 0] = 99
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(next(replies)), tool_calls=None))]
        )

    client = OpenaiClient("sys", model="emet", base_url="http://127.0.0.1:9/v1")
    client._openai = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(loop, "get_llm_client", lambda *a, **kw: client)
    loop.run_agent_with_robot(
        discord=False,
        use_llm=True,
        commands=["Ground the microwave without moving."],
        device="cpu",
        robot_client=robot,
        max_tool_rounds=3,
        thinking_status=False,
        vl_include_camera=True,
    )
    assert len(requests) == 2 and len(captured) == 1
    first = requests[0][-1]["content"]
    pixels = np.asarray(Image.open(BytesIO(base64.b64decode(first[-1]["image_url"]["url"].split(",", 1)[1]))))
    assert np.all(pixels == 80)
    assert captured[0]["decision_view_id"] in first[0]["text"]
    assert captured[0]["decision_view_id"] in json.dumps(requests[1][-1]["content"])
    assert requests[1][1] == requests[0][1]  # The earlier pixel packet stays exact in history.
    assert robot.get_observation.call_count == 2
    executor.assert_not_called()


def test_chat_does_not_reuse_prior_turn_snapshot_when_camera_disappears(monkeypatch):
    import numpy as np

    robot = setup_chat_loop(monkeypatch)
    robot.get_observation.side_effect = [SimpleNamespace(rgb=np.full((4, 6, 3), 80, dtype=np.uint8)), None, None]
    executor = loop.DynamemTaskExecutor()
    executor.agent._grounded_query_target = object()
    requests = []
    replies = iter(
        [
            {"tool_calls": [], "message": "Observed."},
            {"tool_calls": [{"name": "ground_object", "arguments": {"query": "microwave"}}], "message": ""},
            {"tool_calls": [], "message": "The decision view is unavailable."},
        ]
    )

    def create(**kwargs):
        requests.append(json.dumps(kwargs["messages"]))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(next(replies)), tool_calls=None))]
        )

    client = OpenaiClient("sys", model="emet", base_url="http://127.0.0.1:9/v1")
    client._openai = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(loop, "get_llm_client", lambda *a, **kw: client)
    loop.run_agent_with_robot(
        discord=False,
        use_llm=True,
        commands=["Observe.", "Ground the microwave."],
        device="cpu",
        robot_client=robot,
        max_tool_rounds=3,
        thinking_status=False,
        vl_include_camera=True,
    )
    assert len(requests) == 3
    assert "No camera view attached for this decision" in requests[-1]
    executor.agent.ground_current_view.assert_not_called()
    assert executor.agent._grounded_query_target is None


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
    assert requests[1][1]["content"][0]["text"].startswith("Observe this\n\n")
    assert requests[1][1]["content"][1]["type"] == "image_url"


def test_chat_session_trims_real_http_history_and_labels_cached_acquisition(monkeypatch):
    import copy

    import numpy as np

    robot = setup_chat_loop(monkeypatch)
    robot.get_observation.return_value = SimpleNamespace(
        rgb=np.full((4, 6, 3), 80, dtype=np.uint8),
        _seq_id=17,
        image_timing={"head_cam_left": {"timestamp_ns": 123, "clock_domain": "fixture"}},
    )
    requests = []

    def create(**kwargs):
        requests.append(copy.deepcopy(kwargs["messages"]))
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"tool_calls": [], "message": "done"}', tool_calls=None)
                )
            ]
        )

    client = OpenaiClient("current system", model="emet", base_url="http://127.0.0.1:9/v1", max_history_length=8)
    client._openai = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(loop, "get_llm_client", lambda *a, **kw: client)
    monkeypatch.setattr(loop, "get_tools", lambda ctx: [])
    turns = iter([*[f"inspect {i}" for i in range(8)], "Unrelated: help me write a poem.", None])
    loop.run_agent_with_robot(
        discord=False,
        use_llm=True,
        robot_client=robot,
        input_reader=lambda _: next(turns),
        device="cpu",
        vl_include_camera=True,
        thinking_status=False,
    )
    assert len(requests) == 9 and all(len(messages) <= 8 for messages in requests)
    assert all(messages[0] == {"role": "system", "content": "current system"} for messages in requests)
    assert len(client.get_history()) <= 8
    text = requests[-1][-1]["content"][0]["text"]
    assert text.startswith("Unrelated: help me write a poem.")
    assert "may be cached" in text and "freshness is unverified" in text
    assert '"frame_sequence": 17' in text and '"timestamp_ns": 123' in text
    assert "inspect 0" not in str(requests[-1])


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


def test_chat_recovery_preserves_history_and_shows_resulting_camera(monkeypatch):
    """The HTTP follow-up contains both views and measured partial-motion feedback."""
    import copy

    import numpy as np

    robot = setup_chat_loop(monkeypatch)
    robot.get_observation.side_effect = [
        SimpleNamespace(rgb=np.full((4, 6, 3), value, dtype=np.uint8)) for value in (40, 160)
    ]
    requests = []
    replies = iter(
        [
            {"tool_calls": [{"name": "move_forward", "arguments": {"meters": 0.2}}], "message": ""},
            {"tool_calls": [], "message": "The robot stopped short; the target is still visible."},
        ]
    )

    def create(**kwargs):
        requests.append(copy.deepcopy(kwargs["messages"]))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(next(replies)), tool_calls=None))]
        )

    client = OpenaiClient("sys", model="emet", base_url="http://127.0.0.1:9/v1")
    client._openai = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(loop, "get_llm_client", lambda *a, **kw: client)

    def move(meters):
        assert meters == 0.2
        return {
            "schema_version": 1,
            "tool": "move_forward",
            "status": "partial",
            "code": "motion_incomplete",
            "message": "Stopped before the requested endpoint.",
            "data": {"measured_distance_m": 0.08, "requested_distance_m": meters},
        }

    monkeypatch.setattr(
        loop,
        "get_tools",
        lambda ctx: [
            Tool(
                name="move_forward",
                description="move",
                parameters={"type": "object", "properties": {}},
                func=move,
                returns_info=True,
            )
        ],
    )
    loop.run_agent_with_robot(
        discord=False,
        use_llm=True,
        commands=["Move closer to inspect the object."],
        robot_client=robot,
        device="cpu",
        max_tool_rounds=2,
        vl_include_camera=True,
        thinking_status=False,
    )
    assert len(requests) == 2
    before = requests[0][-1]["content"]
    after = requests[1][-1]["content"]
    assert requests[1][1]["content"] == before
    assert before[1]["image_url"] != after[1]["image_url"]
    text = after[0]["text"]
    assert "Earlier images in the dialogue are historical" in text
    assert "motion_incomplete" in text and '"measured_distance_m": 0.08' in text
