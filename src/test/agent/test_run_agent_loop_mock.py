# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Copyright (c) Hello Robot, Inc. All rights reserved.

"""Offline smoke: ``run_agent_with_robot`` with mocked ZMQ client and executor (no sim, no DynaMem load)."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from emet.agent.loop import run_agent_with_robot
from emet.agent.tool_outcome import ToolOutcome
from emet.agent.tools import Tool


class _DummyMemBackend:
    """Minimal backend so ``query_memory`` tool falls through to voxel localize_text."""

    def query_answer(self, *args, **kwargs):
        raise NotImplementedError


@pytest.mark.parametrize("case", ["observe_then_two_actions", "two_actions", "failure", "budget"])
def test_actual_chat_loop_continues_unfinished_goals_and_bounds_execution(case):
    executed = []
    llm_calls = []
    frames = []

    def action(step):
        executed.append(step)
        return ToolOutcome(ok=case != "failure", tool="act", note=f"step {step}")

    def call(name, **arguments):
        return {"name": name, "arguments": arguments}

    rounds = {
        "observe_then_two_actions": [[call("describe_scene")], [call("act", step=1)], [call("act", step=2)], []],
        "two_actions": [[call("act", step=1)], [call("act", step=2)], []],
        "failure": [[call("act", step=1), call("act", step=2)], [call("act", step=3)]],
        "budget": [[call("act", step=n)] for n in (1, 2, 3, 4)],
    }[case]

    def fake_llm(client, text, tools, debug, **kwargs):
        llm_calls.append((text, kwargs))
        return json.dumps({"message": "", "tool_calls": rounds[len(llm_calls) - 1]}), 0.01

    def observation():
        frame = np.full((8, 8, 3), len(frames), dtype=np.uint8)
        frames.append(frame)
        return SimpleNamespace(rgb=frame)

    robot = MagicMock()
    robot.get_observation.side_effect = observation
    tools = [
        Tool(
            "describe_scene", "Observe", {"type": "object", "properties": {}}, lambda: "two objects", returns_info=True
        ),
        Tool("act", "Act", {"type": "object", "properties": {}}, action, returns_info=False),
    ]
    with (
        patch("emet.agent.loop.StretchZmqClient", return_value=robot),
        patch("emet.agent.loop.DynamemTaskExecutor", side_effect=_make_executor),
        patch("emet.agent.loop.get_memory_backend", return_value=_DummyMemBackend()),
        patch("emet.agent.loop.print_memory_view_help_on_quit"),
        patch("emet.agent.loop.get_tools", return_value=tools),
        patch("emet.agent.loop.get_llm_client", return_value=SimpleNamespace(max_tokens=128)),
        patch("emet.agent.loop._call_llm", side_effect=fake_llm),
        patch("emet.agent.loop.ChatLog"),
    ):
        run_agent_with_robot(
            robot_ip="127.0.0.1",
            robot="stretch",
            discord=False,
            use_llm=True,
            commands=["Observe if needed, then perform step one and step two."],
            agent_config="dynav_config.yaml",
            device="cpu",
            vl_include_camera=True,
        )
    assert len(llm_calls) == len(rounds)
    assert executed == {"failure": [1], "budget": [1, 2, 3]}.get(case, [1, 2])
    assert llm_calls[0][1]["reset_context"] is True
    assert all(not kwargs["reset_context"] for _, kwargs in llm_calls[1:])
    if case in ("failure", "budget", "observe_then_two_actions"):
        assert "Do not call any more tools" in llm_calls[-1][0]
    if case != "failure":
        assert "unfinished" in llm_calls[1][0]
        assert "do not repeat completed actions" in llm_calls[1][0]
        assert np.any(llm_calls[0][1]["image"] != llm_calls[1][1]["image"])
    robot.stop.assert_called_once()


def _make_executor(robot, parameters, **kwargs):
    ex = MagicMock()
    ex.robot = robot
    ex.discord_bot = None
    ex._last_memory_save_path = None
    ex.agent = MagicMock()
    ex.agent.get_voxel_map.return_value = MagicMock()
    ex.agent.log = "."
    ex.agent.graph_memory = None
    ex.agent.robot = robot
    ex.agent.planner = None
    ex.__call__ = MagicMock(return_value=True)
    return ex


def test_run_agent_with_robot_quit_no_llm_no_discord():
    """Scripted QUIT exits cleanly; StretchZmqClient and DynamemTaskExecutor never touch real sim."""
    last_robot: dict[str, MagicMock] = {}

    def fake_stretch(*args, **kwargs):
        robot = MagicMock()
        robot.stop = MagicMock()
        robot.get_observation = MagicMock(return_value=None)
        last_robot["r"] = robot
        return robot

    with (
        patch("emet.agent.loop.StretchZmqClient", side_effect=fake_stretch),
        patch("emet.agent.loop.DynamemTaskExecutor", side_effect=_make_executor),
        patch("emet.agent.loop.get_memory_backend", return_value=_DummyMemBackend()),
        patch("emet.agent.loop.print_memory_view_help_on_quit"),
    ):
        run_agent_with_robot(
            robot_ip="127.0.0.1",
            robot="stretch",
            discord=False,
            use_llm=False,
            commands=["QUIT"],
            agent_config="dynav_config.yaml",
        )

    assert last_robot["r"].stop.call_count == 1


def test_merge_chat_agent_manip_parameters():
    """Chat ``agent:`` manip settings must reach the executor's parameters block."""
    from emet.agent.loop import _merge_chat_agent_manip_parameters
    from emet.config.loader import AgentSectionConfig
    from emet.core.parameters import Parameters

    params = Parameters(**{"agent": {"realtime": {"matching_distance": 0.5}}})
    section = AgentSectionConfig(manip_mode="kinematic", manip_collision="aabb", manip_planner="linear")
    _merge_chat_agent_manip_parameters(
        params,
        agent_config="dynav_config.yaml",
        robot="stretch",
        agent_section=section,
    )
    agent = params.get("agent")
    assert agent["manip_mode"] == "kinematic"
    assert agent["manip_collision"] == "aabb"
    assert agent["manip_planner"] == "linear"
    assert agent["realtime"]["matching_distance"] == 0.5


def test_merge_chat_agent_manip_parameters_falls_back_to_config():
    """Without an explicit section, the helper loads the chat ``agent:`` block from YAML."""
    from emet.agent.loop import _merge_chat_agent_manip_parameters
    from emet.core.parameters import Parameters

    params = Parameters()
    _merge_chat_agent_manip_parameters(
        params,
        agent_config="dynav_config.yaml",
        robot="stretch",
        agent_section=None,
    )
    agent = params.get("agent") or {}
    assert agent.get("manip_mode") == "teleport"
