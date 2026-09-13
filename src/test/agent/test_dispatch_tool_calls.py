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

"""Regression tests for ordered tool dispatch and motion validation."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from emet.agent.loop import _dispatch_tool_calls
from emet.agent.tool_outcome import ToolOutcome
from emet.agent.tools import Tool, get_tools


@pytest.mark.parametrize("mode", ["structured", "dict", "exception", "executor", "executor_exception", "unknown"])
def test_failed_tool_stops_remaining_batch_without_quitting_session(mode):
    later = Mock(return_value="must not run")

    def fail():
        if mode == "exception":
            raise RuntimeError("sensor disconnected")
        if mode == "dict":
            return {"ok": False, "note": "rejected"}
        return ToolOutcome(False, note="rejected")

    executor = Mock(return_value=True)
    if mode == "executor_exception":
        executor.side_effect = RuntimeError("motion transport disconnected")
    executor._last_exec_ok = False
    tools = {
        "first": Tool(
            "first",
            "First",
            {},
            fail,
            executor_commands=(lambda args: [("find", "cup")]) if mode.startswith("executor") else None,
        ),
        "later": Tool("later", "Later", {}, later),
    }
    if mode == "unknown":
        del tools["first"]
    ok, results, failed = _dispatch_tool_calls([{"name": "first"}, {"name": "later"}], tools, executor)
    assert ok and failed
    assert len(results) == 1
    later.assert_not_called()


@pytest.mark.parametrize(
    "name,args",
    [
        ("rotate_base", {"degrees": 90}),
        ("move_forward", {"meters": 0.1}),
        ("scan_environment", {}),
        ("pick_place", {"object_name": "cup", "receptacle_name": "table"}),
    ],
)
def test_direct_motion_tools_propagate_executor_failure(name, args):
    executor = Mock(return_value=True)
    executor._last_exec_ok = False
    executor.agent = None
    executor.visual_servo = True
    tools = {tool.name: tool for tool in get_tools({"executor": executor})}
    result = tools[name].func(**args)
    assert isinstance(result, ToolOutcome) and not result.ok


@pytest.mark.parametrize("name,arg", [("rotate_base", "degrees"), ("move_forward", "meters")])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_motion_arguments_do_not_reach_executor(name, arg, value):
    executor = Mock()
    tools = {tool.name: tool for tool in get_tools({"executor": executor})}
    result = tools[name].func(**{arg: value})
    assert not result.ok
    executor.assert_not_called()


def test_rotate_base_clips_degrees_via_func():
    calls: list[list[tuple[str, str]]] = []

    def executor(cmds):
        calls.append(list(cmds))
        return True

    tools = {t.name: t for t in get_tools({"executor": executor})}
    assert tools["rotate_base"].executor_commands is None
    ok, results, failed = _dispatch_tool_calls(
        [{"name": "rotate_base", "arguments": {"degrees": 999}}],
        tools,
        executor,
    )
    assert ok
    assert not failed
    assert calls == [[("rotate_base", "360.0")]]
    assert any("360" in r for r in results)


def test_move_forward_caps_distance_and_rejects_invalid():
    calls: list[list[tuple[str, str]]] = []

    def executor(cmds):
        calls.append(list(cmds))
        return True

    tools = {t.name: t for t in get_tools({"executor": executor})}
    assert tools["move_forward"].executor_commands is None

    ok, results, failed = _dispatch_tool_calls(
        [{"name": "move_forward", "arguments": {"meters": 9}}],
        tools,
        executor,
    )
    assert ok and not failed
    assert calls == [[("move_forward", "1.5")]]
    assert any("1.50" in r for r in results)

    calls.clear()
    ok, results, failed = _dispatch_tool_calls(
        [{"name": "move_forward", "arguments": {"meters": "nope"}}],
        tools,
        executor,
    )
    assert ok and failed
    assert calls == []
    assert any("Invalid meters" in r for r in results)


def test_scan_then_describe_preserves_order():
    order: list[str] = []

    def executor(cmds):
        order.append(f"exec:{cmds[0][0]}")
        return True

    def describe_scene() -> str:
        order.append("describe")
        return "a room"

    tools_by_name = {
        "scan_environment": Tool(
            name="scan_environment",
            description="scan",
            parameters={"type": "object", "properties": {}, "required": []},
            func=lambda: "unused",
            executor_commands=lambda _args: [("scan_environment", "")],
        ),
        "describe_scene": Tool(
            name="describe_scene",
            description="describe",
            parameters={"type": "object", "properties": {}, "required": []},
            func=describe_scene,
            returns_info=True,
        ),
    }
    ok, results, failed = _dispatch_tool_calls(
        [
            {"name": "scan_environment", "arguments": {}},
            {"name": "describe_scene", "arguments": {}},
        ],
        tools_by_name,
        executor,
    )
    assert ok and not failed
    assert order == ["exec:scan_environment", "describe"]
    assert any("a room" in r for r in results)


def test_info_tool_exception_sets_failed():
    def boom() -> str:
        raise RuntimeError("sensor down")

    tools_by_name = {
        "describe_scene": Tool(
            name="describe_scene",
            description="describe",
            parameters={"type": "object", "properties": {}, "required": []},
            func=boom,
            returns_info=True,
        ),
    }
    ok, results, failed = _dispatch_tool_calls(
        [{"name": "describe_scene", "arguments": {}}],
        tools_by_name,
        lambda _cmds: True,
    )
    assert ok
    assert failed
    assert any("sensor down" in r for r in results)


def test_send_object_image_returns_info_and_stashes():
    import numpy as np

    from emet.agent.tools import PENDING_DISCORD_IMAGE_KEY, get_tools

    crop = np.zeros((8, 8, 3), dtype=np.uint8)
    crop[:] = 40

    class _Node:
        best_crop = crop

    class _SG:
        def get_node_by_label(self, label: str):
            return _Node() if label == "mug" else None

    class _Map:
        def get_scene_graph(self):
            return _SG()

    class _Agent:
        def get_voxel_map(self):
            return _Map()

    class _Executor:
        agent = _Agent()

    context: dict = {"executor": _Executor(), "discord_bot": object()}
    tools = {t.name: t for t in get_tools(context)}
    assert tools["send_object_image"].returns_info is True
    ok, results, failed = _dispatch_tool_calls(
        [{"name": "send_object_image", "arguments": {"object_label": "mug"}}],
        tools,
        lambda _cmds: True,
    )
    assert ok and not failed
    assert PENDING_DISCORD_IMAGE_KEY in context
    assert any("Queued crop" in r for r in results)


def test_status_discord_path_does_not_consume_pending_image():
    import numpy as np

    from emet.agent.tools import (
        PENDING_DISCORD_IMAGE_KEY,
        pending_discord_image_for_send,
        stash_discord_image,
    )

    ctx: dict = {}
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    assert stash_discord_image(ctx, img)
    # Thinking / status outbound must leave the stash for the final reply.
    assert pending_discord_image_for_send(ctx, attach_pending_image=False) is None
    assert PENDING_DISCORD_IMAGE_KEY in ctx
    taken = pending_discord_image_for_send(ctx, attach_pending_image=True)
    assert taken is not None
    assert PENDING_DISCORD_IMAGE_KEY not in ctx


def test_scan_environment_returns_info_via_func():
    from emet.agent.tools import get_tools

    calls: list[list[tuple[str, str]]] = []

    def executor(cmds):
        calls.append(list(cmds))
        return True

    tools = {t.name: t for t in get_tools({"executor": executor})}
    scan = tools["scan_environment"]
    assert scan.returns_info is True
    assert scan.executor_commands is None

    ok, results, failed = _dispatch_tool_calls(
        [{"name": "scan_environment", "arguments": {}}],
        tools,
        executor,
    )
    assert ok and not failed
    assert calls == [[("rotate_in_place", "")]]
    assert any("scan" in r.lower() for r in results)


def test_dispatch_unknown_tool_returns_user_facing_info():
    from emet.agent.loop import _dispatch_tool_calls, _format_fast_tool_reply

    ok, results, failed = _dispatch_tool_calls(
        [{"name": "move_forward", "arguments": {"meters": 0.5}}],
        {},
        executor=lambda _cmds: True,
    )
    assert ok and failed
    assert any("don't have a working" in r.lower() or "can't drive" in r.lower() for r in results)
    assert _format_fast_tool_reply(results)
