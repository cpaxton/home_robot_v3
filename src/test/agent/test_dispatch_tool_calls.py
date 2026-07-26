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

from emet.agent.loop import _dispatch_tool_calls
from emet.agent.tools import Tool, get_tools


def test_rotate_base_clips_degrees_via_func():
    calls: list[list[tuple[str, str]]] = []

    def executor(cmds):
        calls.append(list(cmds))
        return True

    tools = {t.name: t for t in get_tools({"executor": executor})}
    assert tools["rotate_base"].executor_commands is None
    ok, results, has_info = _dispatch_tool_calls(
        [{"name": "rotate_base", "arguments": {"degrees": 999}}],
        tools,
        executor,
    )
    assert ok
    assert has_info
    assert calls == [[("rotate_base", "360.0")]]
    assert any("360" in r for r in results)


def test_move_forward_caps_distance_and_rejects_invalid():
    calls: list[list[tuple[str, str]]] = []

    def executor(cmds):
        calls.append(list(cmds))
        return True

    tools = {t.name: t for t in get_tools({"executor": executor})}
    assert tools["move_forward"].executor_commands is None

    ok, results, has_info = _dispatch_tool_calls(
        [{"name": "move_forward", "arguments": {"meters": 9}}],
        tools,
        executor,
    )
    assert ok and has_info
    assert calls == [[("move_forward", "1.5")]]
    assert any("1.50" in r for r in results)

    calls.clear()
    ok, results, has_info = _dispatch_tool_calls(
        [{"name": "move_forward", "arguments": {"meters": "nope"}}],
        tools,
        executor,
    )
    assert ok and has_info
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
    ok, results, has_info = _dispatch_tool_calls(
        [
            {"name": "scan_environment", "arguments": {}},
            {"name": "describe_scene", "arguments": {}},
        ],
        tools_by_name,
        executor,
    )
    assert ok and has_info
    assert order == ["exec:scan_environment", "describe"]
    assert any("a room" in r for r in results)


def test_info_tool_exception_sets_has_info():
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
    ok, results, has_info = _dispatch_tool_calls(
        [{"name": "describe_scene", "arguments": {}}],
        tools_by_name,
        lambda _cmds: True,
    )
    assert ok
    assert has_info
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
    ok, results, has_info = _dispatch_tool_calls(
        [{"name": "send_object_image", "arguments": {"object_label": "mug"}}],
        tools,
        lambda _cmds: True,
    )
    assert ok and has_info
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
    from emet.agent.loop import _FAST_REPLY_TOOLS, _should_skip_llm_summarize
    from emet.agent.tools import get_tools

    calls: list[list[tuple[str, str]]] = []

    def executor(cmds):
        calls.append(list(cmds))
        return True

    tools = {t.name: t for t in get_tools({"executor": executor})}
    scan = tools["scan_environment"]
    assert scan.returns_info is True
    assert scan.executor_commands is None
    assert "scan_environment" in _FAST_REPLY_TOOLS
    assert "go_home" not in _FAST_REPLY_TOOLS
    assert "hand_over" not in _FAST_REPLY_TOOLS

    ok, results, has_info = _dispatch_tool_calls(
        [{"name": "scan_environment", "arguments": {}}],
        tools,
        executor,
    )
    assert ok and has_info
    assert calls == [[("rotate_in_place", "")]]
    assert any("scan" in r.lower() for r in results)
    assert _should_skip_llm_summarize(
        [{"name": "scan_environment", "arguments": {}}],
        results,
    )


def test_dispatch_unknown_tool_returns_user_facing_info():
    from emet.agent.loop import _dispatch_tool_calls, _format_fast_tool_reply, _should_skip_llm_summarize

    ok, results, has_info = _dispatch_tool_calls(
        [{"name": "move_forward", "arguments": {"meters": 0.5}}],
        {},
        executor=lambda _cmds: True,
    )
    assert ok and has_info
    assert any("don't have a working" in r.lower() or "can't drive" in r.lower() for r in results)
    assert _should_skip_llm_summarize([{"name": "move_forward"}], results)
    assert _format_fast_tool_reply(results)
