# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the
# root directory of this source tree.

"""Rich return strings from motion/manipulation tools (func-only path, not batch executor)."""

from unittest.mock import MagicMock

from emet.agent.tools import get_tools


class _RecordingExecutor:
    def __init__(self, ok: bool = True) -> None:
        self.calls: list[list[tuple[str, str]]] = []
        self.ok = ok
        self.agent = MagicMock()
        self.agent.get_voxel_map.return_value = None

    def __call__(self, cmds: list[tuple[str, str]]) -> bool:
        self.calls.append(list(cmds))
        return self.ok


def _minimal_context(executor_ok: bool = True):
    ex = _RecordingExecutor(ok=executor_ok)
    return {"executor": ex, "robot": None}


def test_find_objects_invokes_executor_via_func_only_and_includes_map_hint():
    ctx = _minimal_context(True)
    tools = get_tools(ctx)
    fo = next(t for t in tools if t.name == "find_objects")
    assert fo.to_executor({"text": "red cylinder"}) == []
    out = fo.func(text="red cylinder")
    assert "Executor ran" not in out
    assert "red cylinder" in out
    assert "Map" in out or "map" in out
    assert ctx["executor"].calls == [[("find", "red cylinder")]]


def test_find_objects_failure_message_lists_recovery_tools():
    ctx = _minimal_context(False)
    tools = get_tools(ctx)
    fo = next(t for t in tools if t.name == "find_objects")
    out = fo.func(text="mug")
    assert "failed" in out.lower() or "interrupted" in out.lower()
    assert "explore" in out.lower() or "scan_environment" in out
    assert "navigation_diagnostics" in out or "send_map_snapshot" in out


def test_find_objects_rejects_empty_target():
    ctx = _minimal_context(True)
    tools = get_tools(ctx)
    fo = next(t for t in tools if t.name == "find_objects")
    out = fo.func(text="   ")
    assert "missing" in out.lower() or "empty" in out.lower()
    assert ctx["executor"].calls == []


def test_pick_place_no_batch_executor_mapping():
    ctx = _minimal_context(True)
    tools = get_tools(ctx)
    pp = next(t for t in tools if t.name == "pick_place")
    assert pp.to_executor({"object_name": "a", "receptacle_name": "b"}) == []
    out = pp.func(object_name="cup", receptacle_name="table")
    assert "cup" in out and "table" in out
    assert ctx["executor"].calls == [[("pickup", "cup"), ("place", "table")]]


def test_pick_place_failure_includes_map_context():
    ctx = _minimal_context(False)
    tools = get_tools(ctx)
    pp = next(t for t in tools if t.name == "pick_place")
    out = pp.func(object_name="x", receptacle_name="y")
    assert "failed" in out.lower() or "interrupted" in out.lower()
    assert "find_objects" in out or "explore" in out.lower()
    assert "Map snapshot" in out or "map" in out.lower()


def test_scan_environment_returns_info_and_calls_rotate():
    ctx = _minimal_context(True)
    tools = get_tools(ctx)
    sc = next(t for t in tools if t.name == "scan_environment")
    assert sc.returns_info is True
    assert sc.to_executor({}) == []
    out = sc.func()
    assert "rotate" in out.lower() or "scan" in out.lower()
    assert ctx["executor"].calls == [[("rotate_in_place", "")]]


def test_gesture_tool_uses_func_path_only():
    ctx = _minimal_context(True)
    tools = get_tools(ctx)
    wave = next(t for t in tools if t.name == "wave")
    assert wave.to_executor({}) == []
    msg = wave.func()
    assert "success" in msg.lower() or "completed" in msg.lower()
    assert ctx["executor"].calls == [[("wave", "")]]


def test_query_scene_graph_truncates_very_long_open_vocab_dump():
    class MockSG:
        num_objects = 2

        def to_string(self) -> str:
            return "\n".join(f"rel{i}: A near B" for i in range(80))

    class MockVM:
        def get_scene_graph(self):
            return MockSG()

    ctx = _minimal_context(True)
    ctx["executor"].agent.get_voxel_map.return_value = MockVM()
    tools = get_tools({**ctx, "graph_memory_backend": None})
    qsg = next(t for t in tools if t.name == "query_scene_graph")
    out = qsg.func(question="relations?")
    assert "excerpt" in out or "truncated" in out.lower()
    assert out.count("\n") < 70
