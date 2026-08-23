# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Stubbed agent tool sequences for manip / motion-planning integration (no LLM/VLM)."""

from __future__ import annotations

from unittest.mock import MagicMock

from emet.agent.loop import _dispatch_tool_calls
from emet.agent.tools import get_tools
from emet.controller.manipulation.kinematic_pick_place import (
    KinematicPickPlaceExecutor,
    KinematicPickPlaceResult,
)

# Canned CHAT tool_calls: find then pick_place (no LLM).
_CANNED_FIND_THEN_PICK_PLACE = [
    {"name": "find_objects", "arguments": {"text": "bowl"}},
    {
        "name": "pick_place",
        "arguments": {"object_name": "bowl", "receptacle_name": "table"},
    },
]


class _RecordingExecutor:
    """Stand-in DynamemTaskExecutor that records command batches."""

    def __init__(self, *, last_exec_ok: bool = True) -> None:
        self.calls: list[list[tuple[str, str]]] = []
        self._last_exec_ok = last_exec_ok

    def __call__(self, cmds):
        self.calls.append(list(cmds))
        return True


def test_canned_find_then_pick_place_dispatch_order():
    """Agent dispatch maps find_objects → find, pick_place → guarded fallback commands."""
    exe = _RecordingExecutor()
    tools = {t.name: t for t in get_tools({"executor": exe})}
    ok, results, _has_info = _dispatch_tool_calls(
        list(_CANNED_FIND_THEN_PICK_PLACE),
        tools,
        exe,  # type: ignore[arg-type]
    )
    assert ok
    assert exe.calls == [
        [("find", "bowl")],
        [("pickup", "bowl"), ("place", "table")],
    ]
    assert "[pick_place] Pick and place (bowl -> table) done." in results


def test_pick_place_dispatch_surfaces_last_exec_ok_failure():
    """Failed manip sets _last_exec_ok; dispatch summary says failed (not quit)."""
    exe = _RecordingExecutor(last_exec_ok=False)
    tools = {t.name: t for t in get_tools({"executor": exe})}
    ok, results, _has_info = _dispatch_tool_calls(
        [
            {
                "name": "pick_place",
                "arguments": {"object_name": "bowl", "receptacle_name": "table"},
            }
        ],
        tools,
        exe,  # type: ignore[arg-type]
    )
    assert ok  # keep going
    assert exe.calls == [[("pickup", "bowl"), ("place", "table")]]
    assert "[pick_place] Pick/place failed or interrupted." in results


def _make_kinematic_dynamem_executor():
    from emet.controller.task.dynamem.dynamem_task import DynamemTaskExecutor

    caps = {"sim_set_body_pose": True, "kinematic_manip": True}
    robot = MagicMock()
    # Avoid MagicMock auto-attrs: robot_id_from_client must resolve to a profiled id.
    robot._spec = None
    robot.robot_id = "rby1"
    robot.get_emet_session.return_value = {
        "is_simulation": True,
        "capabilities": caps,
        "emet_robot_id": "rby1",
    }
    robot.say = MagicMock()
    robot.move_to_nav_posture = MagicMock()
    robot.switch_to_navigation_mode = MagicMock()

    agent = MagicMock()
    agent.robot_say = MagicMock(return_value=None)
    agent.get_voxel_map = MagicMock(return_value=None)

    exe = object.__new__(DynamemTaskExecutor)
    exe.robot = robot
    exe.agent = agent
    exe.visual_servo = False
    exe.skip_confirmations = True
    exe.manipulation_only = False
    exe._manip_mode = "kinematic"
    exe._manip_collision = "none"
    exe._manip_planner = "rrt_connect"
    exe._last_sim_picked_body = None
    exe._last_exec_ok = True
    exe._kinematic_executor = None
    exe.discord_bot = None
    exe.emote_task = MagicMock()
    exe.grasp_object = None
    return exe


def test_canned_tool_sequence_selects_kinematic_mp(monkeypatch):
    """find_objects → pick_place through agent tools hits KinematicPickPlaceExecutor, not teleport."""
    from emet.simulation import sim_manipulation

    monkeypatch.delenv("EMET_MANIP_PLANNER", raising=False)
    exe = _make_kinematic_dynamem_executor()
    exe._find = MagicMock(return_value=None)  # type: ignore[method-assign]

    kin_calls: list[tuple] = []
    planners: list[str] = []
    teleport_calls: list[str] = []

    def fake_grasp_only(self, object_query, **_kwargs):
        kin_calls.append(("grasp", object_query))
        planners.append(self.manip_planner)
        return KinematicPickPlaceResult(True, "bowl_1", self.ee_body, 0.01, None, "ok")

    def fake_place_only(self, target_receptacle, object_gt_body=None, **_kwargs):
        kin_calls.append(("place", target_receptacle, object_gt_body))
        planners.append(self.manip_planner)
        return KinematicPickPlaceResult(True, object_gt_body, self.ee_body, None, 0.02, "ok")

    monkeypatch.setattr(KinematicPickPlaceExecutor, "grasp_only", fake_grasp_only)
    monkeypatch.setattr(KinematicPickPlaceExecutor, "place_only", fake_place_only)

    def _teleport_pickup(*_a, **_k):
        teleport_calls.append("pickup")
        return None

    def _teleport_place(*_a, **_k):
        teleport_calls.append("place")
        return False

    monkeypatch.setattr(sim_manipulation, "sim_teleport_pickup", _teleport_pickup)
    monkeypatch.setattr(sim_manipulation, "sim_teleport_place", _teleport_place)

    tools = {t.name: t for t in get_tools({"executor": exe})}
    ok, results, _has_info = _dispatch_tool_calls(
        list(_CANNED_FIND_THEN_PICK_PLACE),
        tools,
        exe,
    )
    assert ok
    assert exe._last_exec_ok is True
    assert kin_calls == [
        ("grasp", "bowl"),
        ("place", "table", "bowl_1"),
    ]
    assert teleport_calls == []
    assert planners == ["rrt_connect", "rrt_connect"]
    assert "[pick_place] Pick and place (bowl -> table) done." in results
    # find_objects, then a nav attempt each for pickup and place.
    assert exe._find.call_count == 3


def test_scene_tasks_does_not_use_stale_ithor_metadata_for_live_robocasa(monkeypatch, tmp_path):
    class _Robot:
        def get_emet_session(self):
            return {
                "is_simulation": True,
                "environment": {"kind": "robocasa", "scene": "kitchen", "index": 0},
                "sim_object_placements": {},
            }

    ithor = tmp_path / "ithor"
    ithor.mkdir()
    (ithor / "FloorPlan1_physics_metadata.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("emet.eval.scene_task_extractor.default_molmospaces_scenes_dir", lambda: tmp_path)

    tools = {t.name: t for t in get_tools({"robot": _Robot()})}

    result = tools["scene_tasks"].func()

    assert result == "No MolmoSpaces metadata matches the active simulation scene."
