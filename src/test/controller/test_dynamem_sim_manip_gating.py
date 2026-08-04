# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for sim GT pickup/place gating (nav miss + success reporting)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np


def _make_executor(*, manip_mode: str = "teleport", session_caps: dict | None = None):
    from emet.controller.task.dynamem.dynamem_task import DynamemTaskExecutor

    caps = session_caps or {"sim_set_body_pose": True}
    robot = MagicMock()
    robot.get_emet_session.return_value = {"is_simulation": True, "capabilities": caps}
    robot.say = MagicMock()
    robot.move_to_nav_posture = MagicMock()

    agent = MagicMock()
    agent.robot_say = MagicMock(return_value=None)
    agent.get_voxel_map = MagicMock(return_value=None)

    # Bypass heavy DynamemTaskExecutor.__init__ (GPU / sensors).
    exe = object.__new__(DynamemTaskExecutor)
    exe.robot = robot
    exe.agent = agent
    exe.visual_servo = False
    exe.skip_confirmations = True
    exe.manipulation_only = False
    exe._manip_mode = manip_mode
    exe._manip_collision = "none"
    exe._manip_planner = "rrt_connect"
    exe._last_sim_picked_body = None
    exe._last_exec_ok = True
    exe.discord_bot = None
    exe.emote_task = MagicMock()
    exe.grasp_object = None
    return exe


def test_pickup_runs_on_nav_miss_when_sim_teleport_available(monkeypatch):
    exe = _make_executor(manip_mode="teleport")
    exe._find = MagicMock(return_value=None)
    calls: list[tuple[str, object]] = []

    def _pickup(target, point=None, skip_confirmations=False):
        calls.append(("pickup", point))
        return True

    def _place(target, point=None):
        calls.append(("place", point))
        return True

    exe._pickup = _pickup  # type: ignore[method-assign]
    exe._place = _place  # type: ignore[method-assign]
    keep = exe([("pickup", "bowl"), ("place", "table")])
    assert keep is True
    assert calls == [("pickup", None), ("place", None)]
    assert exe._last_exec_ok is True


def test_failed_pickup_sets_last_exec_ok_and_skips_place():
    exe = _make_executor(manip_mode="teleport")
    exe._find = MagicMock(return_value=np.array([1.0, 2.0, 0.5]))
    place_calls: list[str] = []

    exe._pickup = MagicMock(return_value=False)  # type: ignore[method-assign]

    def _place(target, point=None):
        place_calls.append(target)
        return True

    exe._place = _place  # type: ignore[method-assign]
    keep = exe([("pickup", "bowl"), ("place", "table")])
    assert keep is True  # keep going (not quit)
    assert exe._last_exec_ok is False
    assert place_calls == []


def test_kinematic_mode_without_cap_still_allows_gt_manip():
    from emet.simulation.sim_manipulation import can_use_sim_gt_manip

    exe = _make_executor(manip_mode="kinematic", session_caps={"sim_set_body_pose": True})
    assert can_use_sim_gt_manip(exe.robot, manip_mode="kinematic", visual_servo=False)
    assert exe._can_sim_gt_manip()


def test_visual_servo_pickup_propagates_grasp_operation_failure():
    """Stretch visual-servo path must not report success when GraspObjectOperation fails."""
    exe = _make_executor(manip_mode="teleport", session_caps={"sim_set_body_pose": True})
    exe.visual_servo = True
    exe.match_method = "class"
    exe.robot.get_head_pose.return_value = np.eye(4)
    exe.robot.get_six_joints.return_value = [0.0, 0.5, 0.0, 0.0, 0.0, 0.0]
    exe.robot.arm_to = MagicMock()
    exe.robot.switch_to_manipulation_mode = MagicMock()
    exe.robot.look_front = MagicMock()
    exe._find = MagicMock(return_value=np.array([1.0, 0.0, 0.5]))

    grasp = MagicMock()
    grasp.was_successful.return_value = False
    exe.grasp_object = grasp

    assert exe._pickup("bowl", point=np.array([1.0, 0.0, 0.5])) is False
    grasp.assert_called_once()
    keep = exe([("pickup", "bowl")])
    assert keep is True
    assert exe._last_exec_ok is False


def test_visual_servo_pickup_propagates_agent_manipulate_failure():
    exe = _make_executor(manip_mode="teleport", session_caps={})
    exe.visual_servo = True
    exe.grasp_object = None
    exe.robot.get_head_pose.return_value = np.eye(4)
    exe.robot.switch_to_manipulation_mode = MagicMock()
    exe.robot.look_front = MagicMock()
    exe.agent.manipulate = MagicMock(return_value=False)

    assert exe._pickup("cup", point=None, skip_confirmations=True) is False
    exe.agent.manipulate.assert_called_once()


def test_visual_servo_place_propagates_agent_place_failure():
    exe = _make_executor(manip_mode="teleport", session_caps={})
    exe.visual_servo = True
    exe.robot.get_head_pose.return_value = np.eye(4)
    exe.robot.switch_to_manipulation_mode = MagicMock()
    exe.robot.move_to_nav_posture = MagicMock()
    exe.agent.place = MagicMock(return_value=False)
    exe._find = MagicMock(return_value=np.array([0.5, 0.0, 0.8]))

    assert exe._place("table", point=None) is False
    exe.agent.place.assert_called_once()
    keep = exe([("place", "table")])
    assert keep is True
    assert exe._last_exec_ok is False
