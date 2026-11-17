# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Task gaze uses measured motion and evidence, without floor or scan policy."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from emet.agent.tools import get_tools
from emet.controller.dynamem import look
from emet.controller.dynamem.head_observation import capture_head_view
from emet.robots.head_capability import HeadCapability


def scene(monkeypatch):
    angles = np.array([0.1, -0.6])
    obs = SimpleNamespace(rgb=np.full((4, 4, 3), 80, dtype=np.uint8), depth=None, image_timing={})
    robot = SimpleNamespace(
        _seq_id=1,
        get_head_capability=lambda: HeadCapability((-1.0, 1.0), (-1.8, 0.5), "fixture"),
        get_pan_tilt=lambda: angles.copy(),
        get_base_pose=lambda: np.zeros(3),
        get_observation=Mock(return_value=obs),
    )

    def move(pan, tilt, **kw):
        angles[:] = [pan, tilt]
        return True

    robot.head_to = Mock(side_effect=move)
    monkeypatch.setattr(look, "wait_post_motion_obs", lambda r, **kw: setattr(r, "_seq_id", r._seq_id + 1))
    agent = SimpleNamespace(robot=robot, look_around=Mock(), update=Mock())
    return agent, obs


def test_task_view_accepts_rgb_without_depth_and_retains_unspecified_joint(monkeypatch):
    agent, obs = scene(monkeypatch)
    out, captured = capture_head_view(agent, tilt_rad=0.2)
    assert out["ok"] and captured is obs
    agent.robot.head_to.assert_called_once_with(0.1, 0.2, blocking=True)
    assert out["observation"]["measured_head_pan_tilt_rad"] == [0.1, 0.2]
    assert out["observation"]["received_frame_sequence"] == 3
    agent.look_around.assert_not_called()
    agent.update.assert_not_called()  # Sensor evidence is not map/geometry admission.
    assert "not independently verified" in out["note"]


@pytest.mark.parametrize("angles", [{"tilt_rad": 0.6}, {"pan_rad": 1.1}, {"tilt_rad": float("nan")}, {"pan_rad": True}])
def test_out_of_range_task_gaze_is_rejected_with_limits_before_motion(monkeypatch, angles):
    agent, _ = scene(monkeypatch)
    out, obs = capture_head_view(agent, **angles)
    assert not out["ok"] and obs is None
    assert out["observation"]["head_inspection_bounds_rad"]["tilt"] == [-1.8, 0.5]
    agent.robot.head_to.assert_not_called()
    agent.robot.get_observation.assert_not_called()


@pytest.mark.parametrize("failure", ["rejected", "stale", "after_arrival_stale", "drift", "disconnected", "limits"])
def test_failed_head_or_capture_never_returns_a_successful_view(monkeypatch, failure):
    agent, _ = scene(monkeypatch)
    if failure == "rejected":
        agent.robot.head_to = Mock(return_value=False)
    elif failure in {"stale", "after_arrival_stale"}:

        def receive(robot, **kw):
            if failure == "after_arrival_stale" and robot._seq_id == 1:
                robot._seq_id += 1

        monkeypatch.setattr(look, "wait_post_motion_obs", receive)
    elif failure == "drift":
        agent.robot.get_base_pose = Mock(side_effect=[np.zeros(3), np.array([0.05, 0, 0])])
    elif failure == "disconnected":
        agent.robot.get_pan_tilt = Mock(side_effect=ConnectionError("disconnected"))
    else:
        agent.robot.get_head_capability = lambda: None
    out, captured = capture_head_view(agent, tilt_rad=-0.3)
    assert not out["ok"] and captured is None
    agent.robot.get_observation.assert_not_called()
    agent.update.assert_not_called()


def test_floor_constraints_remain_downward_and_require_depth(monkeypatch):
    agent, _ = scene(monkeypatch)
    out = look.observe_floor(agent, tilt_rad=-0.3)
    assert not out["ok"] and out["status"] == "invalid_head_tilt"
    agent.robot.head_to.assert_not_called()
    out = look.observe_floor(agent, tilt_rad=-1.0)
    assert not out["ok"] and out["status"] == "calibrated_depth_unavailable"
    agent.update.assert_not_called()


def test_chat_description_exposes_task_gaze_without_an_extra_caption_or_tool(monkeypatch):
    agent, _ = scene(monkeypatch)
    agent.describe_head_camera_scene_text = Mock()
    tools = {t.name: t for t in get_tools({"robot": agent.robot, "executor": SimpleNamespace(agent=agent)})}
    out = tools["describe_scene"].func(tilt_rad=-0.3)
    assert out.ok
    assert "measured_head_pan_tilt_rad" in out.render()
    agent.describe_head_camera_scene_text.assert_not_called()
    agent.robot.head_to.assert_called_once()
    assert len(tools) == 34


def test_eqa_handler_forwards_explicit_gaze_and_preserves_default_call():
    from emet.memory.graph_eqa.agentic.executor_init import _handle_look_around

    executor = SimpleNamespace(_tool_look_around=Mock(return_value={"ok": True}))
    _handle_look_around(executor, {"tilt_rad": -0.3})
    executor._tool_look_around.assert_called_once_with(tilt_rad=-0.3)
    executor._tool_look_around.reset_mock()
    _handle_look_around(executor, {})
    executor._tool_look_around.assert_called_once_with()


def test_eqa_explicit_gaze_captures_the_measured_observation_without_a_sweep(monkeypatch):
    from emet.memory.graph_eqa.agentic.capture import _tool_look_around

    agent, obs = scene(monkeypatch)
    agent.voxel_map = SimpleNamespace(observations=[])
    executor = SimpleNamespace(
        agent=agent,
        _tool_capture_and_update=Mock(return_value={"ok": True}),
        _refresh_room_after_motion=Mock(),
        _append_trace=Mock(),
    )
    out = _tool_look_around(executor, verify=False, tilt_rad=-0.3)
    assert out["ok"] and out["observation"]["measured_head_pan_tilt_rad"] == [0.1, -0.3]
    executor._tool_capture_and_update.assert_called_once_with(observation=obs)
    agent.look_around.assert_not_called()
    executor._tool_capture_and_update.reset_mock()
    out = _tool_look_around(executor, verify=False, tilt_rad=5.0)
    assert not out["ok"] and out["status"] == "invalid_head_tilt"
    executor._tool_capture_and_update.assert_not_called()
