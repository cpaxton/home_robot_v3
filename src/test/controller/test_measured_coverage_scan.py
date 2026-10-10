# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Coverage means measured views, not successful motion requests."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from emet.controller.dynamem import look


@pytest.fixture
def rig(monkeypatch):
    now = [0.0]
    robot = SimpleNamespace(_seq_id=0, pose=np.array([1.0, 2.0, 2.9]), head=np.zeros(2))
    robot.get_base_pose = lambda: robot.pose.copy()
    robot.get_pan_tilt = lambda: robot.head.copy()
    robot.get_head_capability = lambda: SimpleNamespace(contains=lambda target: True)
    robot.get_observation = lambda: SimpleNamespace(
        camera_pose=np.eye(4),
        camera_K=np.eye(3),
        rgb=np.zeros((2, 2, 3)),
        depth=np.ones((2, 2)),
    )

    def head(pan, tilt, **kwargs):
        robot.head = np.array([pan, tilt])
        return None  # Legacy adapters still require measured arrival.

    def move(goal, **kwargs):
        robot.pose = np.array(goal)
        robot.head = np.zeros(2)  # Reacquire the view after any posture reset.
        return True

    def sleep(seconds):
        now[0] += seconds
        robot._seq_id += 1

    robot.head_to = Mock(side_effect=head)
    robot.move_base_to = Mock(side_effect=move)
    monkeypatch.setattr(look.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(look.time, "sleep", sleep)
    return robot, now


def test_full_scan_absolute_headings_and_reacquisition(rig):
    robot, _ = rig
    result = look.measured_coverage_scan(robot, lambda obs: 123)
    assert result["ok"] and result["capture_count"] == 16
    assert robot.move_base_to.call_count == 7
    assert robot.head_to.call_count == 16
    goals = np.array([v["requested_base_xyt"] for v in result["views"]])[::2]
    assert np.allclose(goals[:, :2], [1.0, 2.0])
    assert np.allclose(np.diff(np.unwrap(goals[:, 2])), np.pi / 4)
    assert robot.head[1] == pytest.approx(-np.pi / 3)


def test_failed_turn_preserves_partial_evidence(rig):
    robot, _ = rig
    robot.move_base_to.side_effect = lambda *a, **k: False
    result = look.measured_coverage_scan(robot, lambda obs: 1)
    assert result["status"] == "partial" and result["capture_count"] == 2
    assert result["reason"] == "navigation_failed"


def test_unachieved_head_is_not_captured(rig):
    robot, _ = rig
    robot.head_to.side_effect = lambda *a, **k: None
    result = look.measured_coverage_scan(robot, Mock(side_effect=AssertionError("unsafe capture")))
    assert result["reason"] == "head_pose_unconfirmed"
    assert result["capture_count"] == 0


def test_fixed_head_reports_gap_without_command(rig):
    robot, _ = rig
    robot.get_head_capability = lambda: None
    result = look.measured_coverage_scan(robot, lambda obs: 1)
    assert result["status"] == "partial" and result["capture_count"] == 8
    assert len(result["unsupported_views"]) == 2
    robot.head_to.assert_not_called()


def test_deadline_preserves_evidence(rig):
    robot, now = rig

    def capture(obs):
        now[0] += 1.0
        return 1

    result = look.measured_coverage_scan(robot, capture, deadline=0.5)
    assert result["reason"] == "deadline_exceeded" and result["capture_count"] == 1


def test_stale_frame_is_not_captured(rig, monkeypatch):
    robot, now = rig
    monkeypatch.setattr(look.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    result = look.measured_coverage_scan(robot, Mock(side_effect=AssertionError("stale capture")))
    assert result["reason"] == "stale_observation"


def test_sweep_rejection_prevents_motion(rig):
    robot, _ = rig
    result = look.measured_coverage_scan(robot, lambda obs: 0, validate_turn=lambda a, b: "unobserved_footprint")
    assert result["reason"] == "unsafe_turn" and result["capture_count"] == 2
    robot.move_base_to.assert_not_called()


def test_success_receipt_does_not_replace_pose_check(rig):
    robot, _ = rig
    robot.move_base_to.side_effect = lambda *a, **k: True
    result = look.measured_coverage_scan(robot, lambda obs: 0)
    assert result["reason"] == "base_pose_unconfirmed" and result["capture_count"] == 2


def test_capture_failure_retains_previous_views(rig):
    robot, _ = rig
    capture = Mock(side_effect=[0, RuntimeError("map full")])
    result = look.measured_coverage_scan(robot, capture)
    assert result["reason"] == "capture_failed" and result["capture_count"] == 1


def test_agent_maps_exact_validated_frame(rig):
    robot, _ = rig
    observations = []
    frames = []

    def update(*, full_perception, observation):
        assert full_perception
        observations.append(observation)
        frames.append(observation)

    agent = SimpleNamespace(
        robot=robot,
        voxel_map=SimpleNamespace(observations=observations),
        update=update,
        _filter_unsafe_nav_traj=lambda poses, **kwargs: (poses, None, None),
        _planning_base_xyt=lambda p: p,
    )
    result = look._coverage_scan_agent(agent)
    assert result["ok"] and len(frames) == 16
    assert [v["map_observation_index"] for v in result["views"]] == list(range(16))


@pytest.mark.parametrize("completed", [True, False])
def test_tool_never_adds_an_unchecked_capture(monkeypatch, completed):
    from emet.memory.graph_eqa.agentic import capture, views

    scan = {"ok": completed, "status": "completed" if completed else "partial"}
    executor = SimpleNamespace(
        agent=SimpleNamespace(look_around=Mock(return_value=scan)),
        _tool_capture_and_update=Mock(side_effect=AssertionError("extra unchecked capture")),
        _refresh_room_after_motion=Mock(),
        _append_trace=Mock(),
        _fresh_obs_ids=set(),
    )
    retain = Mock(return_value=SimpleNamespace(obs_id=42, view_id="view42", rgb=np.zeros((2, 2, 3))))
    monkeypatch.setattr(views, "retain_latest_view", retain)
    monkeypatch.setattr(capture, "dump_query_rgb", lambda *a, **k: {})
    result = capture._tool_look_around(executor, profile="coverage", verify=False)
    assert result["capture"]["ok"] is completed
    assert retain.call_count == int(completed)
    assert result["scan"]["status"] == scan["status"]
    executor._tool_capture_and_update.assert_not_called()
