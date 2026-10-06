# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from emet.memory.graph_eqa.agentic.view_quality import (
    aim_arrival_view,
    exploration_view_quality,
    recover_exploration_view,
)
from emet.memory.graph_eqa.agentic.views import CapturedView


def test_close_geometry_not_color_or_distant_wall_triggers_recovery():
    assert exploration_view_quality(np.full((20, 20), 0.2))["obstructed"]
    assert not exploration_view_quality(np.full((20, 20), 2.0))["obstructed"]
    assert not exploration_view_quality(None)["obstructed"]


def test_head_recovery_is_bounded_and_requires_current_capture():
    robot = SimpleNamespace(head_to=Mock(), wait_for_obs=Mock())
    vm = SimpleNamespace(observations=[])
    ex = SimpleNamespace(agent=SimpleNamespace(robot=robot, voxel_map=vm), _captured_views={}, _append_trace=Mock())

    def capture():
        vm.observations.append(SimpleNamespace(depth=np.full((20, 20), 0.2)))
        oid = len(vm.observations)
        ex._captured_views[oid] = CapturedView(oid, oid, np.zeros((20, 20, 3), dtype=np.uint8), None)
        return {"ok": True, "obs_id": oid}

    ex._tool_capture_and_update = capture
    result = recover_exploration_view(ex, capture())
    assert not result["ok"] and result["status"] == "OBSTRUCTED_VIEW"
    assert robot.head_to.call_count == 2
    assert len(vm.observations) == 3


def test_open_view_does_not_move_head():
    robot = SimpleNamespace(head_to=Mock())
    ex = SimpleNamespace(
        agent=SimpleNamespace(
            robot=robot, voxel_map=SimpleNamespace(observations=[SimpleNamespace(depth=np.ones((20, 20)))])
        ),
        _captured_views={1: CapturedView(1, 1, np.zeros((20, 20, 3), dtype=np.uint8), None)},
        _append_trace=Mock(),
    )
    cap = {"ok": True, "obs_id": 1}
    assert recover_exploration_view(ex, cap) == cap
    robot.head_to.assert_not_called()


@pytest.fixture(autouse=True)
def fresh_head_frames(monkeypatch):
    def receive(robot, timeout):
        robot._seq_id += 1

    monkeypatch.setattr("emet.controller.dynamem.look.wait_post_motion_obs", receive)


def arrival_executor():
    robot = SimpleNamespace(_seq_id=1, angles=(0.0, 0.0))
    robot.head_to = Mock(side_effect=lambda pan, tilt, **kw: setattr(robot, "angles", (pan, tilt)))
    robot.get_pan_tilt = lambda: robot.angles
    intrinsics = np.array([[10.0, 0.0, 10.0], [0.0, 10.0, 10.0], [0.0, 0.0, 1.0]])
    view = CapturedView(1, 1, np.zeros((20, 20, 3), dtype=np.uint8), np.eye(4), intrinsics)
    return SimpleNamespace(
        agent=SimpleNamespace(robot=robot),
        _captured_views={1: view},
        _append_trace=Mock(),
        _tool_capture_and_update=Mock(return_value={"ok": True, "obs_id": 1}),
    )


def test_offscreen_arrival_aims_down_and_rejects_stale_capture():
    ex = arrival_executor()
    result = aim_arrival_view(ex, {"ok": True, "obs_id": 1}, [0, 2, 1])
    assert result["status"] == "TARGET_OUTSIDE_VIEW"
    assert result["ok"] is False
    pan, tilt = ex.agent.robot.head_to.call_args.args
    assert pan == 0 and -np.pi / 4 <= tilt < 0


def test_inframe_arrival_does_not_move_or_claim_identity():
    ex = arrival_executor()
    cap = {"ok": True, "obs_id": 1}
    assert aim_arrival_view(ex, cap, [0, 0, 1]) == cap
    ex.agent.robot.head_to.assert_not_called()


def test_aiming_accepts_only_new_inframe_capture():
    ex = arrival_executor()
    old = ex._captured_views[1]
    pose = np.eye(4)
    pose[1, 3] = 2
    ex._captured_views[2] = CapturedView(2, 2, old.rgb, pose, old.camera_K)
    ex._tool_capture_and_update.return_value = {"ok": True, "obs_id": 2}
    result = aim_arrival_view(ex, {"ok": True, "obs_id": 1}, [0, 2, 1])
    assert result == {"ok": True, "obs_id": 2}
    assert ex.agent.robot.head_to.call_count == 1


def test_behind_camera_does_not_trigger_unbounded_turn():
    ex = arrival_executor()
    assert not aim_arrival_view(ex, {"ok": True, "obs_id": 1}, [0, 0, -1])["ok"]
    ex.agent.robot.head_to.assert_called_once_with(0.0, 0.0, blocking=True)


def test_behind_downward_camera_recovers_with_fresh_forward_view():
    ex = arrival_executor()
    old = ex._captured_views[1]
    pose = np.diag([-1.0, 1.0, -1.0, 1.0])
    ex._captured_views[2] = CapturedView(2, 2, old.rgb, pose, old.camera_K)
    ex._tool_capture_and_update.return_value = {"ok": True, "obs_id": 2}
    assert aim_arrival_view(ex, {"ok": True, "obs_id": 1}, [0, 0, -1])["ok"]
    ex.agent.robot.head_to.assert_called_once_with(0.0, 0.0, blocking=True)


def test_new_memory_id_cannot_override_stale_camera_sequence(monkeypatch):
    ex = arrival_executor()
    monkeypatch.setattr("emet.controller.dynamem.look.wait_post_motion_obs", lambda *a, **kw: None)
    ex._tool_capture_and_update.return_value = {"ok": True, "obs_id": 2}
    result = aim_arrival_view(ex, {"ok": True, "obs_id": 1}, [0, 2, 1])
    assert result["reason"] == "stale_observation"
    ex._tool_capture_and_update.assert_not_called()


def test_clipped_head_does_not_authorize_capture(monkeypatch):
    ex = arrival_executor()
    ex.agent.robot.head_to.side_effect = None
    times = iter([0, 6])
    monkeypatch.setattr("emet.memory.graph_eqa.agentic.view_quality.time.monotonic", lambda: next(times))
    result = aim_arrival_view(ex, {"ok": True, "obs_id": 1}, [0, 2, 1])
    assert result["reason"] == "head_pose_unconfirmed"
    ex._tool_capture_and_update.assert_not_called()


def test_forward_reset_and_correction_share_two_command_budget():
    ex = arrival_executor()
    old = ex._captured_views[1]
    forward = np.diag([-1.0, 1.0, -1.0, 1.0])
    aimed = forward.copy()
    aimed[1, 3] = 2
    ex._captured_views[2] = CapturedView(2, 2, old.rgb, forward, old.camera_K)
    ex._captured_views[3] = CapturedView(3, 3, old.rgb, aimed, old.camera_K)
    ex._tool_capture_and_update.side_effect = [{"ok": True, "obs_id": 2}, {"ok": True, "obs_id": 3}]
    result = aim_arrival_view(ex, {"ok": True, "obs_id": 1}, [0, 2, -1])
    assert result["ok"] and result["obs_id"] == 3
    assert ex.agent.robot.head_to.call_count == 2


def test_physical_inspection_rejects_missing_camera_geometry():
    ex = arrival_executor()
    old = ex._captured_views[1]
    ex._captured_views[1] = CapturedView(1, 1, old.rgb, None, None)
    result = aim_arrival_view(ex, {"ok": True, "obs_id": 1}, [0, 0, 1])
    assert not result["ok"] and result["reason"] == "missing_geometry"
    ex.agent.robot.head_to.assert_not_called()


def test_head_limit_is_exposed_before_sending_impossible_command():
    from emet.robots.head_capability import STRETCH_LEGACY_HEAD

    ex = arrival_executor()
    ex.agent.robot.get_head_capability = lambda: STRETCH_LEGACY_HEAD
    result = aim_arrival_view(ex, {"ok": True, "obs_id": 1}, [0, -2, 1])
    assert result["reason"] == "HEAD_LIMIT"
    assert result["look_at"]["requested_pan_tilt"][1] > 0
    assert result["look_at"]["limits"]["tilt"][1] == 0
    assert result["look_at"]["recovery"] == "choose_another_collision_checked_viewpoint"
    ex.agent.robot.head_to.assert_not_called()
