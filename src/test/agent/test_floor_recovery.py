# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Coverage recovery remains observation-only until fresh mapping succeeds."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from emet.agent.tools import get_tools
from emet.controller.dynamem import look


@pytest.mark.parametrize("in_frame", [False, True])
def test_targeted_floor_aim_is_bounded_and_projection_does_not_clear_map(monkeypatch, in_frame):
    pose = np.eye(4)
    pose[2, 3] = -1
    obs = SimpleNamespace(
        rgb=np.zeros((100, 100, 3)),
        depth=np.full((100, 100), 0.4),
        camera_pose=pose,
        camera_K=np.array(
            [
                [100, 0, 50],
                [0, 100, 50],
                [0, 0, 1],
            ]
        ),
    )
    robot = SimpleNamespace(
        _seq_id=1,
        head_to=Mock(),
        get_pan_tilt=lambda: (0, -1),
        get_observation=lambda: obs,
    )
    detail = {
        "reason": "unobserved_footprint",
        "unknown_footprint_cells": 1,
        "unknown_cell_offsets_frame": "world_xy_from_checked_pose",
        "unknown_cell_offsets_m": [[0, 1]],
    }
    agent = SimpleNamespace(
        robot=robot,
        space=SimpleNamespace(is_valid=lambda p: False, last_validity=detail),
        _last_nav_plan={"footprint": {"checked_pose": [0, 0, 0]}},
    )

    def capture(agent, pan_rad, tilt_rad):
        assert -1 <= pan_rad <= 1
        assert -1.4 <= tilt_rad <= -0.7
        if in_frame:
            obs.camera_pose = pose.copy()
            obs.camera_pose[1, 3] = 1
        return {"ok": True, "observation": {"footprint_after": {"pose_valid": False}}}

    capture_mock = Mock(side_effect=capture)
    monkeypatch.setattr(look, "observe_floor", capture_mock)
    result = look.observe_blocked_floor(agent)
    assert capture_mock.call_count == (1 if in_frame else 2)
    assert result["observation"]["footprint_after"]["pose_valid"] is False
    attempt = result["observation"]["targeted_floor"]["attempts"][-1]
    assert attempt["after"]["target_in_frame"] is in_frame
    if in_frame:
        assert attempt["after"]["measured_depth_m"] == 0.4
        assert attempt["after"]["reference_optical_depth_m"] == 1.0
    assert "not clearance" in result["note"]
    robot.head_to.assert_not_called()  # Only the existing capture primitive commands head motion.


def test_final_tool_schema_exposes_floor_tilt_and_feedback():
    tool = next(tool for tool in get_tools({}) if tool.name == "observe_floor")
    tilt = tool.parameters["properties"]["tilt_rad"]
    assert (tilt["minimum"], tilt["maximum"]) == (-1.4, -0.7)
    assert "footprint_before/after" in tool.description


@pytest.mark.parametrize("pan_rad", [None, -0.8, 0.8])
@pytest.mark.parametrize("tilt_rad", [-1.0, -1.4])
@pytest.mark.parametrize(
    "remaining,reason,recovery_status",
    [
        (1, "unobserved_footprint", "unknown_footprint_reduced"),
        (4, "unobserved_footprint", "unknown_footprint_not_reduced"),
        (0, "valid", "checked_pose_valid_replan_required"),
        (0, "occupied_footprint", "checked_pose_still_invalid"),
    ],
)
@pytest.mark.parametrize(
    "failure", [None, "delayed", "post_arrival_stale", "stale", "pose", "depth", "map", "unsupported", "sequence"]
)
def test_floor_observation_requires_fresh_measured_capture(
    monkeypatch, tmp_path, failure, pan_rad, tilt_rad, remaining, reason, recovery_status
):
    import itertools

    clock = itertools.count(step=1.0)
    monkeypatch.setattr(look.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(look.time, "sleep", lambda _: None)
    robot = SimpleNamespace(_seq_id=1, head_to=Mock(return_value=True))
    robot.get_base_pose = lambda: np.array([1, 2, 0.3])
    expected_pan = 0.2 if pan_rad is None else pan_rad
    positions = itertools.chain(
        [(0.2, -0.5)],
        [(expected_pan, -0.5)] * (3 if failure == "delayed" else 0),
        itertools.repeat((expected_pan, 0 if failure == "pose" else tilt_rad)),
    )
    robot.get_pan_tilt = Mock(side_effect=lambda: next(positions))
    robot.get_observation = Mock(
        return_value=SimpleNamespace(
            rgb=np.ones((2, 2, 3)),
            depth=None if failure == "depth" else np.ones((2, 2)),
            camera_K=np.eye(3),
            camera_pose=np.eye(4),
            get_xyz_in_world_frame=lambda: np.ones((2, 2, 3)),
        )
    )
    monkeypatch.setenv("EMET_EQA_EPISODE_DIR", str(tmp_path))
    agent = SimpleNamespace(
        robot=robot,
        _planning_base_xyt=lambda pose: pose + [10, 20, 0],
        voxel_map=SimpleNamespace(
            observations=[],
            get_2d_map=lambda: (torch.zeros((2, 2), dtype=torch.bool), torch.ones((2, 2), dtype=torch.bool)),
        ),
    )
    agent.update = Mock(side_effect=lambda **kw: None if failure == "map" else agent.voxel_map.observations.append(1))
    agent._last_nav_plan = {"footprint": {"checked_pose": [0, 0, 0]}}
    agent.space = SimpleNamespace(last_validity={})

    def check_footprint(pose):
        agent.space.last_validity = {
            "reason": reason if agent.voxel_map.observations else "unobserved_footprint",
            "unknown_footprint_cells": remaining if agent.voxel_map.observations else 4,
        }
        return bool(agent.voxel_map.observations) and reason == "valid"

    agent.space.is_valid = check_footprint

    def receive(robot, timeout):
        if failure != "stale" and not (failure == "post_arrival_stale" and robot._seq_id > 1):
            robot._seq_id += 1

    monkeypatch.setattr(look, "wait_post_motion_obs", receive)
    if failure == "unsupported":
        robot.head_to = None
    if failure == "sequence":
        robot._seq_id = None
    result = look.observe_floor(agent, pan_rad=pan_rad, tilt_rad=tilt_rad)
    assert result["ok"] is (failure in (None, "delayed"))
    if failure not in (None, "delayed", "map"):
        agent.update.assert_not_called()
    if failure in (None, "delayed"):
        artifact = next((tmp_path / "navigation").glob("*.npz"))
        with np.load(artifact) as saved:
            np.testing.assert_allclose(saved["base_pose"], [11, 22, 0.3])
        robot.head_to.assert_called_once_with(expected_pan, tilt_rad, blocking=True)
        agent.update.assert_called_once_with(full_perception=True)
        feedback = result["observation"]
        assert feedback["footprint_before"]["unknown_cells"] == 4
        assert feedback["footprint_after"]["unknown_cells"] == remaining
        assert feedback["footprint_after"]["pose_valid"] is (reason == "valid")
        assert feedback["recovery_status"] == recovery_status
        assert feedback["replan_required"] is True
        from emet.agent.tool_outcome import ToolOutcome

        rendered = ToolOutcome.from_eqa_dict("observe_floor", result).render()
        assert recovery_status in rendered
        assert '"measured_head_pan_tilt_rad"' in rendered


@pytest.mark.parametrize("pan_rad", [float("nan"), float("inf"), -1.01, 1.01, True, "left"])
def test_floor_observation_rejects_invalid_pan_before_motion(pan_rad):
    assert look.observe_floor(SimpleNamespace(), pan_rad=pan_rad) == {"ok": False, "status": "invalid_head_pan"}


@pytest.mark.parametrize("tilt_rad", [float("nan"), float("inf"), -1.41, -0.69, True, "down"])
def test_floor_observation_rejects_invalid_tilt_before_motion(tilt_rad):
    assert look.observe_floor(SimpleNamespace(), tilt_rad=tilt_rad) == {"ok": False, "status": "invalid_head_tilt"}
