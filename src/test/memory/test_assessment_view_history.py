# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from emet.memory.graph_eqa.agentic.view_history import select_assessment_history


def executor():
    frames = [
        SimpleNamespace(
            rgb=np.full((4, 4, 3), i, dtype=np.uint8),
            camera_pose=np.eye(4),
            info={"image_timing": {"head_cam_left": {"timestamp_ns": (i + 1) * 10**9, "clock_domain": "sim"}}},
        )
        for i in range(5)
    ]
    return SimpleNamespace(
        agent=SimpleNamespace(
            robot=SimpleNamespace(),
            parameters={"eqa": {"view_selection": {"policy": "pose"}}},
            voxel_map=SimpleNamespace(observations=frames),
        ),
        _append_trace=Mock(),
    )


def test_history_uses_acquisition_metadata_and_labels_old_views_without_preturn_claim():
    ex = executor()
    result = select_assessment_history(ex, ex.agent.voxel_map.observations[-1].rgb)
    assert len(result.views) == 4
    assert result.views[0].frame.view_id == "5"
    assert result.views[1].age_s == 1
    assert [v.role for v in result.views] == ["current", "context", "context", "context"]
    assert ex._append_trace.call_args.args[0]["event"] == "view_selection"


def test_unknown_current_acquisition_or_mismatched_pixels_falls_back_explicitly():
    ex = executor()
    current = ex.agent.voxel_map.observations[-1]
    assert select_assessment_history(ex, np.full_like(current.rgb, 99)) is None
    current.info = {}
    assert select_assessment_history(ex, current.rgb) is None


def test_budget_and_clock_domain_are_respected():
    ex = executor()
    ex.agent.parameters["eqa_vl"] = {"eqa_max_images": 2}
    ex.agent.voxel_map.observations[-2].info["image_timing"]["head_cam_left"]["clock_domain"] = "other"
    result = select_assessment_history(ex, ex.agent.voxel_map.observations[-1].rgb)
    assert len(result.views) == 2
    assert result.excluded["4"] == "different_scope"


def test_selection_disabled_by_default(monkeypatch):
    monkeypatch.delenv("EMET_EQA_VIEW_SELECTION_POLICY", raising=False)
    ex = executor()
    ex.agent.parameters = {}
    assert select_assessment_history(ex, ex.agent.voxel_map.observations[-1].rgb) is None


def test_habitat_voxel_axis_permutation_is_valid_selection_geometry():
    from emet.habitat.coordinates import habitat_observation_camera_pose

    ex = executor()
    for i, frame in enumerate(ex.agent.voxel_map.observations):
        body = np.eye(4)
        body[0, 3] = i
        frame.camera_pose = habitat_observation_camera_pose(body, floor_y=0, sensor_height=1.5)
        assert np.linalg.det(frame.camera_pose[:3, :3]) < 0
    result = select_assessment_history(ex, ex.agent.voxel_map.observations[-1].rgb)
    assert len(result.views) == 4
    assert result.metrics["mean_translation_m"] > 0


def test_real_preturn_capture_is_pinned_but_stale_one_is_not():
    ex = executor()
    ex.agent.robot.pre_turn_observation = SimpleNamespace(
        rgb=np.full((4, 4, 3), 99, dtype=np.uint8),
        camera_pose=np.eye(4),
        image_timing={"head_cam_left": {"timestamp_ns": 4500000000, "clock_domain": "sim"}},
    )
    rgb = ex.agent.voxel_map.observations[-1].rgb
    result = select_assessment_history(ex, rgb)
    assert result.views[1].role == "pre_turn"
    assert result.views[1].age_s == 0.5
    ex.agent.robot.pre_turn_observation.image_timing["head_cam_left"]["timestamp_ns"] = 6000000000
    result = select_assessment_history(ex, rgb)
    assert all(v.role != "pre_turn" for v in result.views)
