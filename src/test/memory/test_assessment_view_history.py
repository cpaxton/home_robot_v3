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


def test_evidence_context_profile_survives_robot_runtime_settings(monkeypatch):
    from pathlib import Path

    from emet.core.parameters import get_parameters
    from emet.memory.view_selection import SelectionConfig

    profile = Path(__file__).resolve().parents[3] / "configs/emet/evidence_context.yaml"
    monkeypatch.setenv("EMET_CONFIG", str(profile))
    parameters = get_parameters("dynav_config.yaml", robot="stretch")
    eqa = parameters.get("eqa")
    assert eqa["retain_approach_views"] is True
    selection = SelectionConfig(**eqa["view_selection"])
    assert selection.policy == "pose" and selection.spatial_labels
    assert selection.max_images == 4
    assert eqa.get("agentic_single_view_confirm", True) is True


def test_model_final_profile_changes_only_answer_authority(monkeypatch):
    from pathlib import Path

    from emet.core.parameters import get_parameters
    from emet.memory.graph_eqa.agentic.config import resolve_agentic_decision_policy

    root = Path(__file__).resolve().parents[3]
    configurations = []
    for name in ("evidence_context", "model_final"):
        monkeypatch.setenv("EMET_CONFIG", str(root / f"configs/emet/{name}.yaml"))
        configurations.append(dict(get_parameters("dynav_config.yaml", robot="stretch").get("eqa")))
    before, after = configurations
    assert after.pop("agentic_decision_policy") == "model_final"
    before.pop("agentic_decision_policy", None)
    assert before == after
    assert resolve_agentic_decision_policy("model_final") == "model_final"


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


def test_unmapped_current_capture_anchors_preturn_age():
    ex = executor()
    current = SimpleNamespace(
        rgb=np.full((4, 4, 3), 77, dtype=np.uint8),
        camera_pose=np.eye(4),
        image_timing={"head_cam_left": {"timestamp_ns": 7_000_000_000, "clock_domain": "sim"}},
    )
    ex.agent.robot.pre_turn_observation = SimpleNamespace(
        rgb=np.full((4, 4, 3), 66, dtype=np.uint8),
        camera_pose=np.eye(4),
        image_timing={"head_cam_left": {"timestamp_ns": 6_000_000_000, "clock_domain": "sim"}},
    )
    result = select_assessment_history(ex, current.rgb, current_observation=current)
    assert result.views[0].frame.view_id == "assessment:7000000000"
    assert result.views[1].role == "pre_turn"
    assert result.views[1].age_s == 1


def test_map_reset_rejects_prior_epoch_frames_even_with_same_clock_and_recent_time():
    ex = executor()
    vm = ex.agent.voxel_map
    vm.observation_scope_id = "old-world"
    for frame in vm.observations:
        frame.info["observation_scope_id"] = "old-world"
    old_scope = select_assessment_history(ex, vm.observations[-1].rgb).views[0].frame.scope_id
    vm.observation_scope_id = "new-world"
    # Simulate reintroducing a recently captured old frame into the reused map.
    newest = SimpleNamespace(
        rgb=np.full((4, 4, 3), 88, dtype=np.uint8),
        camera_pose=np.eye(4),
        info={
            "observation_scope_id": "new-world",
            "image_timing": {"head_cam_left": {"timestamp_ns": 6_000_000_000, "clock_domain": "sim"}},
        },
    )
    vm.observations.append(newest)
    result = select_assessment_history(ex, newest.rgb)
    assert len(result.views) == 1
    assert result.views[0].frame.scope_id != old_scope


def test_approach_capture_before_new_scope_history_is_not_relabelled_into_new_world():
    ex = executor()
    vm = ex.agent.voxel_map
    vm.observation_scope_id = "new-world"
    vm.observations = vm.observations[-1:]
    vm.observations[0].info["observation_scope_id"] = "new-world"
    ex.agent.robot.navigation_view_history = SimpleNamespace(
        views=[
            SimpleNamespace(
                view_id="approach:old",
                timestamp_ns=4_000_000_000,
                clock_domain="sim",
                rgb=np.full((4, 4, 3), 88, dtype=np.uint8),
                camera_pose=np.eye(4),
            )
        ]
    )
    result = select_assessment_history(ex, vm.observations[-1].rgb)
    assert "approach:old" not in [view.frame.view_id for view in result.views]


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


def quality_executor():
    ex = executor()
    ex.agent.parameters["eqa"]["view_selection"]["policy"] = "quality_coverage"
    ex._inspection_target_by_obs = {7: np.array([0.0, 0.0, 1.0])}
    for frame in ex.agent.voxel_map.observations:
        frame.depth = np.ones((4, 4))
        frame.camera_K = np.eye(3)
    return ex


def test_live_quality_uses_registered_depth_and_caches_per_retained_frame():
    ex = quality_executor()
    rgb = ex.agent.voxel_map.observations[-1].rgb
    result = select_assessment_history(ex, rgb, target_obs_id=7)
    assert result.config.policy == "quality_coverage"
    assert result.views[0].frame.cell_quality
    cache = dict(ex._view_surface_cache)
    select_assessment_history(ex, rgb, target_obs_id=7)
    assert all(ex._view_surface_cache[k] is v for k, v in cache.items())


def test_quality_missing_geometry_falls_back_to_pose_explicitly():
    ex = quality_executor()
    rgb = ex.agent.voxel_map.observations[-1].rgb
    ex._inspection_target_by_obs[7] = None
    assert select_assessment_history(ex, rgb, target_obs_id=7).config.policy == "pose"
    assert ex._append_trace.call_args_list[-2].args[0]["reason"] == "missing_target_geometry"
    ex._inspection_target_by_obs[7] = np.zeros(3)
    ex.agent.voxel_map.observations[-1].depth = np.ones((2, 2))
    assert select_assessment_history(ex, rgb, target_obs_id=7).config.policy == "pose"
    assert ex._append_trace.call_args_list[-2].args[0]["reason"] == "missing_or_unregistered_depth"


def test_history_reset_replaces_world_scope_and_surface_cache():
    ex = quality_executor()
    result = select_assessment_history(ex, ex.agent.voxel_map.observations[-1].rgb, target_obs_id=7)
    old_scope = result.views[0].frame.scope_id
    replacement = quality_executor()
    ex.agent.voxel_map.observations = replacement.agent.voxel_map.observations
    result = select_assessment_history(ex, ex.agent.voxel_map.observations[-1].rgb, target_obs_id=7)
    assert result.views[0].frame.scope_id != old_scope
    assert ex._view_surface_cache["1"][0] is replacement.agent.voxel_map.observations[0]


def test_all_invalid_depth_does_not_fabricate_coverage():
    ex = quality_executor()
    for frame in ex.agent.voxel_map.observations:
        frame.depth[:] = np.nan
    result = select_assessment_history(ex, ex.agent.voxel_map.observations[-1].rgb, target_obs_id=7)
    assert result.config.policy == "pose"
    assert ex._append_trace.call_args_list[-2].args[0]["reason"] == "no_observed_target_local_cells"


def test_approach_views_join_pool_without_becoming_current_or_graph_objects():
    from emet.memory.approach_views import ApproachViewHistory

    ex = executor()
    buffer = ex.agent.robot.navigation_view_history = ApproachViewHistory()
    pose = np.eye(4)
    pose[0, 3] = 2
    obs = SimpleNamespace(
        rgb=np.full((4, 4, 3), 77, dtype=np.uint8),
        camera_pose=pose,
        image_timing={"head_cam_left": {"timestamp_ns": 4500000000, "clock_domain": "sim"}},
    )
    assert buffer.add(obs)
    result = select_assessment_history(ex, ex.agent.voxel_map.observations[-1].rgb)
    assert result.views[0].frame.view_id == "5"
    assert "approach:1" in [v.frame.view_id for v in result.views]
    assert len(ex.agent.voxel_map.observations) == 5


def test_exact_selected_packet_is_saved_in_order(tmp_path):
    import hashlib
    import json

    from PIL import Image

    ex = executor()
    ex.agent._episode_debug_dir = str(tmp_path)
    result = select_assessment_history(ex, ex.agent.voxel_map.observations[-1].rgb)
    directory = tmp_path / "selected_views" / "0001"
    manifest = json.loads((directory / "manifest.json").read_text())
    for selected, row in zip(result.views, manifest["views"], strict=True):
        pixels = np.asarray(Image.open(directory / row["image"]))
        assert np.array_equal(pixels, selected.frame.rgb)
        assert hashlib.sha256(pixels.tobytes()).hexdigest() == row["rgb_sha256"]
