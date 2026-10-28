# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

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


def test_model_final_profile_changes_only_answer_authority(monkeypatch):
    from pathlib import Path

    from emet.core.parameters import get_parameters
    from emet.memory.graph_eqa.agentic.config import resolve_agentic_decision_policy

    root = Path(__file__).resolve().parents[3]
    configurations = []
    for name in ("default", "model_final"):
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


@pytest.mark.parametrize("stamp", [True, False])
def test_history_rejects_boolean_acquisition_timestamp(stamp):
    ex = executor()
    history = ex.agent.voxel_map.observations
    ex.agent.voxel_map.observations = [history[0], history[-1]]
    ex.agent.voxel_map.observations[0].info["image_timing"]["head_cam_left"]["timestamp_ns"] = stamp
    result = select_assessment_history(ex, ex.agent.voxel_map.observations[-1].rgb)
    assert all(view.frame.view_id != "1" for view in result.views)


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


def test_default_handoff_pins_only_current_and_fresh_preturn(monkeypatch):
    monkeypatch.delenv("EMET_EQA_VIEW_SELECTION_POLICY", raising=False)
    ex = executor()
    ex.agent.parameters = {}
    ex.agent.robot.pre_turn_observation = SimpleNamespace(
        rgb=np.full((4, 4, 3), 99, dtype=np.uint8),
        camera_pose=np.eye(4),
        image_timing={"head_cam_left": {"timestamp_ns": 4500000000, "clock_domain": "sim"}},
    )
    result = select_assessment_history(ex, ex.agent.voxel_map.observations[-1].rgb)
    assert [v.role for v in result.views] == ["current", "pre_turn"]
    assert result.views[1].age_s == 0.5
    np.testing.assert_array_equal(result.views[1].frame.rgb, 99)
    assert ex._append_trace.call_args.args[0]["event"] == "assessment_handoff"


@pytest.mark.parametrize("problem", ["old_epoch", "clock", "future", "same_time", "old", "missing"])
def test_default_handoff_rejects_unqualified_preturn(monkeypatch, problem):
    monkeypatch.delenv("EMET_EQA_VIEW_SELECTION_POLICY", raising=False)
    ex = executor()
    ex.agent.parameters = {}
    vm = ex.agent.voxel_map
    vm.observation_scope_id = "episode"
    for frame in vm.observations:
        frame.info["observation_scope_id"] = "episode"
    before = SimpleNamespace(
        rgb=np.full((4, 4, 3), 99, dtype=np.uint8),
        camera_pose=np.eye(4),
        observation_scope_id="old" if problem == "old_epoch" else "episode",
        image_timing={"head_cam_left": {"timestamp_ns": 4500000000, "clock_domain": "sim"}},
    )
    timing = before.image_timing["head_cam_left"]
    if problem == "clock":
        timing["clock_domain"] = "other"
    if problem in ("future", "same_time", "old"):
        timing["timestamp_ns"] = {"future": 6e9, "same_time": 5e9, "old": 0.5e9}[problem]
    if problem == "missing":
        before.image_timing = {}
    ex.agent.robot.pre_turn_observation = before
    assert select_assessment_history(ex, vm.observations[-1].rgb) is None


def test_capture_capability_does_not_require_ranking_config(monkeypatch):
    from unittest.mock import MagicMock

    from emet.memory.graph_eqa import AgenticEQAExecutor

    monkeypatch.delenv("EMET_EQA_VIEW_SELECTION_POLICY", raising=False)
    agent = MagicMock()
    agent.graph_memory = None
    agent.parameters = {}
    agent.voxel_map = SimpleNamespace(observations=[], encoder=None)
    agent.robot = SimpleNamespace(retain_navigation_views=False, pre_turn_observation=object())
    AgenticEQAExecutor(agent, "Where is the bowl?", router=False)
    assert agent.robot.retain_navigation_views is True
    assert agent.robot.pre_turn_observation is None


@pytest.mark.parametrize("budget,duplicate,expected", [(1, False, 1), (4, True, 1), (4, False, 2)])
def test_default_handoff_respects_budget_and_deduplicates(monkeypatch, budget, duplicate, expected):
    monkeypatch.delenv("EMET_EQA_VIEW_SELECTION_POLICY", raising=False)
    ex = executor()
    ex.agent.parameters = {"eqa_vl": {"eqa_max_images": budget}}
    rgb = ex.agent.voxel_map.observations[-1].rgb
    ex.agent.robot.pre_turn_observation = SimpleNamespace(
        rgb=rgb.copy() if duplicate else np.full_like(rgb, 99),
        camera_pose=np.eye(4),
        image_timing={"head_cam_left": {"timestamp_ns": 4500000000, "clock_domain": "sim"}},
    )
    result = select_assessment_history(ex, rgb)
    assert len(result.views) == expected
    assert result.views[0].role == "current"
    assert all(v.role in ("current", "pre_turn") for v in result.views)


def test_default_handoff_uses_before_turn_label_at_model_boundary(monkeypatch):
    from emet.eval.agentic_vlm_assess import assess_view_with_vlm

    monkeypatch.delenv("EMET_EQA_VIEW_SELECTION_POLICY", raising=False)
    ex = executor()
    ex.agent.parameters = {}
    rgb = ex.agent.voxel_map.observations[-1].rgb
    ex.agent.robot.pre_turn_observation = SimpleNamespace(
        rgb=np.full_like(rgb, 99),
        camera_pose=np.eye(4),
        image_timing={"head_cam_left": {"timestamp_ns": 4500000000, "clock_domain": "sim"}},
    )
    selected = select_assessment_history(ex, rgb)
    client = Mock(return_value='{"present":false,"answerable":false}', _vl=None)
    result = assess_view_with_vlm(client, question="Where is the bowl?", rgb=rgb, selected_views=selected)
    assert result.image_count == 2
    assert "arrival before final turn" in result.prompt
    assert "pre-approach" not in result.prompt
    assert "Judge present from Image 1 only" in result.prompt
    assert not result.present and not result.answerable


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
