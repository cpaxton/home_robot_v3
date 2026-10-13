# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
import json
from dataclasses import replace

import numpy as np
import pytest

from emet.memory.view_coverage import (
    CoverageGrid,
    CoverageObjective,
    depth_surface_cells,
    depth_surface_quality,
    gaussian_coverage_objective,
)
from emet.memory.view_selection import SelectionConfig, ViewFrame, select_views


def frame(i, cells):
    return ViewFrame(str(i), "map", float(i), np.full((4, 4, 3), i, np.uint8), visible_cells=cells)


def select(frames, objective, **kwargs):
    return select_views(
        frames,
        current_id="3",
        coverage_objective=objective,
        config=SelectionConfig(policy="weighted_coverage", max_images=3),
        **kwargs,
    )


def test_hill_and_weighted_union():
    objective = gaussian_coverage_objective(
        {"near": (0, 0), "far": (3, 0)}, target_xy=(0, 0), sigma_m=1, scope_id="map", visibility_source="depth"
    )
    assert objective.weight(["near", "near"]) == 1
    assert 0 < objective.cell_weights["far"] < 0.02
    assert objective.marginal({"near", "far"}, {"near"}) == objective.cell_weights["far"]


def test_quality_credits_best_observation_not_sum_and_rewards_upgrades():
    objective = CoverageObjective("map", {"a": 1, "b": 1}, "depth")
    frames = [
        replace(frame(1, {"a"}), cell_quality={"a": 1.0}),
        replace(frame(2, {"b"}), cell_quality={"b": 0.5}),
        replace(frame(3, {"a"}), cell_quality={"a": 0.25}),
    ]
    result = select_views(
        frames,
        current_id="3",
        coverage_objective=objective,
        config=SelectionConfig(policy="quality_coverage", max_images=3),
    )
    assert [v.frame.view_id for v in result.views] == ["3", "1", "2"]
    assert [v.features["quality_coverage"] for v in result.views] == [0.25, 0.75, 0.5]
    assert result.metrics["weighted_coverage"]["selected_quality_weight"] == 1.5
    assert objective.quality_marginal({"a": 0.5}, {"a": 1.0}) == 0


def test_all_one_quality_reproduces_binary_weighted_coverage():
    objective = CoverageObjective("map", {"a": 1, "b": 2, "c": 3}, "depth")
    frames = [
        replace(frame(i, cells), cell_quality=dict.fromkeys(cells, 1.0))
        for i, cells in [(1, {"c"}), (2, {"a", "b"}), (3, {"a"})]
    ]
    kwargs = {"current_id": "3", "coverage_objective": objective}
    a = select_views(frames, config=SelectionConfig(policy="weighted_coverage"), **kwargs)
    b = select_views(frames, config=SelectionConfig(policy="quality_coverage"), **kwargs)
    assert [v.frame.view_id for v in a.views] == [v.frame.view_id for v in b.views]


def test_quality_metadata_does_not_change_legacy_pose_selection():
    frames = [frame(i, {"a"}) for i in (1, 2, 3)]
    plain = select_views(frames, current_id="3", config=SelectionConfig(policy="pose", max_images=2))
    enriched = select_views(
        [replace(f, cell_quality={"a": i / 3}) for i, f in enumerate(frames, 1)],
        current_id="3",
        config=SelectionConfig(policy="pose", max_images=2),
    )
    assert [v.frame.view_id for v in plain.views] == [v.frame.view_id for v in enriched.views]
    assert [v.score for v in plain.views] == [v.score for v in enriched.views]


def test_quality_input_validation():
    with pytest.raises(ValueError, match="exactly"):
        replace(frame(1, {"a"}), cell_quality={"b": 1.0})
    with pytest.raises(ValueError, match="finite"):
        replace(frame(1, {"a"}), cell_quality={"a": float("nan")})
    with pytest.raises(ValueError, match="cell_quality"):
        select_views(
            [frame(3, {"a"})],
            current_id="3",
            coverage_objective=CoverageObjective("map", {"a": 1}, "depth"),
            config=SelectionConfig(policy="quality_coverage"),
        )


def test_pixel_support_quality_preserves_context_and_saturates():
    quality = depth_surface_quality(
        np.ones((2, 2)),
        np.eye(3),
        np.eye(4),
        grid=CoverageGrid(10),
        max_depth_m=4,
        pixel_stride=1,
        reference_pixels=8,
        context_credit=0.5,
    )
    assert quality == {"0,0": 0.75}
    full = depth_surface_quality(
        np.ones((2, 2)),
        np.eye(3),
        np.eye(4),
        grid=CoverageGrid(10),
        max_depth_m=4,
        pixel_stride=1,
        reference_pixels=2,
        context_credit=0.5,
    )
    assert full == {"0,0": 1.0}


def test_pins_current_then_values_local_novelty_not_total_cells():
    objective = CoverageObjective("map", {"a": 10, "b": 8, "c": 1, "d": 1}, "depth")
    frames = [frame(1, {"b"}), frame(2, {"a", "c", "d"}), frame(3, {"a"})]
    result = select(frames, objective)
    assert [v.frame.view_id for v in result.views] == ["3", "1", "2"]
    assert [v.features["weighted_coverage"] for v in result.views] == [10, 8, 2]
    assert result.metrics["weighted_coverage"]["selected_weight"] == 20
    assert result.metrics["weighted_coverage"]["candidate_fraction"] == 1
    assert result.metrics["weighted_coverage"]["selected_cells"] == ["a", "b", "c", "d"]
    json.dumps(result.to_dict(), allow_nan=False)
    assert [v.frame.view_id for v in select(frames[::-1], objective).views] == ["3", "1", "2"]


def test_no_credit_for_duplicate_or_outside_hill_cells():
    objective = CoverageObjective("map", {"a": 1}, "depth")
    result = select([frame(1, {"a"}), frame(2, {"outside"}), frame(3, {"a"})], objective)
    assert len(result.views) == 1
    assert set(result.excluded.values()) == {"no_marginal_coverage"}


def test_missing_visibility_and_wrong_map_are_errors_not_recency_fallback():
    objective = CoverageObjective("map", {"a": 1}, "depth")
    with pytest.raises(ValueError, match="visible_cells"):
        select([frame(3, None)], objective)
    with pytest.raises(ValueError, match="scope"):
        select([frame(3, {"a"})], CoverageObjective("other", {"a": 1}, "depth"))
    with pytest.raises(ValueError, match="explicit"):
        select([frame(3, {"a"})], None)
    result = select([frame(3, set())], objective)
    assert result.metrics["weighted_coverage"]["candidate_fraction"] is None


@pytest.mark.parametrize("weight", [-1, float("nan"), float("inf"), 0])
def test_bad_weights_rejected(weight):
    with pytest.raises(ValueError):
        CoverageObjective("map", {"a": weight}, "depth")


def test_objective_does_not_alias_mutable_caller_weights():
    weights = {"a": 1}
    objective = CoverageObjective("map", weights, "depth")
    weights["a"] = 3
    assert objective.weight(["a"]) == 1
    with pytest.raises(TypeError):
        objective.cell_weights["a"] = 5


def test_depth_surface_grid_deduplicates_xy_and_does_not_fill_rays():
    grid = CoverageGrid(0.5)
    cells = grid.observed_cells(np.array([[2.1, 0.1], [2.2, 0.2], [-0.1, 0.1], [np.nan, 0]]))
    assert cells == {"4,0", "-1,0"}
    assert "0,0" not in cells  # No credit for the space along the camera ray.
    assert grid.centers(cells) == {"4,0": (2.25, 0.25), "-1,0": (-0.25, 0.25)}
    assert grid.observed_cells(np.empty((0, 2))) == frozenset()


@pytest.mark.parametrize("sigma", [0, -1, float("nan"), float("inf")])
def test_invalid_hill_width_rejected(sigma):
    with pytest.raises(ValueError):
        gaussian_coverage_objective(
            {"a": (0, 0)}, target_xy=(0, 0), sigma_m=sigma, scope_id="map", visibility_source="depth"
        )


def test_depth_projects_only_valid_surface_endpoints_with_world_transform():
    depth = np.array([[2.0, np.nan], [0.0, 9.0]])
    pose = np.eye(4)
    pose[:3, 3] = [3, 4, 0]
    cells = depth_surface_cells(depth, np.eye(3), pose, grid=CoverageGrid(1), max_depth_m=4, pixel_stride=1)
    assert cells == {"3,4"}
    assert "0,0" not in cells


def test_depth_optical_axis_can_map_to_world_xy_with_habitat_handedness():
    pose = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float)
    cells = depth_surface_cells(
        np.array([[2.0, 2.0]]), np.eye(3), pose, grid=CoverageGrid(1), max_depth_m=4, pixel_stride=1
    )
    assert cells == {"0,2", "2,2"}
    assert "0,1" not in cells


@pytest.mark.parametrize(
    "kwargs",
    [
        {"intrinsics": np.zeros((3, 3))},
        {"camera_to_world": np.zeros((4, 4))},
        {"pixel_stride": 0},
        {"pixel_stride": True},
        {"max_depth_m": float("inf")},
    ],
)
def test_depth_rejects_invalid_geometry_and_sampling(kwargs):
    values = {
        "depth": np.ones((2, 2)),
        "intrinsics": np.eye(3),
        "camera_to_world": np.eye(4),
        "grid": CoverageGrid(1),
        "max_depth_m": 4,
        **kwargs,
    }
    with pytest.raises(ValueError):
        depth_surface_cells(**values)


def test_depth_all_invalid_is_empty_not_missing():
    assert (
        depth_surface_cells(np.zeros((2, 2)), np.eye(3), np.eye(4), grid=CoverageGrid(1), max_depth_m=4) == frozenset()
    )
