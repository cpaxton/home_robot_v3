# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from emet.controller.manipulation.geometry_models import MeasuredSurfaceModel, PrimitiveModel
from emet.controller.manipulation.geometry_motion import check_geometry_comotion
from emet.controller.manipulation.geometry_tracking import GeometryTracker, fit_geometry, fit_geometry_views
from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.motion.observed_object_bound import ObservedObjectBound, enclose_observed_object
from emet.motion.observed_workspace import ObservedWorkspace


def cylinder_points(*, partial=False):
    theta = np.linspace(-1.0, 1.0, 35) if partial else np.linspace(0, 2 * np.pi, 35, endpoint=False)
    z = np.linspace(-0.015, 0.015, 12) if partial else np.linspace(-0.04, 0.04, 12)
    t, h = np.meshgrid(theta, z)
    points = np.column_stack((0.02 * np.cos(t.ravel()), 0.02 * np.sin(t.ravel()), h.ravel()))
    if not partial:
        caps = []
        for sign in (-1, 1):
            for radius in (0.0, 0.01, 0.018):
                caps.append(
                    np.column_stack((radius * np.cos(theta), radius * np.sin(theta), np.full(len(theta), sign * 0.04)))
                )
        points = np.concatenate([points, *caps])
    return points


def fit(points, model=None, **kwargs):
    return fit_geometry(
        points,
        model or PrimitiveModel("cylinder", (0.02, 0.04)),
        object_ref="opaque",
        observation_ids=("capture",),
        received_monotonic=10.0,
        max_evaluations=18000,
        **kwargs,
    )


def test_textureless_full_geometry_recovers_translation_axis_but_not_yaw():
    rotation = Rotation.from_euler("xyz", [0.2, -0.1, 0.7]).as_matrix()
    center = np.array([0.1, -0.2, 0.5])
    result = fit(cylinder_points() @ rotation.T + center)
    assert result.translation_observable
    assert not result.orientation_observable
    np.testing.assert_allclose(result.poses[0, :3, 3], center, atol=1e-4)
    assert abs(result.poses[0, :3, 2] @ rotation[:, 2]) > 0.999
    assert result.as_dict()["execution_qualified"] is False
    with pytest.raises(ObservationUnavailable, match="ambiguous"):
        result.require_unique(now=10.1, position_error_m=0.002, rotation_error_rad=0.02)
    with pytest.raises(ObservationUnavailable, match="uncalibrated"):
        result.require_unique(now=10.1)


def test_partial_side_patch_does_not_invent_axial_translation_observability():
    result = fit(cylinder_points(partial=True) + [0.1, 0.2, 0.5])
    assert not result.translation_observable
    assert any(np.max(np.linalg.norm(space[:, :3], axis=1)) > 0.5 for space in result.nullspaces if len(space))


def test_bad_surface_and_work_budget_fail_without_fabricated_pose():
    rng = np.random.default_rng(2)
    with pytest.raises(ObservationUnavailable):
        fit(rng.uniform(-0.1, 0.1, (100, 3)))
    with pytest.raises(ObservationUnavailable, match="budget"):
        fit_geometry(
            cylinder_points(),
            PrimitiveModel("cylinder", (0.02, 0.04)),
            object_ref="opaque",
            observation_ids=("capture",),
            received_monotonic=10.0,
            max_evaluations=20,
        )


def test_multiview_fusion_preserves_original_time_and_rejects_scene_changes():
    points = cylinder_points()
    views = [
        {"scene_revision": "epoch", "points_world": p, "observation_id": str(i), "received_monotonic": 10.0 + i * 0.1}
        for i, p in enumerate(np.array_split(points, 2))
    ]
    result = fit_geometry_views(
        views,
        PrimitiveModel("cylinder", (0.02, 0.04)),
        scene_revision="epoch",
        object_ref="opaque",
        max_evaluations=18000,
    )
    assert result.received_monotonic == 10.0
    assert result.observation_ids == ("0", "1")
    views[1]["scene_revision"] = "moved"
    with pytest.raises(ObservationUnavailable, match="scene_changed"):
        fit_geometry_views(views, PrimitiveModel("cylinder", (0.02, 0.04)), scene_revision="epoch", object_ref="opaque")


def test_measured_model_requires_real_enclosure_and_keeps_tier_distinct():
    cells = np.ones((5, 5, 5), np.uint8)
    cells[2, 2, 2] = 2
    grid = ObservedWorkspace(np.full(3, -0.125), 0.05, cells)
    bound = enclose_observed_object(
        grid.partition([[0.0, 0.0, 0.0]], np.array(["target"])),
        "target",
        observation_ids=("view:1", "view:2"),
        received_monotonic=10.0,
        scene_revision="scan",
        topology="connected_rigid",
    )
    rng = np.random.default_rng(0)
    points = rng.normal(size=(100, 3))
    points = points / np.linalg.norm(points, axis=1, keepdims=True) * 0.02
    model = MeasuredSurfaceModel(points, world_from_reference=np.eye(4), enclosure=bound)
    assert model.construction == "observed_multiview_surface"
    assert model.observation_ids == ("view:1", "view:2")
    np.testing.assert_allclose(model.validation_distance(points), 0.0)
    with pytest.raises(ValueError):
        MeasuredSurfaceModel(points + [1.0, 0.0, 0.0], world_from_reference=np.eye(4), enclosure=bound)


def test_geometry_comotion_accepts_symmetry_but_rejects_drop_and_static_lift():
    initial = fit(cylinder_points() + [0, 0, 0.04])
    moved = fit(cylinder_points() + [0, 0, 0.14])
    moved = replace(moved, received_monotonic=10.5, observation_ids=("next-capture",))
    before, after = np.eye(4), np.eye(4)
    before[2, 3] = 0.04
    after[2, 3] = 0.14
    result = check_geometry_comotion(
        initial,
        moved,
        ee_before=before,
        ee_after=after,
        position_error_m=0.001,
        rotation_error_rad=0.01,
        require_lift=True,
        support_top_z=0.0,
    )
    assert result["state"] == "consistent" and result["execution_qualified"] is False
    replayed = replace(moved, observation_ids=initial.observation_ids)
    assert (
        check_geometry_comotion(
            initial, replayed, ee_before=before, ee_after=after, position_error_m=0.001, rotation_error_rad=0.01
        )["state"]
        == "unknown"
    )
    dropped = replace(initial, received_monotonic=10.5, observation_ids=("drop-capture",))
    result = check_geometry_comotion(
        initial,
        dropped,
        ee_before=before,
        ee_after=after,
        position_error_m=0.001,
        rotation_error_rad=0.01,
        require_lift=True,
        support_top_z=0.0,
    )
    assert result["state"] == "inconsistent"
    result = check_geometry_comotion(
        initial,
        dropped,
        ee_before=before,
        ee_after=before,
        position_error_m=0.001,
        rotation_error_rad=0.01,
        require_lift=True,
        support_top_z=0.0,
    )
    assert result["state"] == "inconsistent"


def test_tracker_failure_or_replay_discards_previous_pose():
    tracker = GeometryTracker(PrimitiveModel("cylinder", (0.02, 0.04)), object_ref="target", max_evaluations=18000)
    tracker.update(
        cylinder_points(), observation_id="capture:1", received_monotonic=10.0, scene_epoch="epoch", now=10.1
    )
    with pytest.raises(ObservationUnavailable):
        tracker.update(
            cylinder_points(), observation_id="capture:1", received_monotonic=10.2, scene_epoch="epoch", now=10.3
        )
    assert tracker.latest is None


def test_measured_multiview_surface_registers_without_primitive_prior():
    rng = np.random.default_rng(42)
    points = rng.normal(size=(800, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    points *= [0.02, 0.03, 0.04]
    points[:, 0] += 0.005 * (points[:, 2] / 0.04) ** 2
    enclosure = ObservedObjectBound(np.array([[-0.075] * 3, [0.075] * 3]), ("a", "b"), 10.0, "epoch")
    model = MeasuredSurfaceModel(points, world_from_reference=np.eye(4), enclosure=enclosure)
    # Caller mutations cannot alter the measured reference or its search tree.
    points[:] = 99.0
    reference = model.points_reference.copy()
    np.testing.assert_allclose(model.validation_distance(reference), 0.0)
    rotation = Rotation.from_euler("xyz", [0.07, -0.05, 0.12]).as_matrix()
    center = np.array([0.1, 0.2, 0.5])
    result = fit(reference @ rotation.T + center, model=model, max_residual_m=0.001)
    assert result.model.construction == "observed_multiview_surface"
    np.testing.assert_allclose(result.poses[0, :3, 3], center, atol=1e-5)
    assert Rotation.from_matrix(result.poses[0, :3, :3].T @ rotation).magnitude() < 0.001
    assert not result.as_dict()["execution_qualified"]


def test_fresh_measured_motion_updates_geometry_and_epoch_change_stops_tracking():
    tracker = GeometryTracker(PrimitiveModel("cylinder", (0.02, 0.04)), object_ref="target", max_evaluations=18000)
    tracker.update(cylinder_points(), observation_id="first", received_monotonic=10.0, scene_epoch="epoch", now=10.1)
    result = tracker.update(
        cylinder_points() + [0.02, 0.0, 0.1],
        observation_id="second",
        received_monotonic=10.5,
        scene_epoch="epoch",
        now=10.6,
    )
    np.testing.assert_allclose(result.poses[0, :3, 3], [0.02, 0.0, 0.1], atol=1e-4)
    with pytest.raises(ObservationUnavailable, match="scene_changed"):
        tracker.update(
            cylinder_points(), observation_id="third", received_monotonic=11.0, scene_epoch="changed", now=11.1
        )
    assert tracker.latest is None
