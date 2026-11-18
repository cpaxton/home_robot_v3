# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from emet.eval.grapheqa_navigation import (
    HabitatMotionExecutor,
    ObservedSpaceNavigation,
    base_xyt,
    predicted_camera_pose,
    stationary_floor_reference,
)
from emet.eval.grapheqa_sensor import command_camera_pose
from emet.robots.footprint import CircularFootprint


def test_command_prior_uses_local_camera_heading_and_head_motion_preserves_base():
    camera = command_camera_pose(yaw_deg=90, tilt_deg=-30, sensor_height=1.5)
    forward = predicted_camera_pose(camera, "move_forward", 0.1)
    np.testing.assert_allclose(forward[:3, 3], [0, 0.1, 1.5], atol=1e-12)
    lowered = predicted_camera_pose(camera, "look_down", 5)
    np.testing.assert_allclose(base_xyt(lowered), base_xyt(camera), atol=1e-12)
    assert np.isclose(np.rad2deg(np.arcsin(lowered[2, 2])), -35)


def test_circular_footprint_is_rotation_invariant_and_includes_intersected_cells():
    footprint = CircularFootprint(0.1)
    expected = footprint.grid_cells(0.05, [0, 0, 0], [10, 10])
    np.testing.assert_array_equal(expected, footprint.grid_cells(0.05, [0, 0, np.pi / 4], [10, 10]))
    cells = set(map(tuple, expected))
    assert (12, 11) in cells
    assert (12, 12) not in cells
    assert len(cells) == 21
    shifted = footprint.grid_cells(0.05, [0.025, 0, 0], [10, 10])
    assert (13, 10) in set(map(tuple, shifted))


def test_frontiers_exclude_disconnected_observed_islands():
    nav = object.__new__(ObservedSpaceNavigation)
    nav.pose = np.zeros(3)
    explored = np.zeros((40, 40), dtype=bool)
    explored[3:20, 3:20] = True
    explored[25:35, 25:35] = True
    navigable = explored.copy()
    origin = torch.tensor([4.0, 4.0, 0.0])
    nav.map = SimpleNamespace(grid_origin=origin)
    nav._navigable = lambda: (navigable, explored)
    points = nav.frontiers()
    assert points and all(np.linalg.norm(point[:2]) < 1.0 for point in points)


def test_observed_space_plan_rejects_unobserved_goal_without_sweeping():
    nav = object.__new__(ObservedSpaceNavigation)
    nav.pose = np.zeros(3)
    nav.map = SimpleNamespace(grid_origin=torch.tensor([10.0, 10.0, 0.0]))
    free = np.ones((20, 20), dtype=bool)
    nav._navigable = lambda: (free, free)
    assert nav.plan([100, 100, 0], object_target=False) == ([], "no_valid_goals")


def test_frontier_arrival_preserves_travel_heading_instead_of_resetting_to_world_zero():
    nav = object.__new__(ObservedSpaceNavigation)
    nav.pose = np.zeros(3)
    nav.map = SimpleNamespace(grid_origin=torch.tensor([10.0, 10.0, 0.0]))
    nav.space = SimpleNamespace(is_valid=lambda pose: True)
    free = np.ones((20, 20), dtype=bool)
    nav._navigable = lambda: (free, free)
    route, reason = nav.plan([0, 0.2, 0], object_target=False)
    assert reason == "ok" and route
    assert route[-1][2] == pytest.approx(np.pi / 2)


def test_rotation_intent_survives_small_measured_xy_drift():
    class Sensor:
        def __init__(self):
            self.last = {"camera_pose": command_camera_pose(yaw_deg=0, tilt_deg=-30, sensor_height=1.5)}
            self.actions = []

        def step(self, action, amount):
            self.actions.append(action)
            pose = predicted_camera_pose(self.last["camera_pose"], action, amount)
            pose[0, 3] = 0.021  # Within the shared route's position tolerance.
            self.last = {"camera_pose": pose, "collided": False}
            return self.last

    sensor = Sensor()
    navigation = SimpleNamespace(space=SimpleNamespace(is_valid=lambda pose: True), update=lambda frame: None)
    executor = HabitatMotionExecutor(sensor, navigation)
    assert executor.move_base_to([0, 0, np.pi / 2], blocking=True, world_frame=True, timeout=10)
    assert sensor.actions and set(sensor.actions) == {"turn_left"}


def floor_frame(height):
    k = np.array([[185.0, 0, 320], [0, 185.0, 240], [0, 0, 1]])
    pose = command_camera_pose(yaw_deg=0, tilt_deg=-45, sensor_height=1.5)
    v, u = np.mgrid[:480, :640]
    rays = np.stack(((u - 320) / 185, (v - 240) / 185, np.ones_like(u)), axis=-1) @ pose[:3, :3].T
    depth = (height - 1.5) / np.minimum(rays[..., 2], -1e-9)
    depth[(rays[..., 2] >= 0) | (depth > 6)] = 0
    return {"depth": depth.astype(np.float32), "camera_K": k, "camera_pose": pose, "timestamp_ns": 1}


def test_stationary_measured_floor_reference_preserves_subsequent_drop_hazards():
    from emet.mapping.voxel.floor_evidence import floor_evidence

    measured = stationary_floor_reference(floor_frame(-0.125))
    assert measured["reference_z_m"] == pytest.approx(-0.125, abs=1e-6)
    layers = floor_evidence(
        [[0, 0, -0.125], [0.2, 0, -0.25]], (40, 40), 0.05, [20, 20], reference_z_m=measured["reference_z_m"]
    )
    assert layers.observed[20, 20] and layers.drops[24, 20]


def test_floor_calibration_rejects_missing_and_mixed_underfoot_returns():
    missing = floor_frame(-0.125)
    missing["depth"][:] = 0
    with pytest.raises(ValueError, match="Insufficient"):
        stationary_floor_reference(missing)
    mixed = floor_frame(-0.125)
    mixed["depth"][:, 320:] = floor_frame(0)["depth"][:, 320:]
    with pytest.raises(ValueError, match="Ambiguous"):
        stationary_floor_reference(mixed)


def test_floor_reference_cannot_be_recalibrated_after_leaving_spawn():
    nav = object.__new__(ObservedSpaceNavigation)
    nav.floor_calibration = None
    nav.pose = np.array([1, 0, 0])
    with pytest.raises(RuntimeError, match="original stationary"):
        nav.calibrate_stationary_floor(floor_frame(0))
    nav.pose = np.zeros(3)
    nav.floor_calibration = {"reference_z_m": 0}
    with pytest.raises(RuntimeError, match="fixed for the entire episode"):
        nav.calibrate_stationary_floor(floor_frame(0))
