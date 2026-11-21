# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from emet.mapping.voxel import voxel_dynamem
from emet.mapping.voxel.floor_evidence import floor_evidence
from emet.mapping.voxel.launch_floor_survey import register_launch_floor_survey
from emet.mapping.voxel.voxel_dynamem import SparseVoxelMap


def evidence(extra=(), **options):
    ring = [[-0.1, 0, 0], [0.1, 0, 0], [0, -0.1, 0], [0, 0.1, 0]]
    return floor_evidence(ring + list(extra), (9, 9), 0.1, [4, 4], reference_z_m=0, **options)


def test_single_supported_hole_is_inferred_but_not_observed():
    layers = evidence()
    assert layers.inferred[4, 4] and not layers.observed[4, 4]
    assert layers.inferred.sum() == 1


def test_floor_layers_are_read_only_and_outside_edge_cannot_create_support():
    layers = floor_evidence([[-0.46, 0, 0]], (9, 9), 0.1, [4, 4], reference_z_m=0)
    assert not layers.observed.any()
    for layer in (layers.observed, layers.inferred, layers.drops, layers.raised):
        with pytest.raises(ValueError, match="read-only"):
            layer[0, 0] = True


@pytest.mark.parametrize("height,kind", [(-0.2, "drops"), (0.1, "raised")])
def test_discontinuity_is_hazard_never_filled(height, kind):
    layers = evidence([[0, 0, height]])
    assert getattr(layers, kind)[4, 4]
    assert not layers.inferred[4, 4] and not layers.observed[4, 4]


def test_ambiguous_height_and_missing_returns_remain_unknown():
    layers = evidence([[0, 0, 0.05]])
    assert not layers.inferred[4, 4] and not layers.observed[4, 4]
    empty = floor_evidence([], (9, 9), 0.1, [4, 4], reference_z_m=0)
    assert not any(getattr(empty, key).any() for key in ("observed", "inferred", "drops", "raised"))


def test_conflicting_floor_returns_are_reported_separately_from_missing_depth_and_hazards():
    # The recorded q2 failure has substantial floor support but two returns just
    # outside the band. It is not missing depth, nor a measured 8 cm drop.
    layers = evidence([[0, 0, 0]] * 24 + [[0, 0, -0.0315], [0, 0, -0.0305]])
    assert layers.ambiguous[4, 4] and not layers.observed[4, 4]
    assert not layers.drops[4, 4] and not layers.raised[4, 4]
    assert not evidence().ambiguous[4, 4]
    hazard = evidence([[0, 0, 0], [0, 0, -0.2]])
    assert hazard.drops[4, 4] and not hazard.ambiguous[4, 4]
    with pytest.raises(ValueError, match="read-only"):
        layers.ambiguous[4, 4] = False


def test_no_frontier_growth_or_iterative_fill():
    layers = evidence()
    assert not layers.inferred[4, 6]
    assert not layers.inferred[0].any()
    assert layers.observed.sum() == 4


def test_calibrated_reference_can_be_above_world_zero():
    layers = floor_evidence([[0, 0, 2], [0.1, 0, 1.8]], (9, 9), 0.1, [4, 4], reference_z_m=2)
    assert layers.observed[4, 4] and layers.drops[5, 4]


def test_hazard_wins_over_an_old_floor_return():
    layers = evidence([[0, 0, 0], [0, 0, -0.2]])
    assert layers.drops[4, 4] and not layers.observed[4, 4]


def test_measured_floor_keeps_valid_depth_before_voxel_pooling(tmp_path):
    vm = SparseVoxelMap(
        grid_size=[40, 40], grid_resolution=.1, smooth_kernel_size=0, point_update_threshold=.999,
        log=str(tmp_path), device="cpu", image_shape=None, add_local_radius_points=False,
        parameters={"floor_evidence": {"enabled": True, "reference_z_m": 0.}},
    )
    pose = np.diag([1., -1., -1., 1.])
    pose[2, 3] = 1.
    intrinsics = np.array([[10., 0., 4.], [0., 10., 4.], [0., 0., 1.]])
    vm.process_rgbd_images(
        np.zeros((8, 8, 3), dtype=np.uint8), np.ones((8, 8), dtype=np.float32),
        intrinsics, pose, full_perception=False,
    )
    assert vm.get_navigation_map()[1].sum() == 64


def test_launch_survey_requires_alignment_and_expires_from_the_cached_map(tmp_path, monkeypatch):
    vm = SparseVoxelMap(
        grid_size=[40, 40], grid_resolution=.1, smooth_kernel_size=0,
        log=str(tmp_path), device="cpu", image_shape=None, use_instance_memory=False,
        parameters={"floor_evidence": {"enabled": True, "reference_z_m": 0.}},
    )
    now = 30.
    monkeypatch.setattr(voxel_dynamem, "time", SimpleNamespace(monotonic=lambda: now))
    anchors = np.array([[0., 0., 0.], [.3, 0., 0.], [0., .25, 0.], [.3, .25, 0.]])
    rotation = np.array([[0., 1.], [-1., 0.]])
    offset = np.array([.1, -.1])
    target = np.column_stack((anchors[:, :2] @ rotation + offset, anchors[:, 2]))
    points = np.array([[0., 0., 0.], [.1, 0., 0.], [.1, 0., -.2], [2., 2., 0.]])
    options = {
        "observation_scope_id": vm.observation_scope_id, "survey_captured_at": 25.,
        "anchors_captured_at": 29., "now": now, "start_xy": [0., 0.],
        "radius_m": .5, "max_age_s": 10., "max_anchor_error_m": .015,
    }
    assert not vm.get_navigation_map()[1].any()
    misaligned = target.copy()
    misaligned[0, 2] += .1
    with pytest.raises(ValueError, match="anchor_misaligned"):
        register_launch_floor_survey(points, anchors, misaligned, **options)
    assert not vm.get_navigation_map()[1].any()
    survey = register_launch_floor_survey(points, anchors, target, **options)
    np.testing.assert_allclose(survey.points[:, :2], points[:3, :2] @ rotation + offset, atol=1e-12)
    vm.install_launch_floor_survey(survey)
    blocked, observed = vm.get_navigation_map()
    assert observed.sum() == 1
    assert blocked.sum() == 1
    now = 35.
    assert not vm.get_navigation_map()[1].any()
    with pytest.raises(ValueError, match="stale_or_wrong_scope"):
        vm.install_launch_floor_survey(survey)


@pytest.mark.parametrize("options", [{"reference_z_m": float("nan")}, {"tolerance_m": 0.1, "hazard_height_m": 0.05}])
def test_invalid_calibration_fails(options):
    with pytest.raises(ValueError):
        floor_evidence([], (9, 9), 0.1, [4, 4], **({"reference_z_m": 0} | options))


@pytest.mark.parametrize("origin", [[4, 4, 0, 0], [4], [float("nan"), 4]])
def test_invalid_origin_fails_explicitly(origin):
    with pytest.raises(ValueError, match="origin"):
        floor_evidence([], (9, 9), 0.1, origin, reference_z_m=0)


def test_numeric_calibration_strings_are_normalized_before_comparison():
    layers = floor_evidence([[0, 0, 0]], (9, 9), "0.1", [4, 4, 0], reference_z_m="0")
    assert layers.observed[4, 4]
    with pytest.raises(ValueError):
        floor_evidence([], (9, 9), "not-a-resolution", [4, 4], reference_z_m=0)


@pytest.mark.parametrize("reference", [0, 2])
def test_map_integration_does_not_turn_visited_disk_into_floor(tmp_path, reference):
    vm = SparseVoxelMap(
        grid_size=[40, 40],
        grid_resolution=0.1,
        smooth_kernel_size=0,
        obs_min_density=0.1,
        log=str(tmp_path),
        device="cpu",
        image_shape=None,
        parameters={"floor_evidence": {"enabled": True, "reference_z_m": reference}},
    )
    xyz = torch.tensor([[-0.1, 0, 0], [0.1, 0, 0], [0, -0.1, 0], [0, 0.1, 0], [0.5, 0.5, -0.2]])
    xyz[:, 2] += reference
    vm.voxel_pcd.add(xyz, features=None, rgb=torch.zeros(5, 3), obs_count=1)
    vm._visited[:] = 1
    obstacles, known = vm.get_navigation_map()
    assert known.sum() == 4  # Only measured floor, not the disk or interpolation.
    assert obstacles.any()  # The negative surface is blocked even if visited.
    assert vm.get_sensor_observed_cells().sum() == 5  # Inferred cell is not added to sensor coverage.
    center = tuple(vm.grid_origin[:2].long().tolist())
    assert vm._floor_evidence.inferred[center]
    assert not known[center] and not vm.get_sensor_observed_cells()[center]
    from emet.mapping.voxel.voxel_map_dynamem import SparseVoxelMapNavigationSpace
    from emet.robots.footprint import Footprint

    space = SparseVoxelMapNavigationSpace(vm, obstacle_map_mode="physical", footprint=Footprint(0.1, 0.1))
    assert not space.is_valid(np.array([0.5, 0.5, 0.0]))
    assert space.last_validity["terrain_hazard"] == "drop"


@pytest.mark.parametrize("height,kind", [(-0.2, "drops"), (0.12, "raised"), (0.0, "observed")])
def test_mujoco_measured_surface_distinguishes_step_drop_and_floor(height, kind):
    import mujoco

    model = mujoco.MjModel.from_xml_string(f"""<mujoco><worldbody>
      <geom name="surface" type="box" pos="0 0 {height - 0.1}" size="1 1 .1"/>
      </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    hit = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(model, data, np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0]), None, True, -1, hit)
    assert hit[0] >= 0
    layers = floor_evidence([[0, 0, 1.0 - distance]], (9, 9), 0.1, [4, 4], reference_z_m=0)
    assert getattr(layers, kind)[4, 4]
