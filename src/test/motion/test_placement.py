# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Placement geometry and multi-candidate path regressions with real MuJoCo FK/IK."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from emet.motion.placement import (
    placement_base_candidates,
    plan_placement_paths,
    surface_placement_centers,
    validated_dense_path,
)
from emet.motion.placement_geometry import HeldObject, PlacementCollisionChecker, PlacementScene


@pytest.fixture
def rig():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="base"><freejoint name="base_pose"/><geom type="box" size=".1 .1 .1" pos="0 0 -.5"/>
        <body name="tool"><joint name="x" type="slide" axis="1 0 0" range="-2 2"/>
          <joint name="y" type="slide" axis="0 1 0" range="-2 2"/>
          <joint name="z" type="slide" axis="0 0 1" range="-2 2"/>
          <geom type="box" size=".025 .025 .025"/>
        </body>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    payload = HeldObject.from_world_bounds([[.04, -.03, -.03], [.14, .03, .03]],
                                          ee_position=[0, 0, 0], ee_rotation=np.eye(3))
    return model, data, payload


def checker(rig, boxes, **kwargs):
    model, _, payload = rig
    return PlacementCollisionChecker(model, robot_body="base", ee_body="tool", payload=payload,
        scene=PlacementScene(boxes, source="ground_truth"), contact_bodies=("tool",), margin_m=0, **kwargs)


def set_base(model, data, pose):
    data.qpos[:2] = pose[:2]
    data.qpos[3:7] = [np.cos(pose[2] / 2), 0, 0, np.sin(pose[2] / 2)]
    return True


def test_payload_hits_obstacle_that_arm_clears(rig):
    model, data, _ = rig
    collision = checker(rig, [[[.1, -.01, -.01], [.12, .01, .01]]])
    assert collision.configuration_collides(model, data)
    assert collision.last_reason == "payload_scene_collision"
    assert not checker(rig, [[[.3, -.01, -.01], [.32, .01, .01]]]).configuration_collides(model, data)


def test_arm_volume_not_only_link_origin(rig):
    model, data, _ = rig
    collision = checker(rig, [[[-.03, -.01, -.01], [-.02, .01, .01]]])
    assert collision.configuration_collides(model, data)
    assert collision.last_reason == "robot_scene_collision"


def test_rotation_moves_offset_payload(rig):
    model, data, _ = rig
    collision = checker(rig, [[[-.01, .1, -.01], [.01, .12, .01]]])
    assert not collision.configuration_collides(model, data)
    set_base(model, data, [0, 0, np.pi / 2])
    assert collision.configuration_collides(model, data)
    assert collision.last_reason == "payload_scene_collision"


def test_payload_robot_collision_except_explicit_grasp_contacts(rig):
    model, data, _ = rig
    data.qpos[9] = -.5
    collision = checker(rig, [])
    assert collision.configuration_collides(model, data)
    assert collision.last_reason == "payload_robot_collision"


def test_edge_catches_payload_collision_with_clear_endpoints(rig):
    model, data, _ = rig
    collision = checker(rig, [[[.58, -.01, -.01], [.62, .01, .01]]])
    assert validated_dense_path(model, data, ("x", "y", "z"),
                                [np.zeros(3), np.array([1., 0, 0])], collision) is None


def test_voxel_scene_uses_height_and_rejects_unknown():
    scene = PlacementScene.from_voxels([[0, 0, 1]], resolution=.1, workspace=[[-2]*3, [2]*3])
    assert scene.collides([[-.01, -.01, .99], [.01, .01, 1.01]])
    assert not scene.collides([[-.01, -.01, 0], [.01, .01, .1]])
    assert scene.collides([[2, 0, 0], [2.1, .1, .1]])
    unknown = PlacementScene.from_voxels(np.empty((0, 3)), resolution=.1, workspace=[[-2]*3, [2]*3],
                                        known_free=lambda bounds: bounds[1, 0] < 0)
    assert unknown.collides([[0, 0, 0], [.1, .1, .1]])
    assert not unknown.collides([[-1, 0, 0], [-.5, .1, .1]])


def test_pointcloud_centroids_are_conservative():
    vm = SimpleNamespace(voxel_resolution=.1, get_pointcloud=lambda: (np.array([[0., 0, 0]]),))
    scene = PlacementScene.from_pointcloud(vm, workspace=[[-1]*3, [1]*3])
    assert scene.collides([[.09, 0, 0], [.095, .01, .01]])


def test_gt_only_excludes_held_object_and_requires_bounds():
    placements = {"held": {"bounds": [[0]*3, [1]*3]}, "support": {"bounds": [[2]*3, [3]*3]}}
    scene = PlacementScene.from_placements(placements, held_object="held")
    assert not scene.collides([[.1]*3, [.2]*3])
    assert scene.collides([[2.1]*3, [2.2]*3])
    with pytest.raises(ValueError, match="Missing scene bounds"):
        PlacementScene.from_placements({"held": {}, "support": {}}, held_object="held")


def test_surface_targets_fit_whole_payload(rig):
    _, _, payload = rig
    points = surface_placement_centers([[-.3, -.3, -.2], [.3, .3, 0]], payload=payload, ee_rotation=np.eye(3))
    assert len(points) == 9
    assert all(np.isclose(p[2], .05) for p in points)
    assert surface_placement_centers([[0]*3, [.01]*3], payload=payload, ee_rotation=np.eye(3)) == []


@pytest.mark.parametrize("source", ["ground_truth", "observed_voxels"])
def test_search_rejects_blocked_base_finds_multiple_complete_paths_preserves_input(rig, source):
    model, data, payload = rig
    original = data.qpos.copy()
    # First base intersects obstacle; alternatives can reach target with payload.
    boxes = [[[-.11, -.11, -.61], [.11, .11, -.39]]]
    scene = PlacementScene(boxes, source=source, workspace=[[-4]*3, [4]*3])
    result = plan_placement_paths(model, data, joint_names=("x", "y", "z"), ee_body="tool", robot_body="base",
        scene=scene, payload=payload, object_centers=[[.7, .2, .1], [.7, -.2, .1]],
        base_candidates=[[0, 0, 0], [.4, 0, 0]], set_base=set_base, contact_bodies=("tool",),
        max_solutions=2, ik_attempts=1, rrt_max_iter=100)
    assert len(result.paths) == 2, result.rejections
    assert result.rejections["robot_scene_collision"] == 1
    np.testing.assert_array_equal(data.qpos, original)
    for path in result.paths:
        np.testing.assert_allclose(path.base_xyt, [.4, 0, 0])
        set_base(model, data, path.base_xyt)
        data.qpos[7:10] = path.segments[-1][-1]
        mujoco.mj_forward(model, data)
        vertices = payload.world_vertices(data.body("tool").xpos, data.body("tool").xmat)
        np.testing.assert_allclose((vertices.min(0) + vertices.max(0)) / 2, path.object_center, atol=.005)


def test_no_feasible_candidate_returns_no_path(rig):
    model, data, payload = rig
    result = plan_placement_paths(model, data, joint_names=("x", "y", "z"), ee_body="tool", robot_body="base",
        scene=PlacementScene([[[-3]*3, [3]*3]], source="ground_truth"), payload=payload,
        object_centers=[[.5, 0, 0]], base_candidates=[[0, 0, 0]], set_base=set_base, contact_bodies=("tool",))
    assert not result.paths
    assert sum(result.rejections.values()) == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 0])
def test_invalid_voxel_resolution_fails_closed(value):
    with pytest.raises(ValueError):
        PlacementScene.from_voxels([[0, 0, 0]], resolution=value, workspace=[[-1]*3, [1]*3])


def test_candidates_span_distances_and_directions():
    poses = placement_base_candidates([0, 0], current_xyt=[2, 0, 0], count=8)
    assert len(poses) == 25
    assert len({tuple(p) for p in poses}) == 25


def test_voxel_executor_requires_explicit_observed_provider(rig):
    from emet.controller.manipulation.kinematic_pick_place import KinematicPickPlaceExecutor

    executor = object.__new__(KinematicPickPlaceExecutor)
    executor.manip_collision = "voxel"
    executor.placement_geometry_provider = None
    with pytest.raises(ValueError, match="observed_placement_geometry_provider_required"):
        executor._placement_geometry("held", "support")


def test_executor_search_accepts_observed_geometry_without_gt_query(rig):
    from unittest.mock import Mock

    from emet.controller.manipulation.kinematic_pick_place import KinematicPickPlaceExecutor

    model, data, payload = rig
    executor = object.__new__(KinematicPickPlaceExecutor)
    executor._model, executor._data = model, data
    executor.ee_body, executor.joint_names, executor.arm = "tool", ("x", "y", "z"), "left"
    executor.place_z_offset_m, executor.rrt_max_iter = .02, 100
    executor._sync_qpos_from_robot = lambda: True
    executor._world_base_xyt = lambda: np.zeros(3)
    executor._planar_joint_names = lambda: None
    executor.profile = SimpleNamespace(base_freejoint_name="base_pose", gripper_contact_bodies=lambda: ("tool",))
    executor.robot = SimpleNamespace(_spec=SimpleNamespace(base_link_name="base", tamp_approach="front"),
                                    _state={}, move_base_to=Mock())
    scene = PlacementScene.from_voxels(np.empty((0, 3)), resolution=.02, workspace=[[-3]*3, [3]*3])
    executor.placement_geometry_provider = lambda *args: (scene, payload, [[.4, -.3, -.2], [1., .3, -.1]])
    result, _ = executor._search_placement("held", "support", approach_base=False)
    assert len(result.paths) == 3
    assert result.geometry_source == "observed_voxels"
    executor.robot.move_base_to.assert_not_called()


def test_planning_budget_and_live_endpoint_rejection(rig):
    model, data, payload = rig
    common = {"joint_names": ("x", "y", "z"), "ee_body": "tool", "robot_body": "base",
        "scene": PlacementScene([], source="ground_truth"), "payload": payload,
        "object_centers": [[.7, .2, .1]], "base_candidates": [[0, 0, 0]],
        "set_base": set_base, "contact_bodies": ("tool",)}
    result = plan_placement_paths(model, data, **common, endpoint_validator=lambda pose: False)
    assert not result.paths
    assert result.rejections == {"base_endpoint_rejected": 1}
    result = plan_placement_paths(model, data, **common, max_ik_calls=1)
    assert not result.paths  # one successful preplace cannot stand for a complete placement
    assert result.rejections == {"ik_budget_exhausted": 1}


def test_refresh_rejects_moved_support_before_execution(rig):
    from emet.controller.manipulation.kinematic_pick_place import KinematicPickPlaceExecutor

    _, _, payload = rig
    executor = object.__new__(KinematicPickPlaceExecutor)
    executor._sync_qpos_from_robot = lambda: True
    executor._placement_support_bounds = np.array([[0, 0, 0], [1, 1, 1]])
    executor._placement_geometry = lambda *args: (None, payload, [[0, 0, 0], [2, 1, 1]])
    with pytest.raises(ValueError, match="placement_support_moved"):
        executor._refresh_placement_checker("held", "support", SimpleNamespace(payload=payload))
