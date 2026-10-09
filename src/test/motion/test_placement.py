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
    assert len(points) == 25
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


@pytest.mark.parametrize("query", ["none", "batched", "invalid_second_batch"])
def test_executor_search_accepts_observed_geometry_without_gt_query(rig, query):
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
    batches = []

    def check(poses):
        assert 1 <= len(poses) <= 32
        batches.append(np.array(poses))
        if query == "invalid_second_batch" and len(batches) == 2:
            return {"clear": [1] * len(poses)}
        return {"clear": [True] * len(poses)}

    executor.robot._state["sim_base_pose_query"] = query != "none"
    executor.robot.check_base_poses = check
    if query == "invalid_second_batch":
        with pytest.raises(ValueError, match="invalid_placement_clearance_response"):
            executor._search_placement("held", "support", approach_base=True)
        executor.robot.move_base_to.assert_not_called()
        return
    result, _ = executor._search_placement("held", "support", approach_base=query != "none")
    if query == "batched":
        assert [len(batch) for batch in batches] == [32, 17]
        expected = placement_base_candidates([.7, 0.], current_xyt=np.zeros(3))
        np.testing.assert_allclose(np.concatenate(batches), expected)
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
    scene = PlacementScene([], source="ground_truth")
    executor._placement_geometry = lambda *args: (scene, payload, [[0, 0, 0], [2, 1, 1]])
    with pytest.raises(ValueError, match="placement_support_moved"):
        executor._refresh_placement_checker("held", "support", SimpleNamespace(payload=payload, scene=scene))


def test_rrt_routes_held_object_around_blocked_direct_path(rig, monkeypatch):
    model, data, payload = rig
    scene = PlacementScene([[[.45, -.15, -.15], [.55, .15, .15]]], source="ground_truth")
    def global_rng_forbidden(*args, **kwargs):
        raise AssertionError("Seeded placement must not consume global RNG")

    monkeypatch.setattr(np.random, "random", global_rng_forbidden)
    monkeypatch.setattr(np.random, "randint", global_rng_forbidden)
    monkeypatch.setattr("emet.motion.algo.rrt.random", global_rng_forbidden)

    def search():
        return plan_placement_paths(model, data, joint_names=("x", "y", "z"), ee_body="tool", robot_body="base",
            scene=scene, payload=payload, object_centers=[[1.09, 0, 0]], base_candidates=[[0, 0, 0]],
            set_base=set_base, contact_bodies=("tool",), max_solutions=1, ik_attempts=1, rrt_max_iter=500, seed=7)

    result = search()
    repeat = search()
    assert len(repeat.paths) == 1
    for first, second in zip(result.paths[0].segments, repeat.paths[0].segments, strict=True):
        np.testing.assert_array_equal(first, second)
    assert len(result.paths) == 1, result.rejections
    collision = PlacementCollisionChecker(model, robot_body="base", ee_body="tool", scene=scene,
        payload=payload, contact_bodies=("tool",))
    for segment in result.paths[0].segments:
        assert validated_dense_path(model, data, ("x", "y", "z"), segment, collision, joint_step=.005) is not None
    assert validated_dense_path(model, data, ("x", "y", "z"),
        [np.zeros(3), result.paths[0].segments[0][-1]], collision) is None


def test_collision_checker_preserves_input_pose_fk_and_contact_buffers(rig):
    model, data, _ = rig
    collision = checker(rig, [[[.1, -.01, -.01], [.12, .01, .01]]])
    data.qpos[7] = .3  # intentionally leave FK at the previous state
    qpos, xpos, geom_xpos = data.qpos.copy(), data.xpos.copy(), data.geom_xpos.copy()
    ncon, contacts = data.ncon, data.contact.dist.copy()
    collision.configuration_collides(model, data)
    np.testing.assert_array_equal(data.qpos, qpos)
    np.testing.assert_array_equal(data.xpos, xpos)
    np.testing.assert_array_equal(data.geom_xpos, geom_xpos)
    assert data.ncon == ncon
    np.testing.assert_array_equal(data.contact.dist, contacts)


def test_refreshed_obstacle_invalidates_previously_clear_segment(rig):
    from emet.controller.manipulation.kinematic_pick_place import KinematicPickPlaceExecutor

    model, data, payload = rig
    old = checker(rig, [])
    support = [[-1, -1, -.5], [1, 1, -.4]]
    new_scene = PlacementScene([[[.58, -.01, -.01], [.62, .01, .01]]], source="ground_truth")
    assert old.scene.geometry_digest != new_scene.geometry_digest
    executor = object.__new__(KinematicPickPlaceExecutor)
    executor._model, executor._data, executor.ee_body = model, data, "tool"
    executor.robot = SimpleNamespace(_spec=SimpleNamespace(base_link_name="base"))
    executor.profile = SimpleNamespace(gripper_contact_bodies=lambda: ("tool",))
    executor._sync_qpos_from_robot = lambda: True
    executor._placement_support_bounds = np.asarray(support)
    executor._placement_geometry = lambda *args: (new_scene, payload, support)
    path = [np.zeros(3), np.array([1., 0, 0])]
    assert validated_dense_path(model, data, ("x", "y", "z"), path, old) is not None
    refreshed = executor._refresh_placement_checker("held", "support", old)
    assert refreshed.scene is new_scene
    assert validated_dense_path(model, data, ("x", "y", "z"), path, refreshed) is None


def test_occupied_support_targets_rejected_before_ik_or_base_validation(rig):
    from unittest.mock import Mock

    model, data, payload = rig
    endpoint = Mock(side_effect=AssertionError("should not validate base for an occupied target"))
    # The current held pose is clear; only the requested destination is occupied.
    result = plan_placement_paths(model, data, joint_names=("x", "y", "z"), ee_body="tool", robot_body="base",
        scene=PlacementScene([[[.5, -.2, -.2], [.8, .2, .3]]], source="ground_truth"), payload=payload,
        object_centers=[[.6, 0, 0]], base_candidates=[[0, 0, 0]], set_base=set_base,
        contact_bodies=("tool",), endpoint_validator=endpoint)
    assert not result.paths
    assert result.rejections == {"target_payload_collision": 1}
    endpoint.assert_not_called()


def test_distant_start_does_not_starve_reachable_alternative_base(rig):
    model, data, payload = rig
    result = plan_placement_paths(model, data, joint_names=("x", "y", "z"), ee_body="tool", robot_body="base",
        scene=PlacementScene([], source="ground_truth"), payload=payload,
        object_centers=[[3., 0, 0], [3.1, 0, 0]], base_candidates=[[0, 0, 0], [2, 0, 0]],
        set_base=set_base, contact_bodies=("tool",), max_solutions=1,
        max_ik_calls=8, max_ik_calls_per_base=3)
    assert len(result.paths) == 1, result.rejections
    assert result.rejections["base_ik_budget_reached"] == 1
    np.testing.assert_allclose(result.paths[0].base_xyt, [2, 0, 0])


def test_snapshot_replays_exact_search_and_rejects_tampering(rig, tmp_path):
    from emet.motion.placement_replay import replay_snapshot, save_snapshot

    model, data, payload = rig
    path = save_snapshot(tmp_path / "snapshot", model, data,
        scene=PlacementScene([], source="ground_truth"), payload=payload,
        object_centers=[[.4, 0, 0]], base_candidates=[[0, 0, 0]], joint_names=("x", "y", "z"),
        ee_body="tool", robot_body="base", contact_bodies=("tool",),
        base_writer={"freejoint_name": "base_pose"})
    first, second = replay_snapshot(path), replay_snapshot(path)
    assert first.pop("planning_wall_s") >= 0
    second.pop("planning_wall_s")
    assert first == second
    assert first["solutions"] == 1
    with (path / "inputs.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        replay_snapshot(path)


def test_snapshot_never_drops_unknown_space_predicate(rig, tmp_path):
    from emet.motion.placement_replay import replay_snapshot, save_snapshot

    model, data, payload = rig
    path = save_snapshot(tmp_path / "snapshot", model, data,
        scene=PlacementScene([], source="observed_voxels", known_free=lambda _: False),
        payload=payload, object_centers=[[.4, 0, 0]], base_candidates=[[0, 0, 0]],
        joint_names=("x", "y", "z"), ee_body="tool", robot_body="base", contact_bodies=("tool",),
        base_writer={"freejoint_name": "base_pose"})
    with pytest.raises(ValueError, match="Unsupported"):
        replay_snapshot(path)
