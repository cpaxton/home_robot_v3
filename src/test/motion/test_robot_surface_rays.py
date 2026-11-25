# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from emet.controller.manipulation.robot_surface_rays import measured_robot_surfaces


def test_robot_ray_association_preserves_nearer_obstacle_returns():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
        <geom type="box" pos="0 0 1" size=".3 .3 .1"/>
        </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    depth = np.full((21, 21), .91)
    depth[10, 10] = .85
    obs = SimpleNamespace(rgb=np.zeros((21, 21, 3), np.uint8), depth=depth,
                          camera_K=np.array([[100., 0., 10.], [0., 100., 10.], [0., 0., 1.]]),
                          camera_pose=np.eye(4))
    points = measured_robot_surfaces(model, data, [obs], tolerance_m=.015)
    assert len(points) == 440
    assert np.all(points[:, 2] == .91)


def test_visual_surface_matching_uses_visual_mesh_instead_of_collision_proxy():
    model = mujoco.MjModel.from_xml_string('''<mujoco><asset>
      <mesh name="visual" vertex="-.1 -.1 -.1  .1 -.1 -.1  -.1 .1 -.1  .1 .1 -.1
                                  -.1 -.1 .1   .1 -.1 .1   -.1 .1 .1   .1 .1 .1"/>
      </asset><worldbody>
      <geom type="box" pos="0 0 1" size=".1 .1 .1" group="0"/>
      <geom type="mesh" mesh="visual" pos="0 0 1.1" group="1" contype="0" conaffinity="0"/>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    obs = SimpleNamespace(rgb=np.zeros((11, 11, 3), np.uint8), depth=np.ones((11, 11)),
                          camera_K=np.array([[100., 0., 5.], [0., 100., 5.], [0., 0., 1.]]), camera_pose=np.eye(4))
    material = SimpleNamespace(matches_surface=lambda points, **kwargs: np.zeros(len(points), bool))
    matches = measured_robot_surfaces(model, data, [obs], known_material=material)
    assert len(matches) == 121


def test_explicit_proxy_association_accepts_interiors_and_preserves_external_returns():
    model = mujoco.MjModel.from_xml_string('''<mujoco><asset>
      <mesh name="proxy" vertex="-.1 -.1 -.1  .1 -.1 -.1  -.1 .1 -.1  .1 .1 -.1
                                 -.1 -.1 .1   .1 -.1 .1   -.1 .1 .1   .1 .1 .1"/>
      </asset><worldbody><geom type="mesh" mesh="proxy" pos="0 0 1"/></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    depth = np.full((11, 11), .95)
    depth[5, 5] = .85
    obs = SimpleNamespace(rgb=np.zeros((11, 11, 3), np.uint8), depth=depth,
                          camera_K=np.array([[100., 0., 5.], [0., 100., 5.], [0., 0., 1.]]), camera_pose=np.eye(4))
    ordinary = measured_robot_surfaces(model, data, [obs])
    relaxed = measured_robot_surfaces(model, data, [obs], collision_proxy_self_association=True)
    assert len(ordinary) == 0
    assert len(relaxed) == 120
    assert np.all(relaxed[:, 2] == .95)


def test_proxy_association_does_not_absorb_returns_just_outside_robot():
    model = mujoco.MjModel.from_xml_string('''<mujoco><asset>
      <mesh name="proxy" vertex="-.1 -.1 -.1  .1 -.1 -.1  -.1 .1 -.1  .1 .1 -.1
                                 -.1 -.1 .1   .1 -.1 .1   -.1 .1 .1   .1 .1 .1"/>
      </asset><worldbody><geom type="mesh" mesh="proxy" pos="0 0 1"/></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    obs = SimpleNamespace(rgb=np.zeros((11, 11, 3), np.uint8), depth=np.ones((11, 11)),
        camera_K=np.array([[100., 0., 10.5], [0., 100., 5.], [0., 0., 1.]]), camera_pose=np.eye(4))
    matches = measured_robot_surfaces(model, data, [obs], tolerance_m=.015, collision_proxy_self_association=True)
    assert len(matches) == 110
    assert np.all(matches[:, 0] >= -.1)  # The returns 5 mm outside remain environment evidence.


def test_visual_triangle_matching_recovers_occluded_self_without_absorbing_obstacle():
    model = mujoco.MjModel.from_xml_string('''<mujoco><asset>
      <mesh name="visual" vertex="-.1 -.1 -.1  .1 -.1 -.1  -.1 .1 -.1  .1 .1 -.1
                                  -.1 -.1 .1   .1 -.1 .1   -.1 .1 .1   .1 .1 .1"/>
      </asset><worldbody>
      <geom type="mesh" mesh="visual" pos="0 0 1" group="1" contype="0" conaffinity="0"/>
      <geom type="mesh" mesh="visual" pos="0 0 2" group="1" contype="0" conaffinity="0"/>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    depth = np.full((5, 5), 1.9005)  # Sub-mm error at the rear visual surface.
    depth[2, 2] = 1.895  # Nearby obstacle remains outside the 3 mm surface band.
    obs = SimpleNamespace(rgb=np.zeros((5, 5, 3), np.uint8), depth=depth,
                          camera_K=np.array([[100., 0., 2.], [0., 100., 2.], [0., 0., 1.]]),
                          camera_pose=np.eye(4))
    material = SimpleNamespace(matches_surface=lambda points, **kwargs: np.zeros(len(points), bool))
    matches = measured_robot_surfaces(model, data, [obs], known_material=material)
    assert len(matches) == 24
    assert np.all(matches[:, 2] == 1.9005)
    # Explicit wider association is available to the scored simulator diagnostic,
    # but is not a claim that nearby real obstacles can be distinguished.
    expanded = measured_robot_surfaces(model, data, [obs], known_material=material,
                                      visual_surface_tolerance_m=.008)
    assert len(expanded) == 25
    assert len(measured_robot_surfaces(model, data, [obs], known_material=material,
        tolerance_m=.03, visual_surface_tolerance_m=.03)) == 25
    with pytest.raises(ValueError, match='invalid_visual_surface_tolerance'):
        measured_robot_surfaces(model, data, [obs], visual_surface_tolerance_m=.031)


def test_candidate_only_surface_qualification_matches_full_frame_subset():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
        <geom type="box" pos="0 0 1" size=".3 .3 .1"/>
        </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    depth = np.full((21, 21), .91)
    depth[10, 10] = .85
    obs = SimpleNamespace(rgb=np.zeros((21, 21, 3), np.uint8), depth=depth,
        camera_K=np.array([[100., 0., 10.], [0., 100., 10.], [0., 0., 1.]]), camera_pose=np.eye(4))
    mask = np.zeros(depth.shape, bool)
    mask[8:13, 8:13] = True
    full = measured_robot_surfaces(model, data, [obs])
    selected = measured_robot_surfaces(model, data, [obs], pixel_masks=[mask])
    assert len(selected) == 24  # The nearer obstacle remains excluded.
    assert all(np.any(np.all(full == point, axis=1)) for point in selected)
    for masks in ([], [np.zeros((2, 2), bool)], [np.zeros(depth.shape, int)]):
        with pytest.raises(ValueError, match='invalid_robot_surface_masks'):
            measured_robot_surfaces(model, data, [obs], pixel_masks=masks)


def test_ray_query_prepares_private_data_from_fk_only_capture(monkeypatch):
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
        <body pos="0 0 1"><joint type="slide" axis="0 0 1"/>
        <geom type="box" size=".3 .3 .1" mass="1"/></body>
        </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    data.qpos[0] = .2
    mujoco.mj_kinematics(model, data)
    before = data.qpos.copy(), data.geom_xpos.copy()
    obs = SimpleNamespace(rgb=np.zeros((3, 3, 3), np.uint8), depth=np.full((3, 3), 1.1),
        camera_K=np.array([[100., 0., 1.], [0., 100., 1.], [0., 0., 1.]]), camera_pose=np.eye(4))
    original = mujoco.mj_multiRay
    scratch_data = []

    def query(model, scratch, *args):
        assert scratch is not data
        scratch_data.append(scratch)
        return original(model, scratch, *args)

    monkeypatch.setattr(mujoco, 'mj_multiRay', query)
    for _ in range(2):
        assert len(measured_robot_surfaces(model, data, [obs])) == 9
    assert scratch_data[0] is not scratch_data[1]
    np.testing.assert_array_equal(data.qpos, before[0])
    np.testing.assert_array_equal(data.geom_xpos, before[1])
