# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from emet.eval.native_pose_targets import horizontal_image_direction, pose_waypoints


def test_approach_rotation_progresses_gradually_and_reaches_requested_pose():
    target_rotation = Rotation.from_euler('z', 180, degrees=True).as_matrix()
    path = list(pose_waypoints(np.zeros(3), np.eye(3), np.array([.2, 0., 0.]), target_rotation))
    previous_position, previous_rotation = np.zeros(3), np.eye(3)
    for position, rotation in path:
        assert np.linalg.norm(position-previous_position) <= .020001
        assert Rotation.from_matrix(previous_rotation.T @ rotation).magnitude() <= .100001
        previous_position, previous_rotation = position, rotation
    np.testing.assert_allclose(path[-1][0], [.2, 0., 0.])
    np.testing.assert_allclose(path[-1][1], target_rotation, atol=1e-10)


def test_oblique_image_direction_roundtrips_through_horizontal_plane():
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler('x', 140, degrees=True).as_matrix()
    pose[:3, 3] = [0., 0., 2.]
    intrinsic = np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]])
    points = np.array([[.3, .4, 1.], [.31, .41, 1.]])
    camera_points = (points-pose[:3, 3]) @ pose[:3, :3]
    pixels = camera_points @ intrinsic.T
    pixels = pixels[:, :2]/pixels[:, 2:]
    direction = pixels[1]-pixels[0]
    angle = np.rad2deg(np.arctan2(direction[1], direction[0]))
    observation = SimpleNamespace(depth=np.ones((480, 640)), camera_K=intrinsic, camera_pose=pose)
    recovered = horizontal_image_direction(observation, pixels[0]/[640, 480]*1000, angle, 1.)
    np.testing.assert_allclose(recovered, np.array([1., 1.])/np.sqrt(2), atol=1e-8)
    naive = pose[:3, :3] @ [np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)), 0.]
    assert np.linalg.norm(naive[:2]/np.linalg.norm(naive[:2])-recovered) > .01


def test_grasp_center_ray_uses_estimated_interior_height_not_visible_surface():
    from emet.eval.native_pose_targets import image_point_at_height

    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler('x', 140, degrees=True).as_matrix()
    pose[:3, 3] = [0., 0., 2.]
    intrinsic = np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]])
    center = np.array([.2, .3, 1.])
    camera_point = (center-pose[:3, 3]) @ pose[:3, :3]
    pixel = intrinsic @ camera_point
    pixel = pixel[:2]/pixel[2]
    obs = SimpleNamespace(depth=np.ones((480, 640)), camera_K=intrinsic, camera_pose=pose)
    hint = pixel/[640, 480]*1000
    np.testing.assert_allclose(image_point_at_height(obs, hint, 1.), center, atol=1e-10)
    surface = image_point_at_height(obs, hint, 1.05)
    assert np.linalg.norm(surface[:2]-center[:2]) > .01


def test_hand_support_offset_uses_known_hand_geometry_not_object_geometry():
    import mujoco

    from emet.eval.native_pose_targets import hand_lowest_offset

    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="ee" pos="0 0 1">'
        '<geom name="finger" type="box" size=".01 .02 .03" pos="0 0 -.04"/>'
        '</body></worldbody></mujoco>')
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    assert abs(hand_lowest_offset(model, data, 'ee', ['finger'], np.eye(3))+.07) < 1e-10
    flipped = Rotation.from_euler('x', 180, degrees=True).as_matrix()
    assert abs(hand_lowest_offset(model, data, 'ee', ['finger'], flipped)-.01) < 1e-10
