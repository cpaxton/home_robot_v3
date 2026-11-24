# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.
"""Robot pose and camera-plane conversions for observed native manipulation."""
import numpy as np
from scipy.spatial.transform import Rotation


def pose_waypoints(start, start_rotation, target, target_rotation, *, step_m=.02, step_rad=.1, min_steps=2):
    start, target = np.asarray(start, float), np.asarray(target, float)
    origin_rotation = np.asarray(start_rotation, float)
    delta = Rotation.from_matrix(origin_rotation.T @ target_rotation).as_rotvec()
    steps = max(min_steps, int(np.ceil(np.linalg.norm(target-start)/step_m)), int(np.ceil(np.linalg.norm(delta)/step_rad)))
    for fraction in np.linspace(0., 1., steps+1)[1:]:
        yield start+(target-start)*fraction, origin_rotation @ Rotation.from_rotvec(delta*fraction).as_matrix()


def horizontal_image_direction(observation, point_1000, angle_degrees, height_m):
    """Backproject an image direction onto the measured horizontal grasp plane."""
    h, w = observation.depth.shape
    pixel = np.asarray(point_1000, float)/1000*[w, h]
    angle = np.deg2rad(angle_degrees)
    pixels = np.array([pixel, pixel+np.array([np.cos(angle), np.sin(angle)])*5.])
    rays = np.column_stack((pixels, np.ones(2))) @ np.linalg.inv(observation.camera_K).T
    rays = rays @ observation.camera_pose[:3, :3].T
    origin = observation.camera_pose[:3, 3]
    if np.any(np.abs(rays[:, 2]) < 1e-6):
        raise ValueError('grasp_direction_plane_unobservable')
    distances = (height_m-origin[2])/rays[:, 2]
    if np.any(distances <= 0):
        raise ValueError('grasp_direction_plane_behind_camera')
    points = origin+rays*distances[:, None]
    delta = points[1, :2]-points[0, :2]
    if not np.isfinite(delta).all() or np.linalg.norm(delta) < 1e-8:
        raise ValueError('grasp_direction_plane_unobservable')
    return delta/np.linalg.norm(delta)


def image_point_at_height(observation, point_1000, height_m):
    """Interpret the semantic grasp-center pixel at an estimated interior height.

    A visible surface return is not the object's interior grasp center. This
    ray/plane estimate makes no CAD or completed-shape assumption.
    """
    h, w = observation.depth.shape
    pixel = np.asarray(point_1000, float)/1000*[w, h]
    ray = observation.camera_pose[:3, :3] @ np.linalg.solve(observation.camera_K, [*pixel, 1.])
    origin = observation.camera_pose[:3, 3]
    if abs(ray[2]) < 1e-6:
        raise ValueError('grasp_center_plane_unobservable')
    distance = (height_m-origin[2])/ray[2]
    if distance <= 0 or not np.isfinite(distance):
        raise ValueError('grasp_center_plane_behind_camera')
    return origin+distance*ray


def hand_lowest_offset(model, data, ee_body, geometry_names, rotation):
    """Lowest known hand collision point under a proposed orientation; no object model."""
    import itertools

    import mujoco

    ee = data.body(ee_body)
    local = []
    for name in geometry_names:
        geom = model.geom(name)
        kind = int(geom.type[0])
        if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh = int(geom.dataid[0])
            start = int(model.mesh_vertadr[mesh])
            vertices = model.mesh_vert[start:start+int(model.mesh_vertnum[mesh])]
        elif kind == int(mujoco.mjtGeom.mjGEOM_BOX):
            vertices = np.array(list(itertools.product((-1., 1.), repeat=3)))*geom.size
        else:
            raise ValueError('unsupported_hand_collision_geometry')
        placed = data.geom(name)
        world = vertices @ placed.xmat.reshape(3, 3).T+placed.xpos
        local.append((world-ee.xpos) @ ee.xmat.reshape(3, 3))
    if not local:
        raise ValueError('hand_collision_geometry_unavailable')
    return float((np.concatenate(local) @ rotation.T)[:, 2].min())
