# Copyright (c) Allen Institute for AI (MolmoSpaces). Apache-2.0.
# Vendored from molmo_spaces/utils/mj_model_and_data_utils.py (geom_aabb + mesh_aabb + body_pose).

from __future__ import annotations

import itertools

import mujoco
import numpy as np

from emet.simulation.molmo_occupancy._pose import pos_quat_to_pose_mat


def body_pose(data: mujoco.MjData, body_id: int) -> np.ndarray:
    trf = np.eye(4)
    trf[:3, 3] = data.xpos[body_id]
    trf[:3, :3] = data.xmat[body_id].reshape(3, 3)
    return trf


def mesh_aabb(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> tuple[np.ndarray, np.ndarray]:
    assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH.value
    mesh_id = model.geom_dataid[geom_id]
    vertadr = model.mesh_vertadr[mesh_id]
    n_vert = model.mesh_vertnum[mesh_id]
    geom_rel_pose = pos_quat_to_pose_mat(model.geom_pos[geom_id], model.geom_quat[geom_id])
    geom_body_id = model.geom_bodyid[geom_id]
    geom_pose = body_pose(data, geom_body_id) @ geom_rel_pose
    vertices_local = model.mesh_vert[vertadr : vertadr + n_vert]
    vertices = vertices_local @ geom_pose[:3, :3].T + geom_pose[:3, 3]
    aabb_min = np.min(vertices, axis=0)
    aabb_max = np.max(vertices, axis=0)
    return (aabb_min + aabb_max) / 2, aabb_max - aabb_min


def geom_aabb(
    model: mujoco.MjModel, data: mujoco.MjData, geom_ids: list[int], tight_mesh: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    if not geom_ids:
        return np.zeros(3), np.zeros(3)
    corners = np.array(list(itertools.product([-1.0, 1.0], repeat=3)))
    vertices: list[np.ndarray] = []
    for geom_id in geom_ids:
        if tight_mesh and model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH.value:
            mesh_aabb_center, mesh_aabb_size = mesh_aabb(model, data, geom_id)
            vertices.append(mesh_aabb_center + corners * mesh_aabb_size / 2)
        else:
            geom_rotmat = data.geom_xmat[geom_id].reshape(3, 3)
            geom_pos = data.geom_xpos[geom_id]
            aabb = model.geom_aabb[geom_id]
            local_corners = aabb[:3] + corners * aabb[3:]
            world_corners = local_corners @ geom_rotmat.T + geom_pos
            vertices.append(world_corners)
    vertices_arr = np.concatenate(vertices, axis=0)
    merged_min = np.min(vertices_arr, axis=0)
    merged_max = np.max(vertices_arr, axis=0)
    return (merged_min + merged_max) / 2, merged_max - merged_min
