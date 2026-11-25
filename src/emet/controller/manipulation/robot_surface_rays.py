# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Associate measured depth with a robot-only model at capture-aligned joints."""

import mujoco
import numpy as np
from scipy.spatial import cKDTree
from trimesh.triangles import closest_point

from emet.controller.task.tamp.observed_capture import capture_arrays
from emet.motion.observed_solids import WorldSolid


def measured_robot_surfaces(model, data, observations, *, tolerance_m=.015, known_material=None,
                            collision_proxy_self_association=False, visual_surface_tolerance_m=.003, pixel_masks=None):
    if not np.isfinite(tolerance_m) or not 0 <= tolerance_m <= .03:
        raise ValueError('invalid_robot_surface_tolerance')
    if not np.isfinite(visual_surface_tolerance_m) or not 0 <= visual_surface_tolerance_m <= .03:
        raise ValueError('invalid_visual_surface_tolerance')
    # Ray queries use position-dependent BVH/stack state, not just geom poses.
    # Captures may contain FK-only data, and mapping/tracking can query one
    # capture concurrently. Prepare private scratch; never mutate or share the
    # caller's captured mjData through native ray operations.
    scratch = mujoco.MjData(model)
    mujoco.mj_copyData(scratch, model, data)
    mujoco.mj_fwdPosition(model, scratch)
    data = scratch
    matched = []
    groups = None
    if known_material is not None:
        mesh = (model.geom_type == mujoco.mjtGeom.mjGEOM_MESH) & (model.geom_contype == 0) & (model.geom_conaffinity == 0)
        if not np.any(mesh):
            mesh = (model.geom_type == mujoco.mjtGeom.mjGEOM_MESH) & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
        groups = np.zeros(6, dtype=np.uint8)
        groups[np.unique(model.geom_group[mesh])] = 1
    observations = tuple(observations)
    if pixel_masks is not None and len(pixel_masks) != len(observations):
        raise ValueError('invalid_robot_surface_masks')
    for index, observation in enumerate(observations):
        _, depth, world = capture_arrays(observation)
        valid = np.isfinite(depth) & (depth >= .1) & (depth <= 5.)
        if pixel_masks is not None:
            mask = np.asarray(pixel_masks[index])
            if mask.dtype != bool or mask.shape != depth.shape:
                raise ValueError('invalid_robot_surface_masks')
            valid &= mask
        points = world[valid]
        if known_material is not None:
            primitive = known_material.matches_surface(points, point_error_m=tolerance_m)
            matched.append(points[primitive])
            points = points[~primitive]
        if collision_proxy_self_association:
            interior = np.zeros(len(points), dtype=bool)
            for gid in range(model.ngeom):
                if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH or not (
                    model.geom_contype[gid] or model.geom_conaffinity[gid]
                ):
                    continue
                proxy = WorldSolid.from_geom(model, data, gid, convex_material_calibration_id='robot-model-self-association')
                # A collision proxy may explain returns inside captured robot
                # geometry, but must not absorb nearby external obstacles.
                # Surface uncertainty is handled by the visual-ray test below.
                bounds = proxy.bounds
                candidates = np.all((points >= bounds[0]) & (points <= bounds[1]), axis=1)
                interior[candidates] |= proxy.lower_distance(points[candidates]) <= 0.
            matched.append(points[interior])
            points = points[~interior]
        candidate = np.zeros(len(points), dtype=bool)
        for gid in range(model.ngeom):
            if groups is not None and not groups[model.geom_group[gid]]:
                continue
            rotation = data.geom_xmat[gid].reshape(3, 3)
            center = rotation @ model.geom_aabb[gid, :3] + data.geom_xpos[gid]
            half = np.abs(rotation) @ model.geom_aabb[gid, 3:] + tolerance_m
            low, high = center - half, center + half
            inside = (points[:, 0] >= low[0]) & (points[:, 0] <= high[0])
            inside &= (points[:, 1] >= low[1]) & (points[:, 1] <= high[1])
            inside &= (points[:, 2] >= low[2]) & (points[:, 2] <= high[2])
            candidate |= inside
        points = points[candidate]
        if not len(points):
            continue
        origin = np.asarray(observation.camera_pose[:3, 3], float)
        rays = points - origin
        lengths = np.linalg.norm(rays, axis=1)
        rays = np.ascontiguousarray(rays / lengths[:, None])
        ids = np.full(len(rays), -1, dtype=np.int32)
        distances = np.full(len(rays), -1., dtype=float)
        mujoco.mj_multiRay(model, data, origin, rays.ravel(), groups, 1, -1, ids, distances,
                          None, len(rays), 5.)
        belongs = (ids >= 0) & (distances >= 0) & (np.abs(lengths - distances) <= tolerance_m)
        # A wrist camera can lie inside a visual mesh, or a foreground robot
        # link can occlude another robot surface in the collision ray model.
        # Test the actual visual triangles for missed returns; never use a
        # convex hull or expand a collision proxy. The default 3 mm cap deliberately
        # stays much smaller than the ordinary ray association tolerance.
        surface_band = min(tolerance_m, visual_surface_tolerance_m)
        for gid in range(model.ngeom):
            if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH or (
                model.geom_contype[gid] or model.geom_conaffinity[gid]
            ):
                continue
            rotation = data.geom_xmat[gid].reshape(3, 3)
            local = (points - data.geom_xpos[gid]) @ rotation
            mesh = int(model.geom_dataid[gid])
            va, vn = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            fa, fn = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
            vertices = model.mesh_vert[va:va+vn]
            ids = np.flatnonzero(~belongs & np.all(
                (local >= vertices.min(0)-surface_band) &
                (local <= vertices.max(0)+surface_band), axis=1))
            if not len(ids) or not fn:
                continue
            triangles = vertices[model.mesh_face[fa:fa+fn]]
            # Nearby triangle centres propose candidates only. A match always
            # requires an exact point-on-triangle distance. Missing a proposal
            # retains the return as an obstacle rather than claiming ownership.
            _, choices = cKDTree(triangles.mean(1)).query(local[ids], k=min(64, fn))
            choices = np.asarray(choices).reshape(len(ids), -1)
            query = np.repeat(local[ids], choices.shape[1], axis=0)
            nearest = closest_point(triangles[choices.ravel()], query)
            distance = np.linalg.norm(nearest-query, axis=1).reshape(choices.shape)
            belongs[ids] |= np.any(distance <= surface_band, axis=1)
        matched.append(points[belongs])
    return np.concatenate(matched) if matched else np.empty((0, 3))
