#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Record a payload prior from an isolated object asset or measured cylinder."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from emet.motion.observed_solids import WorldSolid
from emet.motion.placement_geometry import box_corners


def sample_surface(triangles, count):
    sides = triangles[:, 1:] - triangles[:, :1]
    cross = np.cross(sides[:, 0], sides[:, 1])
    area = np.linalg.norm(cross, axis=1)
    valid = area > 1e-12
    if not valid.any():
        raise ValueError("asset has no nondegenerate visual triangles")
    rng = np.random.default_rng(0)
    triangles, cross, area = triangles[valid], cross[valid], area[valid]
    indices = rng.choice(len(triangles), count, p=area / area.sum())
    u = np.sqrt(rng.random(count))
    v = rng.random(count)
    weights = np.stack((1 - u, u * (1 - v), u * v), axis=1)
    return np.einsum("bi,bij->bj", weights, triangles[indices]), cross[indices] / area[indices, None]


def isolated_asset_envelope(asset, *, root="object", annotation_geoms=(), visual_meshes=()):
    model = mujoco.MjModel.from_xml_path(str(asset))
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    object_root = model.body(root)
    descendants = {int(object_root.id)}
    for body_id in range(object_root.id + 1, model.nbody):
        if int(model.body_parentid[body_id]) in descendants:
            descendants.add(body_id)
    if model.nu or model.njnt or any(int(body) not in descendants for body in model.geom_bodyid):
        raise ValueError("payload calibration requires an isolated rigid asset")
    excluded = {int(model.geom(name).id) for name in annotation_geoms}
    if any(model.geom_contype[geom_id] or model.geom_conaffinity[geom_id] for geom_id in excluded):
        raise ValueError("a collision geom cannot be excluded as an annotation")
    rotation = data.xmat[object_root.id].reshape(3, 3)
    geometry = []
    visual_triangles = []
    seen_visual = set()
    for geom_id in range(model.ngeom):
        if geom_id in excluded:
            continue
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = int(model.geom_dataid[geom_id])
            start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
            vertices = model.mesh_vert[start : start + count]
            world = vertices @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
            mesh_name = model.mesh(mesh_id).name
            if mesh_name in visual_meshes:
                seen_visual.add(mesh_name)
                first, faces = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
                visual_triangles.append(((world - data.xpos[object_root.id]) @ rotation)[
                    model.mesh_face[first : first + faces]
                ])
        else:
            # A primitive's world-frame AABB is a conservative envelope.
            world = box_corners(WorldSolid.from_geom(model, data, geom_id).bounds)
        geometry.append((world - data.xpos[object_root.id]) @ rotation)
    if not geometry:
        raise ValueError("asset has no geometry evidence")
    points = np.concatenate(geometry)
    hull = ConvexHull(points)
    vertices = points[hull.vertices]
    geometry_digest = hashlib.sha256(np.ascontiguousarray(points).tobytes()).hexdigest()
    center = vertices.mean(axis=0)
    if not np.isfinite(vertices).all() or hull.volume <= 0:
        raise ValueError("payload envelope is degenerate")
    result = {
        "construction": "known_model_prior",
        "kind": "convex",
        "scope": "isolated visual and collision asset geometry in object-local coordinates",
        "asset_sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
        "geometry_sha256": geometry_digest,
        "excluded_annotation_geoms": list(annotation_geoms),
        "visual_meshes": list(visual_meshes),
        "vertices_reference_m": vertices.tolist(),
        "bounds_reference_m": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
        "nominal_reference_center_m": center.tolist(),
    }
    if set(visual_meshes) != seen_visual:
        raise ValueError("a declared visual mesh was not present in the asset")
    if visual_triangles:
        surface, normals = sample_surface(np.concatenate(visual_triangles), 12_000)
        result.update(surface_points_reference_m=surface.tolist(), surface_normals_reference=normals.tolist())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--asset", type=Path)
    source.add_argument("--cylinder", type=float, nargs=2, metavar=("RADIUS_M", "HEIGHT_M"))
    parser.add_argument("--exclude-annotation-geom", action="append", default=[], metavar="NAME")
    parser.add_argument("--visual-mesh", action="append", default=[], metavar="NAME")
    parser.add_argument("--calibration-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.asset:
        result = isolated_asset_envelope(
            args.asset, annotation_geoms=args.exclude_annotation_geom, visual_meshes=args.visual_mesh,
        )
    else:
        if args.exclude_annotation_geom or args.visual_mesh:
            parser.error("mesh and geom selections require an isolated asset")
        if not np.isfinite(args.cylinder).all() or min(args.cylinder) <= 0:
            parser.error("cylinder radius and height must be finite and positive")
        result = {
            "construction": "known_model_prior",
            "kind": "cylinder",
            "scope": "independently specified radius and height in meters",
            "dimensions_m": [args.cylinder[0], args.cylinder[1] / 2],
            "dimension_convention": "radius, half-height",
        }
    result.update(schema_version=1, calibration_id=args.calibration_id)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
