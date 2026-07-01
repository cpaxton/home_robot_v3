# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Load ScanNet meshes for Open3D rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from emet.benchmarks.sqa3d.scannet.config import scene_mesh_path


def load_scannet_mesh(scene_id: str, scannet_root: Path | None = None) -> o3d.geometry.TriangleMesh:
    path = scene_mesh_path(scene_id, scannet_root)
    if not path.is_file():
        raise FileNotFoundError(
            f"ScanNet mesh not found for {scene_id}: {path}\n"
            "Run: uv run python scripts/download_scannet_data.py --accept-tos --scene "
            f"{scene_id}"
        )
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty():
        raise ValueError(f"Empty ScanNet mesh: {path}")
    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors)
        if colors.size and float(colors.max()) > 1.0 + 1e-3:
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors / 255.0)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    return mesh


def mesh_scene_bounds(mesh: o3d.geometry.TriangleMesh) -> tuple[np.ndarray, np.ndarray]:
    aabb = mesh.get_axis_aligned_bounding_box()
    return np.asarray(aabb.get_min_bound(), dtype=np.float64), np.asarray(aabb.get_max_bound(), dtype=np.float64)
