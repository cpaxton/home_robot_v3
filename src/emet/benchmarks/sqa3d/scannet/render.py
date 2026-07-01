# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Open3D material and lighting helpers for ScanNet mesh replay."""

from __future__ import annotations

import open3d as o3d


def mesh_material(mesh: o3d.geometry.TriangleMesh) -> o3d.visualization.rendering.MaterialRecord:
    """Prefer vertex-color unlit shading when ScanNet PLY carries RGB."""
    mat = o3d.visualization.rendering.MaterialRecord()
    if mesh.has_vertex_colors():
        mat.shader = "defaultUnlit"
    else:
        mat.shader = "defaultLit"
        mat.base_color = [0.85, 0.82, 0.78, 1.0]
    return mat


def configure_scene_lighting(renderer: o3d.visualization.rendering.OffscreenRenderer) -> None:
    """Brighter indoor lighting for mesh-only replay."""
    renderer.scene.set_background([0.92, 0.92, 0.94, 1.0])
    try:
        renderer.scene.scene.set_sun_light([0.2, -0.4, -0.9], [1.0, 1.0, 1.0], 75000)
        renderer.scene.scene.enable_sun_light(True)
        renderer.scene.scene.set_indirect_light_intensity(25000.0)
    except AttributeError:
        pass
