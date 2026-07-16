# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Shared pipeline: voxel / saved FrameBlob → graph node adds + detection cache (live + offline)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np

from emet.memory.format import FrameBlob
from emet.memory.graph_eqa.graph_memory import GraphEQAMemory
from emet.memory.graph_eqa.instance_observations import frame_instances_to_detections


def dense_world_xyz_from_frame(fr: FrameBlob) -> np.ndarray | None:
    """Return HxWx3 world XYZ; use saved map or unproject from depth + intrinsics + pose."""
    if fr.depth is None or fr.camera_K is None or fr.camera_pose is None:
        return None
    wx = fr.world_xyz
    if wx is not None and np.asarray(wx).ndim == 3:
        return np.asarray(wx, dtype=np.float32)
    from emet.utils.image import Camera, camera_xyz_to_global_xyz

    d = np.asarray(fr.depth, dtype=np.float64)
    k = np.asarray(fr.camera_K, dtype=np.float64)
    pose = np.asarray(fr.camera_pose, dtype=np.float64)
    h, w = d.shape[:2]
    camera = Camera.from_K(k, width=w, height=h)
    cam_xyz = camera.depth_to_xyz(d)
    return np.asarray(camera_xyz_to_global_xyz(cam_xyz, pose), dtype=np.float32)


def frameblob_to_labels_xyz(
    fr: FrameBlob,
    *,
    min_depth: float,
    max_depth: float,
    detection_model: Any | None = None,
    min_points: int = 10,
) -> list[tuple[str, np.ndarray, tuple[int, int, int, int]]]:
    """Run instance→centroid logic on a saved ``FrameBlob`` (numpy arrays)."""
    if fr.depth is None:
        return []
    wx = dense_world_xyz_from_frame(fr)
    if wx is None:
        return []
    if fr.instance is None:
        return []

    fr = replace(fr, world_xyz=wx)

    class _Adapt:
        pass

    a = _Adapt()
    a.instance = fr.instance
    a.full_world_xyz = fr.world_xyz
    a.depth = fr.depth
    a.instance_classes = fr.instance_classes

    dets = frame_instances_to_detections(
        a,
        min_depth=min_depth,
        max_depth=max_depth,
        detection_model=detection_model,
        min_points=min_points,
    )
    return [
        (
            d["label_short"],
            np.asarray(d["xyz"], dtype=np.float64),
            tuple(d["bbox_xyxy"]),
        )
        for d in dets
    ]


def apply_instance_items_to_graph(
    graph_memory: GraphEQAMemory,
    rgb: np.ndarray,
    items: list[tuple[str, np.ndarray] | tuple[str, np.ndarray, tuple[int, int, int, int]]],
    *,
    dedup_skips: Callable[[str, np.ndarray], bool],
    viewer_xyz: np.ndarray | None = None,
    scene_profile: str | None = None,
) -> None:
    """Add one node per (label, xyz[, bbox_xyxy]) with optional dedup / scene label filter."""
    from emet.memory.graph_eqa.graph_label_filter import is_graph_label_allowed

    profile = scene_profile
    if profile is None:
        from emet.memory.graph_eqa.graph_label_filter import resolve_graph_scene_profile

        profile = resolve_graph_scene_profile(parameters=getattr(graph_memory, "parameters", None))
    for item in items:
        if len(item) >= 3:
            label, xyz, bbox_xyxy = item[0], item[1], item[2]
        else:
            label, xyz = item[0], item[1]
            bbox_xyxy = None
        if not is_graph_label_allowed(str(label), scene_profile=profile):
            continue
        if dedup_skips(label, xyz):
            continue
        graph_memory.add_observation(
            rgb, xyz, [label], viewer_xyz=viewer_xyz, bbox_xyxy=bbox_xyxy
        )


def build_detections_json_rows(
    fr: FrameBlob,
    *,
    min_depth: float,
    max_depth: float,
    detection_model: Any | None = None,
    min_points: int = 10,
) -> list[dict[str, Any]]:
    """Structured rows for ``detections_NNNN.json`` on disk."""
    if fr.depth is None or fr.instance is None:
        return []
    wx = dense_world_xyz_from_frame(fr)
    if wx is None:
        return []
    fr = replace(fr, world_xyz=wx)

    class _Adapt:
        pass

    a = _Adapt()
    a.instance = fr.instance
    a.full_world_xyz = fr.world_xyz
    a.depth = fr.depth
    a.instance_classes = fr.instance_classes

    return frame_instances_to_detections(
        a,
        min_depth=min_depth,
        max_depth=max_depth,
        detection_model=detection_model,
        min_points=min_points,
    )
