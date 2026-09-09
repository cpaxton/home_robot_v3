# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Captured camera evidence, independent of graph objects and search proposals.

The integer tool adapter uses a separate positive namespace for voxel frames.
IDs reference actual retained frames, never fabricated object observations.
"""

from dataclasses import dataclass

import numpy as np

VIEW_ID_BASE = 1 << 40


@dataclass(frozen=True)
class CapturedView:
    obs_id: int
    source_obs_id: int
    rgb: np.ndarray
    camera_pose: np.ndarray | None
    camera_K: np.ndarray | None = None

    @property
    def view_id(self) -> str:
        return f"voxel-frame:{self.source_obs_id}"


def captured_view(executor, obs_id) -> CapturedView | None:
    return getattr(executor, "_captured_views", {}).get(obs_id)


def retain_latest_view(executor, *, after: int) -> CapturedView | None:
    from emet.memory.graph_eqa.ingest.instance_observations import frame_rgb_hwc_uint8

    vm = getattr(executor.agent, "voxel_map", None)
    frames = getattr(vm, "observations", ())
    if len(frames) <= after:
        return None
    frame = frames[-1]
    rgb = frame_rgb_hwc_uint8(frame)
    if rgb is None:
        return None
    source = len(frames)
    pose = getattr(frame, "camera_pose", None)
    if hasattr(pose, "detach"):
        pose = pose.detach().cpu().numpy()
    intrinsics = getattr(frame, "camera_K", None)
    if hasattr(intrinsics, "detach"):
        intrinsics = intrinsics.detach().cpu().numpy()
    view = CapturedView(
        VIEW_ID_BASE + source,
        source,
        rgb.copy(),
        None if pose is None else np.asarray(pose).copy(),
        None if intrinsics is None else np.asarray(intrinsics).copy(),
    )
    if not hasattr(executor, "_captured_views"):
        executor._captured_views = {}
    executor._captured_views[view.obs_id] = view
    # Evidence RGB is bounded per query; the voxel map owns the original frames.
    while len(executor._captured_views) > 32:
        del executor._captured_views[next(iter(executor._captured_views))]
    return view


def target_in_view(view: CapturedView, target_xyz) -> dict:
    """Project a world-space search anchor into the exact captured optical frame.

    In-frame is necessary but does not imply visibility through scene geometry,
    correct localization, or object verification.
    """
    if view.camera_pose is None or view.camera_K is None or target_xyz is None:
        return {"status": "missing_geometry", "target_in_frame": None}
    xyz = np.asarray(target_xyz, dtype=float).reshape(-1)[:3]
    camera_xyz = np.linalg.solve(view.camera_pose, np.append(xyz, 1.0))[:3]
    row = {
        "target_world_xyz": xyz.tolist(),
        "target_camera_xyz": camera_xyz.tolist(),
        "camera_target_distance_m": float(np.linalg.norm(camera_xyz)),
    }
    if camera_xyz[2] <= 0:
        return {**row, "status": "behind_camera", "target_in_frame": False}
    uvw = view.camera_K @ camera_xyz
    u, v = uvw[:2] / uvw[2]
    height, width = view.rgb.shape[:2]
    inside = bool(0 <= u < width and 0 <= v < height)
    return {
        **row,
        "target_pixel_xy": [float(u), float(v)],
        "target_in_frame": inside,
        "status": "in_frame" if inside else "outside_frame",
    }
