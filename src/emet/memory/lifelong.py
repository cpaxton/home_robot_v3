# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Lifelong checkpoint load/save and local start-pose refinement.

Assumes external/sim pose is good enough to start, then optionally estimates a
small SE(2)/SE(3) correction between a saved map cloud and a short live scan so
imperfect spawn / localization does not require a perfect match.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from emet.memory.format import OPEN_VOCAB_SCENE_GRAPH_DIR, VOXEL_PICKLE_FILENAME, is_memory_directory
from emet.utils.logger import Logger
from emet.utils.point_cloud import ransac_transform

logger = Logger(__name__)

# Default gates: local fudge only (not kidnapped-robot global search).
DEFAULT_MAX_XY_M = 0.75
DEFAULT_MAX_YAW_RAD = 0.5  # ~28.6°
DEFAULT_MIN_FITNESS = 0.15
DEFAULT_MAX_RMSE_M = 0.35
DEFAULT_MIN_POINTS = 64
DEFAULT_DISTANCE_THRESHOLD_M = 0.25


@dataclass(frozen=True)
class RefineResult:
    """Outcome of a local start-pose refine attempt."""

    accepted: bool
    transform: np.ndarray  # (4, 4) maps saved → live
    fitness: float
    inlier_rmse: float
    num_inliers: int
    translation_xy_m: float
    yaw_rad: float
    reason: str

    @property
    def identity(self) -> bool:
        return bool(np.allclose(self.transform, np.eye(4), atol=1e-8))


def se3_translation_xy_yaw(transform: np.ndarray) -> tuple[float, float]:
    """Return planar translation norm (m) and yaw (rad) from a 4×4 matrix."""
    t = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    xy = float(np.linalg.norm(t[:2, 3]))
    # yaw from rotation about +Z
    yaw = float(np.arctan2(t[1, 0], t[0, 0]))
    return xy, yaw


def se2_matrix(x: float, y: float, yaw: float) -> np.ndarray:
    """Build a planar SE(3) transform (z=0) from ``(x, y, yaw)``."""
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    t = np.eye(4, dtype=np.float64)
    t[0, 0], t[0, 1] = c, -s
    t[1, 0], t[1, 1] = s, c
    t[0, 3], t[1, 3] = float(x), float(y)
    return t


def transform_points_xyz(xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply 4×4 ``transform`` to Nx3 points (homogeneous)."""
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    t = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    if pts.size == 0:
        return pts.copy()
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    homo = np.concatenate([pts, ones], axis=1)
    out = (t @ homo.T).T[:, :3]
    return out


def transform_pose_matrix(pose: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Left-multiply a 4×4 camera/base pose by ``transform``."""
    p = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    t = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return t @ p


def _as_numpy_xyz(points: Any) -> np.ndarray | None:
    if points is None:
        return None
    if hasattr(points, "detach"):
        points = points.detach().cpu().numpy()
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim == 1 and arr.size >= 3:
        return arr.reshape(1, 3)[:, :3]
    if arr.ndim != 2 or arr.shape[1] < 3 or arr.shape[0] == 0:
        return None
    return arr[:, :3].copy()


def voxel_semantic_xyz(voxel_map: Any) -> np.ndarray | None:
    """Best-effort XYZ from a DynaMem / SparseVoxelMap semantic or voxel cloud."""
    if voxel_map is None:
        return None
    sm = getattr(voxel_map, "semantic_memory", None)
    if sm is not None and getattr(sm, "_points", None) is not None:
        return _as_numpy_xyz(sm._points)
    if hasattr(voxel_map, "get_pointcloud"):
        try:
            out = voxel_map.get_pointcloud()
            if isinstance(out, (tuple, list)) and len(out) >= 1:
                return _as_numpy_xyz(out[0])
        except Exception:
            pass
    vp = getattr(voxel_map, "voxel_pcd", None)
    if vp is not None and getattr(vp, "_points", None) is not None:
        return _as_numpy_xyz(vp._points)
    return None


def refine_start_pose(
    saved_xyz: np.ndarray,
    live_xyz: np.ndarray,
    *,
    max_xy_m: float = DEFAULT_MAX_XY_M,
    max_yaw_rad: float = DEFAULT_MAX_YAW_RAD,
    min_fitness: float = DEFAULT_MIN_FITNESS,
    max_rmse_m: float = DEFAULT_MAX_RMSE_M,
    min_points: int = DEFAULT_MIN_POINTS,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD_M,
    visualize: bool = False,
    grid_xy_step_m: float = 0.15,
    grid_yaw_step_rad: float = 0.15,
) -> RefineResult:
    """Estimate a local transform mapping ``saved_xyz`` → ``live_xyz``.

    Tries a coarse SE(2) grid around identity (within the acceptance caps), then
    ICP from each seed. Acceptance gates keep this a **fudge** (small residual),
    not global place recognition. On failure returns identity with
    ``accepted=False`` so callers keep the assumed pose.
    """
    identity = np.eye(4, dtype=np.float64)
    saved = _as_numpy_xyz(saved_xyz)
    live = _as_numpy_xyz(live_xyz)
    if saved is None or live is None:
        return RefineResult(
            accepted=False,
            transform=identity,
            fitness=0.0,
            inlier_rmse=float("inf"),
            num_inliers=0,
            translation_xy_m=0.0,
            yaw_rad=0.0,
            reason="missing_point_cloud",
        )
    if saved.shape[0] < int(min_points) or live.shape[0] < int(min_points):
        return RefineResult(
            accepted=False,
            transform=identity,
            fitness=0.0,
            inlier_rmse=float("inf"),
            num_inliers=0,
            translation_xy_m=0.0,
            yaw_rad=0.0,
            reason=f"too_few_points saved={saved.shape[0]} live={live.shape[0]} min={min_points}",
        )

    # Subsample for speed on large clouds.
    max_n = 2000
    if saved.shape[0] > max_n:
        idx = np.linspace(0, saved.shape[0] - 1, max_n).astype(int)
        saved = saved[idx]
    if live.shape[0] > max_n:
        idx = np.linspace(0, live.shape[0] - 1, max_n).astype(int)
        live = live[idx]

    xy_steps = np.arange(-float(max_xy_m), float(max_xy_m) + 1e-9, float(grid_xy_step_m))
    if xy_steps.size == 0:
        xy_steps = np.array([0.0])
    yaw_steps = np.arange(-float(max_yaw_rad), float(max_yaw_rad) + 1e-9, float(grid_yaw_step_rad))
    if yaw_steps.size == 0:
        yaw_steps = np.array([0.0])

    best: RefineResult | None = None
    for dx in xy_steps:
        for dy in xy_steps:
            if float(np.hypot(dx, dy)) > float(max_xy_m) + 1e-9:
                continue
            for dyaw in yaw_steps:
                init = se2_matrix(float(dx), float(dy), float(dyaw))
                seeded = transform_points_xyz(saved, init)
                t_delta, fitness, inlier_rmse, num_inliers = ransac_transform(
                    seeded,
                    live,
                    visualize=False,
                    distance_threshold=float(distance_threshold),
                )
                # Composition: init then ICP delta (ICP maps seeded→live)
                t = np.asarray(t_delta, dtype=np.float64).reshape(4, 4) @ init
                xy_m, yaw = se3_translation_xy_yaw(t)
                fit = float(fitness)
                rmse = float(inlier_rmse)
                n_in = int(num_inliers)
                cand = RefineResult(
                    accepted=True,
                    transform=t,
                    fitness=fit,
                    inlier_rmse=rmse,
                    num_inliers=n_in,
                    translation_xy_m=xy_m,
                    yaw_rad=yaw,
                    reason="ok",
                )
                if best is None or (fit > best.fitness) or (abs(fit - best.fitness) < 1e-6 and rmse < best.inlier_rmse):
                    best = cand

    if best is None:
        return RefineResult(
            accepted=False,
            transform=identity,
            fitness=0.0,
            inlier_rmse=float("inf"),
            num_inliers=0,
            translation_xy_m=0.0,
            yaw_rad=0.0,
            reason="no_grid_candidates",
        )

    fit = best.fitness
    rmse = best.inlier_rmse
    xy_m = best.translation_xy_m
    yaw = best.yaw_rad
    t = best.transform

    if fit < float(min_fitness):
        reason = f"fitness {fit:.3f} < {min_fitness}"
        accepted = False
    elif rmse > float(max_rmse_m):
        reason = f"rmse {rmse:.3f} m > {max_rmse_m}"
        accepted = False
    elif xy_m > float(max_xy_m):
        reason = f"translation_xy {xy_m:.3f} m > {max_xy_m}"
        accepted = False
    elif abs(yaw) > float(max_yaw_rad):
        reason = f"yaw {yaw:.3f} rad > {max_yaw_rad}"
        accepted = False
    else:
        reason = "ok"
        accepted = True

    if not accepted:
        t = identity
    if visualize and accepted:
        ransac_transform(saved, live, visualize=True, distance_threshold=float(distance_threshold))

    return RefineResult(
        accepted=accepted,
        transform=t,
        fitness=fit,
        inlier_rmse=rmse,
        num_inliers=best.num_inliers,
        translation_xy_m=xy_m,
        yaw_rad=yaw,
        reason=reason,
    )


def _transform_bounds_3d(bounds: dict[str, Any] | None, transform: np.ndarray) -> dict[str, Any] | None:
    if not bounds or "min" not in bounds or "max" not in bounds:
        return bounds
    mn = np.asarray(bounds["min"], dtype=np.float64).reshape(3)
    mx = np.asarray(bounds["max"], dtype=np.float64).reshape(3)
    # Transform the eight corners, then re-AABB (rotation-safe for small yaw).
    corners = np.array(
        [
            [mn[0], mn[1], mn[2]],
            [mn[0], mn[1], mx[2]],
            [mn[0], mx[1], mn[2]],
            [mn[0], mx[1], mx[2]],
            [mx[0], mn[1], mn[2]],
            [mx[0], mn[1], mx[2]],
            [mx[0], mx[1], mn[2]],
            [mx[0], mx[1], mx[2]],
        ],
        dtype=np.float64,
    )
    tc = transform_points_xyz(corners, transform)
    new_min = tc.min(axis=0)
    new_max = tc.max(axis=0)
    center = 0.5 * (new_min + new_max)
    size = new_max - new_min
    return {
        "min": new_min.tolist(),
        "max": new_max.tolist(),
        "center": center.tolist(),
        "size": size.tolist(),
    }


def _transform_xyz_list_field(value: Any, transform: np.ndarray) -> Any:
    """Transform a length-3 xyz list/array; return ``value`` unchanged if not xyz-like."""
    if value is None:
        return value
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return value
    if arr.size < 3:
        return value
    out = transform_points_xyz(arr[:3].reshape(1, 3), transform)[0]
    if isinstance(value, list):
        return out.tolist()
    return out


def _transform_change_event(event: dict[str, Any], transform: np.ndarray) -> dict[str, Any]:
    """Copy a belief/change event dict with spatial fields mapped by ``transform``."""
    out = dict(event)
    for key in ("xyz", "from_xyz", "to_xyz", "last_xyz"):
        if key in out:
            out[key] = _transform_xyz_list_field(out[key], transform)
    return out


def _transform_position_covariance(cov: Any, transform: np.ndarray) -> np.ndarray | None:
    if cov is None:
        return None
    c = np.asarray(cov, dtype=np.float64)
    if c.shape != (3, 3):
        return c
    r = np.asarray(transform, dtype=np.float64).reshape(4, 4)[:3, :3]
    return r @ c @ r.T


def apply_se2_to_graph(graph_memory: Any, transform: np.ndarray) -> int:
    """Transform graph node / observation XYZ (and AABB bounds) by ``transform``.

    Also remaps belief sidecars from main (``position_history``, ``change_events``,
    ``position_covariance``) so a --refine-start fudge stays consistent with the
    new uncertain-track fields.

    Returns the number of object/viewpoint nodes updated.
    """
    if graph_memory is None:
        return 0
    t = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    if np.allclose(t, np.eye(4), atol=1e-10):
        return 0

    nodes = list(getattr(graph_memory, "_nodes", None) or graph_memory.get_nodes())
    new_nodes = []
    n_updated = 0
    for n in nodes:
        xyz = transform_points_xyz(np.asarray(n.xyz, dtype=np.float64).reshape(1, 3), t)[0]
        bounds = _transform_bounds_3d(getattr(n, "bounds_3d", None), t)
        history = [
            _transform_change_event(dict(entry), t) for entry in (getattr(n, "position_history", None) or [])
        ]
        changes = [
            _transform_change_event(dict(entry), t) for entry in (getattr(n, "change_events", None) or [])
        ]
        cov = _transform_position_covariance(getattr(n, "position_covariance", None), t)
        extent = getattr(n, "extent_half", None)
        if bounds is not None and bounds.get("size") is not None:
            size = np.asarray(bounds["size"], dtype=np.float64).reshape(3)
            extent = 0.5 * size
        kwargs: dict[str, Any] = {
            "xyz": xyz,
            "bounds_3d": bounds,
        }
        # Only pass fields that exist on this GraphNode version (frozen replace).
        fields = getattr(type(n), "__dataclass_fields__", {})
        if "position_history" in fields:
            kwargs["position_history"] = history
        if "change_events" in fields:
            kwargs["change_events"] = changes
        if "position_covariance" in fields:
            kwargs["position_covariance"] = cov
        if "extent_half" in fields and extent is not None:
            kwargs["extent_half"] = np.asarray(extent, dtype=np.float64).reshape(3)
        new_nodes.append(replace(n, **kwargs))
        n_updated += 1
    if hasattr(graph_memory, "_nodes"):
        graph_memory._nodes = new_nodes
    elif hasattr(graph_memory, "set_nodes"):
        graph_memory.set_nodes(new_nodes)

    observations = getattr(graph_memory, "_observations", None)
    if observations:
        for o in observations:
            if getattr(o, "xyz", None) is not None:
                o.xyz = transform_points_xyz(np.asarray(o.xyz, dtype=np.float64).reshape(1, 3), t)[0]
            if getattr(o, "viewer_xyz", None) is not None:
                o.viewer_xyz = transform_points_xyz(np.asarray(o.viewer_xyz, dtype=np.float64).reshape(1, 3), t)[0]

    mem_events = getattr(graph_memory, "_change_events", None)
    if isinstance(mem_events, list) and mem_events:
        graph_memory._change_events = [_transform_change_event(dict(e), t) for e in mem_events]

    rebuild = getattr(graph_memory, "_rebuild_viewpoint_index", None)
    if callable(rebuild):
        rebuild()
    # Spatial relations (near/on/contains) depend on node XYZ; rebuild when available.
    update_edges = getattr(graph_memory, "_update_edges", None)
    if callable(update_edges):
        update_edges()
    return n_updated


def _transform_frame_base_pose(raw_bp: Any, transform: np.ndarray) -> Any:
    """Map a Frame ``base_pose`` (4×4 or planar xyt) by ``transform``."""
    shape = getattr(raw_bp, "shape", None)
    if shape is not None and tuple(shape) == (4, 4):
        out = transform_pose_matrix(raw_bp, transform)
        if hasattr(raw_bp, "detach"):
            import torch

            return torch.as_tensor(out, device=raw_bp.device, dtype=raw_bp.dtype)
        return out
    bp = np.asarray(raw_bp, dtype=np.float64).reshape(-1)
    if bp.size < 3:
        return raw_bp
    # DynaMem stores planar (x, y, theta)
    xy = transform_points_xyz(np.array([[bp[0], bp[1], 0.0]]), transform)[0]
    _, dyaw = se3_translation_xy_yaw(transform)
    out = np.array([xy[0], xy[1], float(bp[2]) + dyaw], dtype=np.float64)
    if hasattr(raw_bp, "detach"):
        import torch

        return torch.as_tensor(out, device=raw_bp.device, dtype=raw_bp.dtype)
    return out


def _transform_frame_xyz_field(val: Any, transform: np.ndarray) -> Any:
    if val is None:
        return None
    arr_np = _as_numpy_xyz(val) if getattr(val, "ndim", 0) == 2 else None
    if arr_np is None:
        arr = np.asarray(val, dtype=np.float64)
        if arr.ndim == 1 and arr.size >= 3:
            out = transform_points_xyz(arr.reshape(1, 3), transform)[0]
        elif arr.ndim == 2 and arr.shape[1] >= 3:
            out = transform_points_xyz(arr[:, :3], transform)
        else:
            return val
    else:
        out = transform_points_xyz(arr_np, transform)
    if hasattr(val, "detach"):
        import torch

        return torch.as_tensor(out, device=val.device, dtype=val.dtype)
    return out


def _replace_or_set_frame(fr: Any, **updates: Any) -> Any:
    """Return an updated Frame. SparseVoxelMap uses immutable ``namedtuple`` Frames."""
    if hasattr(fr, "_replace"):
        # namedtuple / dataclass replace — only known fields
        fields = getattr(fr, "_fields", None)
        if fields is not None:
            updates = {k: v for k, v in updates.items() if k in fields}
        return fr._replace(**updates)
    for key, value in updates.items():
        try:
            setattr(fr, key, value)
        except AttributeError:
            continue
    return fr


def apply_se2_to_voxel_map(voxel_map: Any, transform: np.ndarray) -> bool:
    """In-place transform of DynaMem / SparseVoxelMap clouds and observation poses."""
    if voxel_map is None:
        return False
    t = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    if np.allclose(t, np.eye(4), atol=1e-10):
        return False

    sm = getattr(voxel_map, "semantic_memory", None)
    if sm is not None and getattr(sm, "_points", None) is not None:
        pts = _as_numpy_xyz(sm._points)
        if pts is not None:
            transformed = transform_points_xyz(pts, t)
            if hasattr(sm._points, "detach"):
                import torch

                sm._points = torch.as_tensor(transformed, device=sm._points.device, dtype=sm._points.dtype)
            else:
                sm._points = transformed
            if hasattr(sm, "_mins"):
                sm._mins = sm._points.min(dim=0).values if hasattr(sm._points, "min") else transformed.min(axis=0)
            if hasattr(sm, "_maxs"):
                sm._maxs = sm._points.max(dim=0).values if hasattr(sm._points, "max") else transformed.max(axis=0)

    vp = getattr(voxel_map, "voxel_pcd", None)
    if vp is not None and getattr(vp, "_points", None) is not None:
        pts = _as_numpy_xyz(vp._points)
        if pts is not None:
            transformed = transform_points_xyz(pts, t)
            if hasattr(vp._points, "detach"):
                import torch

                vp._points = torch.as_tensor(transformed, device=vp._points.device, dtype=vp._points.dtype)
            else:
                vp._points = transformed

    observations = getattr(voxel_map, "observations", None)
    if observations:
        new_obs = []
        for fr in observations:
            updates: dict[str, Any] = {}
            if getattr(fr, "camera_pose", None) is not None:
                cam = transform_pose_matrix(fr.camera_pose, t)
                if hasattr(fr.camera_pose, "detach"):
                    import torch

                    cam = torch.as_tensor(cam, device=fr.camera_pose.device, dtype=fr.camera_pose.dtype)
                updates["camera_pose"] = cam
            if getattr(fr, "base_pose", None) is not None:
                updates["base_pose"] = _transform_frame_base_pose(fr.base_pose, t)
            for attr in ("xyz", "full_world_xyz", "world_xyz"):
                val = getattr(fr, attr, None)
                if val is None:
                    continue
                updates[attr] = _transform_frame_xyz_field(val, t)
            new_obs.append(_replace_or_set_frame(fr, **updates) if updates else fr)
        # Prefer writing back a list (namedtuple Frames); some maps keep a mutable list.
        try:
            voxel_map.observations = new_obs
        except AttributeError:
            observations[:] = new_obs

    return True


def apply_se2_to_memory(
    *,
    graph_memory: Any | None = None,
    voxel_map: Any | None = None,
    transform: np.ndarray,
) -> dict[str, Any]:
    """Apply one SE(2)/SE(3) transform to both graph and voxel memory."""
    n_nodes = apply_se2_to_graph(graph_memory, transform) if graph_memory is not None else 0
    voxel_ok = apply_se2_to_voxel_map(voxel_map, transform) if voxel_map is not None else False
    ov_n = 0
    sg = None
    if voxel_map is not None and hasattr(voxel_map, "get_scene_graph"):
        sg = voxel_map.get_scene_graph()
    if sg is not None:
        ov_n = apply_se2_to_open_vocab_scene_graph(sg, transform)
    xy_m, yaw = se3_translation_xy_yaw(transform)
    return {
        "nodes_updated": int(n_nodes),
        "voxel_transformed": bool(voxel_ok),
        "open_vocab_nodes_updated": int(ov_n),
        "translation_xy_m": float(xy_m),
        "yaw_rad": float(yaw),
    }


def _voxel_map_from_controller(controller: Any) -> Any | None:
    if hasattr(controller, "get_voxel_map"):
        return controller.get_voxel_map()
    return getattr(controller, "voxel_map", None)


def open_vocab_scene_graph_from_controller(controller: Any) -> Any | None:
    """Return the live OpenVocabSceneGraph if attached, else None."""
    vm = _voxel_map_from_controller(controller)
    if vm is not None and hasattr(vm, "get_scene_graph"):
        sg = vm.get_scene_graph()
        if sg is not None:
            return sg
    proc = getattr(controller, "_open_vocab_sg_processor", None)
    if proc is not None:
        return getattr(proc, "scene_graph", None)
    return None


def apply_se2_to_open_vocab_scene_graph(scene_graph: Any, transform: np.ndarray) -> int:
    """Transform open-vocab node centers / point clouds in place. Returns nodes touched."""
    nodes = getattr(scene_graph, "nodes", None)
    if not nodes:
        return 0
    t = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    n_updated = 0
    for node in nodes.values():
        touched = False
        center = getattr(node, "center", None)
        if center is not None:
            c = np.asarray(center, dtype=np.float64).reshape(-1)
            if c.size >= 3 and np.isfinite(c[:3]).all():
                node.center = transform_points_xyz(c[:3].reshape(1, 3), t)[0]
                touched = True
        pts = getattr(node, "point_cloud", None)
        if pts is not None:
            import torch

            if hasattr(pts, "detach"):
                arr = pts.detach().cpu().numpy()
            else:
                arr = np.asarray(pts)
            if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] >= 3:
                out = transform_points_xyz(arr[:, :3], t)
                if hasattr(pts, "detach"):
                    node.point_cloud = torch.as_tensor(out, dtype=pts.dtype)
                else:
                    node.point_cloud = out
                touched = True
                # Recompute AABB if present.
                if getattr(node, "bounds", None) is not None and hasattr(node, "point_cloud"):
                    pc = node.point_cloud
                    if hasattr(pc, "min"):
                        node.bounds = torch.stack([pc.min(dim=0).values, pc.max(dim=0).values], dim=0)
        if touched:
            n_updated += 1
    if n_updated and hasattr(scene_graph, "update_edges"):
        try:
            scene_graph.update_edges()
        except Exception:
            pass
    return n_updated


def _patch_manifest_open_vocab(path: Path, *, present: bool) -> None:
    import json

    from emet.memory.format import MANIFEST_FILENAME

    man_path = path / MANIFEST_FILENAME
    if not man_path.is_file():
        return
    try:
        data = json.loads(man_path.read_text(encoding="utf-8"))
    except Exception:
        return
    data["has_open_vocab_scene_graph"] = bool(present)
    man_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def save_open_vocab_scene_graph_sidecar(controller_or_sg: Any, path: str | Path) -> bool:
    """Write ``open_vocab_scene_graph/`` under *path* when an OV graph with objects exists."""
    path_obj = Path(path)
    sg = controller_or_sg
    if not hasattr(sg, "save") or not hasattr(sg, "num_objects"):
        sg = open_vocab_scene_graph_from_controller(controller_or_sg)
    if sg is None or int(getattr(sg, "num_objects", 0) or 0) <= 0:
        _patch_manifest_open_vocab(path_obj, present=False)
        return False
    out = path_obj / OPEN_VOCAB_SCENE_GRAPH_DIR
    out.mkdir(parents=True, exist_ok=True)
    sg.save(str(out))
    # Keep a dedicated report so we do not overwrite GraphEQA scene_graph_report.txt.
    try:
        (out / "open_vocab_report.txt").write_text(sg.to_string() + "\n", encoding="utf-8")
    except Exception:
        pass
    _patch_manifest_open_vocab(path_obj, present=True)
    logger.info(f"lifelong: saved open-vocab scene graph ({sg.num_objects} objects) → {out}")
    return True


class _OpenVocabHolder:
    """Minimal processor stand-in so ``voxel_map.get_scene_graph()`` works after load."""

    def __init__(self, scene_graph: Any):
        self.scene_graph = scene_graph


def load_open_vocab_scene_graph_sidecar(controller: Any, path: str | Path) -> bool:
    """Restore ``open_vocab_scene_graph/`` into the controller voxel map when present."""
    path_obj = Path(path) / OPEN_VOCAB_SCENE_GRAPH_DIR
    sg_json = path_obj / "scene_graph.json"
    if not sg_json.is_file():
        return False
    from emet.mapping.scene_graph.open_vocab_scene_graph import OpenVocabSceneGraph

    loaded = OpenVocabSceneGraph.load(str(path_obj))
    n = int(getattr(loaded, "num_objects", 0) or 0)
    proc = getattr(controller, "_open_vocab_sg_processor", None)
    if proc is not None and getattr(proc, "scene_graph", None) is not None:
        live = proc.scene_graph
        live.nodes = loaded.nodes
        live.edges = loaded.edges
        live._next_id = loaded._next_id
        target = live
    else:
        holder = _OpenVocabHolder(loaded)
        controller._open_vocab_sg_processor = holder
        vm = _voxel_map_from_controller(controller)
        if vm is not None and hasattr(vm, "set_scene_graph_processor"):
            vm.set_scene_graph_processor(holder)
        target = loaded
    logger.info(f"lifelong: restored open-vocab scene graph ({n} objects) from {path_obj}")
    # Optional Rerun refresh for OV graph.
    viz = getattr(controller, "rerun_visualizer", None)
    if viz is not None and hasattr(viz, "update_open_vocab_scene_graph") and n > 0:
        try:
            viz.update_open_vocab_scene_graph(target)
        except Exception as exc:
            logger.debug(f"lifelong: open-vocab Rerun update skipped ({exc})")
    return n > 0


def checkpoint_expected_counts(path: str | Path) -> dict[str, Any]:
    """Read expected graph/voxel sizes from an on-disk memory directory (no MuJoCo)."""
    import json

    path_obj = Path(path)
    out: dict[str, Any] = {
        "n_graph_nodes": 0,
        "has_graph": False,
        "has_voxel_pickle": False,
        "final_step": None,
        "voxel_pickle_bytes": 0,
    }
    manifest_path = path_obj / "manifest.json"
    if manifest_path.is_file():
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
        out["has_graph"] = bool(man.get("has_graph"))
        out["has_voxel_pickle"] = bool(man.get("has_voxel_pickle"))
        if man.get("final_step") is not None:
            out["final_step"] = int(man["final_step"])
    graph_path = path_obj / "graph.json"
    if graph_path.is_file():
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        nodes = graph.get("nodes") or []
        out["n_graph_nodes"] = int(len(nodes))
        out["has_graph"] = out["has_graph"] or bool(nodes)
    voxel_pickle = path_obj / VOXEL_PICKLE_FILENAME
    if voxel_pickle.is_file():
        out["has_voxel_pickle"] = True
        out["voxel_pickle_bytes"] = int(voxel_pickle.stat().st_size)
    return out


def verify_lifelong_restore(
    controller: Any,
    path: str | Path,
    info: dict[str, Any],
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Confirm graph + voxel memory match the checkpoint; raise if restore clearly failed.

    ``strict=True`` (default) raises ``RuntimeError`` when the checkpoint has content but the
    controller does not after load. Soft mode only logs errors.
    """
    expected = checkpoint_expected_counts(path)
    gm = getattr(controller, "graph_memory", None)
    n_nodes = 0
    if gm is not None and hasattr(gm, "get_nodes"):
        n_nodes = int(len(gm.get_nodes()))
    n_obs = 0
    if gm is not None:
        obs = getattr(gm, "_observations", None) or getattr(gm, "observations", None)
        if obs is not None:
            try:
                n_obs = int(len(obs))
            except TypeError:
                n_obs = 0
    voxel_pts = int(info.get("voxel_points") or 0)
    semantic_pts = int(info.get("semantic_points") or 0)

    report = {
        **expected,
        "restored_graph_nodes": n_nodes,
        "restored_graph_obs": n_obs,
        "restored_voxel_points": voxel_pts,
        "restored_semantic_points": semantic_pts,
        "ok": True,
        "errors": [],
    }

    if expected["n_graph_nodes"] > 0 and n_nodes == 0:
        report["errors"].append(
            f"checkpoint has {expected['n_graph_nodes']} graph nodes but controller has 0 after load"
        )
    elif expected["n_graph_nodes"] > 0 and n_nodes < max(1, expected["n_graph_nodes"] // 2):
        report["errors"].append(
            f"checkpoint has {expected['n_graph_nodes']} graph nodes but only {n_nodes} restored"
        )

    if expected["has_voxel_pickle"] and expected["voxel_pickle_bytes"] > 10_000:
        if voxel_pts == 0 and semantic_pts == 0:
            report["errors"].append(
                f"checkpoint has {VOXEL_PICKLE_FILENAME} "
                f"({expected['voxel_pickle_bytes'] / 1e6:.1f} MiB) but voxel/semantic clouds are empty"
            )

    if not info.get("graph_loaded") and expected["has_graph"] and expected["n_graph_nodes"] > 0:
        report["errors"].append("checkpoint has a graph but graph_loaded=False")
    if expected["has_voxel_pickle"] and not info.get("voxel_pickle_loaded"):
        report["errors"].append(f"checkpoint has {VOXEL_PICKLE_FILENAME} but voxel_pickle_loaded=False")

    report["ok"] = len(report["errors"]) == 0
    if report["errors"]:
        msg = "Lifelong restore verification failed: " + "; ".join(report["errors"])
        if strict:
            raise RuntimeError(msg)
        logger.error(msg)
    else:
        logger.info(
            f"lifelong restore OK: graph_nodes={n_nodes}/{expected['n_graph_nodes']} "
            f"obs={n_obs} voxel_pts={voxel_pts} semantic_pts={semantic_pts}"
        )
    return report


def load_lifelong_checkpoint(
    controller: Any,
    path: str | Path,
    *,
    refine_start: bool = False,
    refine_kwargs: dict[str, Any] | None = None,
    memory_backend: str | None = None,
) -> dict[str, Any]:
    """Load graph (+ optional ``voxel_map.pkl``) and restore the staleness clock.

    Matches ``GraphEQAController`` constructor resume semantics so CHAT agent
    ``--input-path`` stays consistent with ``emet run dynagraph --input-path``.

    When ``refine_start`` is true, any **pre-existing** live voxel cloud (e.g. from a
    short scan before load) is used to estimate a local SE(2) fudge mapping the
    saved map into the live frame. If refine is skipped or rejected, the assumed
    pose is kept (identity).

    ``memory_backend`` selects which object-graph sidecar to restore (``dynagraph`` /
    ``graph_eqa`` → ``graph.json``; ``open_vocab`` → ``open_vocab_scene_graph/``).
    Legacy dual directories load only the active plug-in.
    """
    from emet.memory.backend import get_memory_backend

    path_obj = Path(path)
    if not path_obj.is_dir() or not is_memory_directory(str(path_obj)):
        raise FileNotFoundError(f"Not a memory directory: {path}")

    mb = str(memory_backend or "").strip().lower().replace("-", "_") or None

    gm = getattr(controller, "graph_memory", None)
    vm = None
    if hasattr(controller, "get_voxel_map"):
        vm = controller.get_voxel_map()
    elif hasattr(controller, "voxel_map"):
        vm = controller.voxel_map

    live_before = voxel_semantic_xyz(vm) if refine_start else None
    if live_before is not None:
        live_before = live_before.copy()
        logger.info(f"lifelong: live cloud for refine has {live_before.shape[0]} points")
    elif refine_start:
        logger.warning("lifelong: no live cloud before load; --refine-start will likely skip")

    info: dict[str, Any] = {
        "path": str(path_obj),
        "graph_loaded": False,
        "voxel_pickle_loaded": False,
        "open_vocab_loaded": False,
        "final_step": None,
        "saved_xyz": None,
        "refine": None,
        "memory_backend": mb,
    }

    if gm is not None and mb != "open_vocab":
        logger.info(f"lifelong: loading graph from {path_obj}")
        backend = get_memory_backend("graph_eqa", graph_memory=gm, voxel_map=vm)
        backend.load(str(path_obj))
        n_after = len(gm.get_nodes()) if hasattr(gm, "get_nodes") else 0
        info["graph_loaded"] = n_after > 0
        info["graph_nodes"] = int(n_after)
        final_step = getattr(backend, "loaded_final_step", None)
        if final_step is not None and int(final_step) > 0:
            info["final_step"] = int(final_step)
            if hasattr(controller, "obs_count"):
                controller.obs_count = max(int(controller.obs_count), int(final_step))
            if hasattr(gm, "set_graph_timestep"):
                gm.set_graph_timestep(int(getattr(controller, "obs_count", final_step)))
        logger.info(f"lifelong: graph restore → {n_after} nodes, final_step={info.get('final_step')}")
    elif vm is not None and mb != "open_vocab":
        logger.info(f"lifelong: loading dynamem backend from {path_obj}")
        backend = get_memory_backend("dynamem", voxel_map=vm)
        backend.load(str(path_obj))
        info["graph_loaded"] = False
        info["graph_nodes"] = 0
    else:
        info["graph_nodes"] = 0

    voxel_pickle = path_obj / VOXEL_PICKLE_FILENAME
    if voxel_pickle.is_file() and vm is not None and hasattr(vm, "read_from_pickle"):
        size_mb = voxel_pickle.stat().st_size / (1024 * 1024)
        logger.info(f"lifelong: reading {voxel_pickle.name} ({size_mb:.1f} MiB)…")
        vm.read_from_pickle(str(voxel_pickle))
        info["voxel_pickle_loaded"] = True
        logger.info("lifelong: voxel_map.pkl loaded")

    want_ov = mb == "open_vocab" or (mb is None and gm is None)
    ov_dir = path_obj / OPEN_VOCAB_SCENE_GRAPH_DIR
    if want_ov:
        info["open_vocab_loaded"] = load_open_vocab_scene_graph_sidecar(controller, path_obj)
    elif ov_dir.is_dir():
        logger.info(
            f"lifelong: ignoring {OPEN_VOCAB_SCENE_GRAPH_DIR}/ "
            f"(memory_backend={mb or 'graph/dynagraph'})"
        )
        info["open_vocab_loaded"] = False

    saved_xyz = voxel_semantic_xyz(vm)
    if saved_xyz is not None:
        saved_xyz = saved_xyz.copy()
    info["saved_xyz"] = saved_xyz

    if refine_start:
        logger.info("lifelong: running local SE(2) refine…")
    refine = maybe_refine_loaded_memory(
        controller,
        saved_xyz=saved_xyz,
        live_xyz=live_before,
        enable=bool(refine_start),
        **(refine_kwargs or {}),
    )
    info["refine"] = refine

    # Point counts for the load banner / debugging empty Rerun.
    info["voxel_points"] = 0
    info["semantic_points"] = 0
    if vm is not None:
        vp = getattr(vm, "voxel_pcd", None)
        if vp is not None and getattr(vp, "_points", None) is not None:
            info["voxel_points"] = int(vp._points.shape[0])
        sm = getattr(vm, "semantic_memory", None)
        if sm is not None and getattr(sm, "_points", None) is not None:
            info["semantic_points"] = int(sm._points.shape[0])

    refresh = refresh_rerun_after_memory_load(controller)
    info["rerun_refreshed"] = bool(refresh.get("ok"))
    if info["voxel_pickle_loaded"] and info["voxel_points"] == 0 and info["semantic_points"] == 0:
        logger.warning(
            "lifelong: voxel_map.pkl loaded but point clouds are empty — Rerun will look blank"
        )
    elif refresh.get("ok"):
        logger.info(
            f"lifelong: Rerun refreshed "
            f"(voxel_pts={info['voxel_points']} semantic_pts={info['semantic_points']})"
        )

    info["verify"] = verify_lifelong_restore(controller, path_obj, info, strict=True)
    return info


def _controller_robot_xy(controller: Any) -> tuple[float, float] | None:
    robot = getattr(controller, "robot", None)
    if robot is None or not hasattr(robot, "get_base_pose"):
        return None
    try:
        pose = np.asarray(robot.get_base_pose(), dtype=np.float64).reshape(-1)
        if pose.size >= 2:
            return (float(pose[0]), float(pose[1]))
    except Exception:
        return None
    return None


def refresh_rerun_after_memory_load(controller: Any) -> dict[str, Any]:
    """Push restored voxel map + dynagraph into Rerun after ``--input-path`` load.

    Live ``update()`` normally feeds Rerun; a pickle restore skips that path, so the
    viewer stays empty until the next scan unless we force a refresh here.
    """
    out: dict[str, Any] = {"ok": False, "voxel": False, "semantic": False, "graph": False}
    viz = getattr(controller, "rerun_visualizer", None)
    if viz is None or getattr(viz, "enabled", True) is False:
        return out
    if not hasattr(viz, "update_voxel_map") and not hasattr(viz, "log_dynagraph_state"):
        return out

    vm = None
    if hasattr(controller, "get_voxel_map"):
        vm = controller.get_voxel_map()
    elif hasattr(controller, "voxel_map"):
        vm = controller.voxel_map
    space = getattr(controller, "space", None)
    robot_xy = _controller_robot_xy(controller)

    try:
        if (
            space is not None
            and vm is not None
            and getattr(getattr(vm, "voxel_pcd", None), "_points", None) is not None
            and hasattr(viz, "update_voxel_map")
        ):
            viz.update_voxel_map(space=space, robot_base_xy=robot_xy, force=True)
            out["voxel"] = True
        sm = getattr(vm, "semantic_memory", None) if vm is not None else None
        if (
            sm is not None
            and getattr(sm, "_points", None) is not None
            and hasattr(viz, "log_custom_pointcloud")
        ):
            rgb = getattr(sm, "_rgb", None)
            pts = sm._points
            if hasattr(pts, "detach"):
                pts_np = pts.detach().cpu()
                rgb_np = rgb.detach().cpu() / 255.0 if rgb is not None else None
            else:
                pts_np = pts
                rgb_np = np.asarray(rgb) / 255.0 if rgb is not None else None
            if rgb_np is not None:
                viz.log_custom_pointcloud(
                    "world/semantic_memory/pointcloud",
                    pts_np,
                    rgb_np,
                    0.03,
                )
                out["semantic"] = True
        gm = getattr(controller, "graph_memory", None)
        if gm is not None and hasattr(viz, "log_dynagraph_state"):
            viz.log_dynagraph_state(
                gm,
                ground_truth_mode=bool(getattr(controller, "ground_truth_mode", False)),
                force=True,
            )
            out["graph"] = True
        out["ok"] = bool(out["voxel"] or out["semantic"] or out["graph"])
    except Exception as exc:
        logger.warning(f"lifelong: Rerun refresh after load failed ({exc})")
    return out


def save_lifelong_checkpoint(
    controller: Any,
    path: str | Path,
    *,
    save_voxel_pickle: bool = True,
    memory_backend: str | None = None,
) -> str:
    """Save active graph plug-in + optional voxel pickle + ``final_step`` for later resume.

    For ``dynagraph`` / ``graph_eqa``, writes ``graph.json`` (+ voxels) and does **not**
    write an open-vocab sidecar. For ``open_vocab``, writes ``open_vocab_scene_graph/``.
    When ``memory_backend`` is omitted, infer from the controller (graph_memory → graph
    path; else OV if present).
    """
    import json

    from emet.memory.backend import get_memory_backend
    from emet.memory.format import MANIFEST_FILENAME, MemoryManifest
    from emet.memory.headless_export import export_graph_eqa_dir

    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    gm = getattr(controller, "graph_memory", None)
    vm = None
    if hasattr(controller, "get_voxel_map"):
        vm = controller.get_voxel_map()
    elif hasattr(controller, "voxel_map"):
        vm = controller.voxel_map
    final_step = int(getattr(controller, "obs_count", 0) or 0)

    mb = str(memory_backend or "").strip().lower().replace("-", "_") or None
    if mb is None:
        if gm is not None:
            mb = "dynagraph"
        elif open_vocab_scene_graph_from_controller(controller) is not None:
            mb = "open_vocab"
        else:
            mb = "dynamem"

    if mb == "open_vocab":
        from dataclasses import asdict

        man_path = path_obj / MANIFEST_FILENAME
        if not man_path.is_file():
            man = MemoryManifest(backend="open_vocab", has_open_vocab_scene_graph=False)
            man_path.write_text(json.dumps(asdict(man), indent=2) + "\n", encoding="utf-8")
        if save_voxel_pickle and vm is not None and hasattr(vm, "write_to_pickle"):
            vm.write_to_pickle(str(path_obj / VOXEL_PICKLE_FILENAME))
        save_open_vocab_scene_graph_sidecar(controller, path_obj)
        return str(path_obj)

    if gm is not None:
        export_graph_eqa_dir(
            gm,
            vm,
            str(path_obj),
            final_step=final_step,
            save_voxel_pickle=bool(save_voxel_pickle),
        )
    elif vm is not None:
        backend = get_memory_backend("dynamem", voxel_map=vm)
        backend.save(str(path_obj))
        if save_voxel_pickle and hasattr(vm, "write_to_pickle"):
            vm.write_to_pickle(str(path_obj / VOXEL_PICKLE_FILENAME))
    else:
        raise RuntimeError("Controller has no graph_memory or voxel_map to save")

    _patch_manifest_open_vocab(path_obj, present=False)
    return str(path_obj)


def maybe_refine_loaded_memory(
    controller: Any,
    *,
    saved_xyz: np.ndarray | None,
    live_xyz: np.ndarray | None = None,
    enable: bool = True,
    **refine_kwargs: Any,
) -> RefineResult:
    """If ``enable``, refine and apply transform to the loaded controller memory.

    When refine is disabled or rejected, returns an identity ``RefineResult`` and
    leaves memory unchanged (assumed pose is fine).
    """
    identity = RefineResult(
        accepted=False,
        transform=np.eye(4, dtype=np.float64),
        fitness=0.0,
        inlier_rmse=float("inf"),
        num_inliers=0,
        translation_xy_m=0.0,
        yaw_rad=0.0,
        reason="skipped",
    )
    if not enable:
        return identity

    def _skip(reason: str) -> RefineResult:
        return RefineResult(
            accepted=False,
            transform=np.eye(4, dtype=np.float64),
            fitness=0.0,
            inlier_rmse=float("inf"),
            num_inliers=0,
            translation_xy_m=0.0,
            yaw_rad=0.0,
            reason=reason,
        )

    vm = None
    if hasattr(controller, "get_voxel_map"):
        vm = controller.get_voxel_map()
    elif hasattr(controller, "voxel_map"):
        vm = controller.voxel_map
    live = live_xyz if live_xyz is not None else voxel_semantic_xyz(vm)
    # When live == saved (just loaded, no new scan), skip — nothing to align against.
    if saved_xyz is None or live is None:
        return _skip("no_clouds")
    saved_arr = _as_numpy_xyz(saved_xyz)
    live_arr = _as_numpy_xyz(live)
    if saved_arr is None or live_arr is None:
        return _skip("no_clouds")
    if saved_arr.shape == live_arr.shape and np.allclose(saved_arr, live_arr, atol=1e-5):
        return _skip("live_equals_saved")

    result = refine_start_pose(saved_arr, live_arr, **refine_kwargs)
    if result.accepted:
        gm = getattr(controller, "graph_memory", None)
        apply_se2_to_memory(graph_memory=gm, voxel_map=vm, transform=result.transform)
        logger.info(
            f"lifelong refine accepted: xy={result.translation_xy_m:.3f} m "
            f"yaw={result.yaw_rad:.3f} rad fitness={result.fitness:.3f} rmse={result.inlier_rmse:.3f}"
        )
    else:
        logger.warning(f"lifelong refine rejected ({result.reason}); keeping assumed pose")
    return result
