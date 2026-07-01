# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Export Robocasa / MuJoCo object ground truth (3D bounds + optional head-camera 2D boxes)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from emet.memory.graph_eqa.mujoco_align import _norm_label
from emet.simulation.molmo_occupancy._geom_aabb import geom_aabb

GT_SCHEMA_VERSION = 1

# Default render size used by robosuite server primary camera.
DEFAULT_IMAGE_HW = (480, 640)


def _body_geom_ids(model: mujoco.MjModel, body_id: int) -> list[int]:
    return [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == int(body_id)]


def _camera_K_from_model(model: mujoco.MjModel, camera_name: str, width: int, height: int) -> np.ndarray:
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        return np.eye(3, dtype=np.float64)
    fovy = float(model.cam_fovy[cam_id])
    f = 0.5 * height / np.tan(np.radians(fovy) / 2.0)
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def camera_pose_world_opencv(model: mujoco.MjModel, data: mujoco.MjData, camera_name: str) -> np.ndarray:
    """4×4 OpenCV camera-to-world (same convention as :meth:`RobosuiteZmqServer._camera_pose_world`)."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        return np.eye(4, dtype=np.float64)
    R = np.asarray(data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
    pos = np.asarray(data.cam_xpos[cam_id], dtype=np.float64).reshape(3)
    d = np.diag([1.0, -1.0, -1.0])
    r_cv = R @ d
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = r_cv
    T[:3, 3] = pos
    return T


def world_to_camera_opencv(T_cam_world: np.ndarray) -> np.ndarray:
    """Inverse of camera-to-world OpenCV pose."""
    R = T_cam_world[:3, :3]
    t = T_cam_world[:3, 3]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ t
    return T


def project_world_points_to_pixels(
    points_world: np.ndarray,
    *,
    T_cam_world: np.ndarray,
    K: np.ndarray,
    image_hw: tuple[int, int],
    min_depth: float = 0.05,
) -> np.ndarray | None:
    """Project N×3 world points to N×2 pixel coords; return None if none in front of camera."""
    h, w = image_hw
    T_w2c = world_to_camera_opencv(T_cam_world)
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    hom = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    cam = (T_w2c @ hom.T).T[:, :3]
    z = cam[:, 2]
    valid = z > min_depth
    if not np.any(valid):
        return None
    cam = cam[valid]
    u = K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]
    v = K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]
    uv = np.stack([u, v], axis=1)
    in_img = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    if not np.any(in_img):
        return None
    return uv[in_img]


def aabb_corners(min_xyz: np.ndarray, max_xyz: np.ndarray) -> np.ndarray:
    """8×3 world corners from axis-aligned bounds."""
    mn = np.asarray(min_xyz, dtype=np.float64).reshape(3)
    mx = np.asarray(max_xyz, dtype=np.float64).reshape(3)
    return np.array(
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


def bounds_3d_from_geoms(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> dict[str, list[float]] | None:
    gids = _body_geom_ids(model, body_id)
    if not gids:
        return None
    center, size = geom_aabb(model, data, gids, tight_mesh=True)
    half = np.asarray(size, dtype=np.float64).reshape(3) * 0.5
    c = np.asarray(center, dtype=np.float64).reshape(3)
    mn = (c - half).tolist()
    mx = (c + half).tolist()
    return {
        "center": c.tolist(),
        "size": np.asarray(size, dtype=np.float64).reshape(3).tolist(),
        "min": mn,
        "max": mx,
    }


def bbox_xyxy_from_bounds(
    bounds: dict[str, list[float]],
    *,
    T_cam_world: np.ndarray,
    K: np.ndarray,
    image_hw: tuple[int, int],
) -> list[int] | None:
    mn = np.asarray(bounds["min"], dtype=np.float64)
    mx = np.asarray(bounds["max"], dtype=np.float64)
    corners = aabb_corners(mn, mx)
    uv = project_world_points_to_pixels(corners, T_cam_world=T_cam_world, K=K, image_hw=image_hw)
    if uv is None or uv.shape[0] < 2:
        return None
    x0 = int(np.floor(float(uv[:, 0].min())))
    y0 = int(np.floor(float(uv[:, 1].min())))
    x1 = int(np.ceil(float(uv[:, 0].max())))
    y1 = int(np.ceil(float(uv[:, 1].max())))
    h, w = image_hw
    x0 = max(0, min(x0, w - 1))
    x1 = max(0, min(x1, w))
    y0 = max(0, min(y0, h - 1))
    y1 = max(0, min(y1, h))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def iter_placement_objects(placements: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """``(body_name, info)`` for manipulable objects (skip spawn hints)."""
    out: list[tuple[str, dict[str, Any]]] = []
    for name, info in placements.items():
        if name.startswith("_"):
            continue
        if not isinstance(info, dict):
            continue
        if not name.endswith("_main"):
            continue
        out.append((name, info))
    return sorted(out, key=lambda x: x[0])


def build_gt_scene_payload(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    placements: dict[str, Any],
    robot: str,
    seed: int,
    layout: int,
    style: int,
    camera_name: str,
    image_hw: tuple[int, int] = DEFAULT_IMAGE_HW,
    project_head_bbox: bool = True,
    source: str = "robocasa",
) -> dict[str, Any]:
    mujoco.mj_forward(model, data)
    h, w = image_hw
    K = _camera_K_from_model(model, camera_name, width=w, height=h)
    T_cam = camera_pose_world_opencv(model, data, camera_name)
    T_w2c = world_to_camera_opencv(T_cam)

    objects: list[dict[str, Any]] = []
    for body_name, info in iter_placement_objects(placements):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        cat = str(info.get("cat", body_name.replace("_main", "")))
        pos = np.asarray(info.get("pos", [0.0, 0.0, 0.0]), dtype=np.float64).ravel()
        quat = np.asarray(info.get("quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64).ravel()
        if pos.size < 3:
            pos = np.zeros(3, dtype=np.float64)
        if quat.size < 4:
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        bounds = None
        if bid >= 0:
            bounds = bounds_3d_from_geoms(model, data, bid)
        row: dict[str, Any] = {
            "id": body_name,
            "label": cat,
            "label_norm": _norm_label(cat),
            "body_id": int(bid),
            "pos_world": pos[:3].tolist(),
            "quat_world": quat[:4].tolist(),
        }
        if bounds is not None:
            row["bounds_3d"] = bounds
            if project_head_bbox:
                bb = bbox_xyxy_from_bounds(bounds, T_cam_world=T_cam, K=K, image_hw=image_hw)
                if bb is not None:
                    row["bbox_xyxy_head"] = bb
        objects.append(row)

    return {
        "schema_version": GT_SCHEMA_VERSION,
        "source": source,
        "robot": robot,
        "seed": int(seed),
        "layout": int(layout),
        "style": int(style),
        "image_hw": [int(h), int(w)],
        "camera": {
            "name": camera_name,
            "K": K.tolist(),
            "extrinsic_cam_to_world": T_cam.tolist(),
            "extrinsic_world_to_cam": T_w2c.tolist(),
        },
        "objects": objects,
    }


def build_gt_scene_payload_from_session_placements(
    placements: dict[str, Any],
    *,
    robot: str,
    seed: int = 0,
    layout: int = 1,
    style: int = 1,
    image_hw: tuple[int, int] = DEFAULT_IMAGE_HW,
) -> dict[str, Any]:
    """
    Build a ``load_gt_scene_json``-compatible payload from live ``sim_object_placements``.

    Uses manipulable ``*_main`` bodies so offline tuning matches the running sim episode.
    """
    h, w = image_hw
    objects: list[dict[str, Any]] = []
    for body_name, info in iter_placement_objects(placements):
        cat = str(info.get("cat", body_name.replace("_main", "")))
        pos = np.asarray(info.get("pos", [0.0, 0.0, 0.0]), dtype=np.float64).ravel()
        if pos.size < 3:
            pos = np.zeros(3, dtype=np.float64)
        row: dict[str, Any] = {
            "id": body_name,
            "label": cat,
            "label_norm": _norm_label(cat),
            "pos_world": pos[:3].tolist(),
        }
        bounds = info.get("bounds")
        if bounds is not None:
            b = np.asarray(bounds, dtype=np.float64).reshape(2, 3)
            mn = b[0]
            mx = b[1]
            c = 0.5 * (mn + mx)
            size = mx - mn
            row["bounds_3d"] = {
                "center": c.tolist(),
                "size": size.tolist(),
                "min": mn.tolist(),
                "max": mx.tolist(),
            }
        objects.append(row)

    return {
        "schema_version": GT_SCHEMA_VERSION,
        "source": "sim_session",
        "robot": robot,
        "seed": int(seed),
        "layout": int(layout),
        "style": int(style),
        "image_hw": [int(h), int(w)],
        "objects": objects,
    }


def write_gt_scene_json(path: str | Path, payload: dict[str, Any]) -> Path:
    dest = Path(path).expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dest


def load_gt_scene_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def primary_head_camera_name(robot: str) -> str:
    from emet.robots import get_robot_spec

    spec = get_robot_spec(robot)
    if spec is None:
        return "head_left"
    names = list(spec.camera_names)
    if not names:
        return "head_camera"
    return names[0]


def export_robocasa_gt_scene(
    *,
    robot: str = "innate_mars",
    seed: int = 0,
    layout: int = 1,
    style: int = 1,
    task: str = "PickPlaceCounterToCabinet",
    out_path: str | Path,
    project_head_bbox: bool = True,
) -> Path:
    """Build kitchen via ``robocasa_gen``, forward physics, write GT JSON."""
    import numpy as np

    np.random.seed(int(seed))
    from emet.simulation.stretch_mujoco.robocasa_gen import model_generation_wizard

    model, _xml, placements = model_generation_wizard(
        task=task,
        layout=int(layout),
        style=int(style),
        robot=robot,
        seed=int(seed),
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cam = primary_head_camera_name(robot)
    payload = build_gt_scene_payload(
        model,
        data,
        placements=placements,
        robot=robot,
        seed=int(seed),
        layout=int(layout),
        style=int(style),
        camera_name=cam,
        project_head_bbox=project_head_bbox,
    )
    return write_gt_scene_json(out_path, payload)


def augment_ground_truth_json_with_objects(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    placements: dict[str, Any] | None,
    robot: str,
    camera_name: str | None = None,
    image_hw: tuple[int, int] = DEFAULT_IMAGE_HW,
) -> dict[str, Any]:
    """Optional ``objects`` block for ``mujoco_ground_truth_dump`` JSON."""
    if not placements:
        return {}
    cam = camera_name or primary_head_camera_name(robot)
    payload = build_gt_scene_payload(
        model,
        data,
        placements=placements,
        robot=robot,
        seed=0,
        layout=-1,
        style=-1,
        camera_name=cam,
        project_head_bbox=True,
        source="robocasa_live",
    )
    return {"objects": payload["objects"], "camera": payload["camera"], "image_hw": payload["image_hw"]}
