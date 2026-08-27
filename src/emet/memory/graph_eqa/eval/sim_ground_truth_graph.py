# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Build GraphEQA / Dynagraph scene graphs from sim ground-truth placements (no VLM)."""

from __future__ import annotations

from typing import Any

import numpy as np

from emet.memory.graph_eqa.eval.mujoco_align import compare_graph_to_placements_report
from emet.memory.graph_eqa.graph_memory import GT_BODY_DESC_PREFIX, GraphEQAMemory, is_ground_truth_node

_EMET_INTERNAL_KEYS = frozenset({"_emet_spawn_hint_xyt"})


def ground_truth_description(body_name: str) -> str:
    """Stable graph node description for one sim body."""
    return f"{GT_BODY_DESC_PREFIX}{body_name}"


def extent_half_from_bounds(bounds: np.ndarray | list | None) -> np.ndarray | None:
    """Axis-aligned half-extents (3,) from ``bounds`` shaped ``(2, 3)`` min/max."""
    if bounds is None:
        return None
    b = np.asarray(bounds, dtype=np.float64).reshape(2, 3)
    half = (b[1] - b[0]) / 2.0
    if float(np.max(half)) <= 1e-6:
        return None
    return half


def read_sim_object_placements(session: dict[str, Any] | None) -> dict[str, dict[str, Any]] | None:
    """
    Return ``sim_object_placements`` from ``robot.get_emet_session()``, or ``None``.

    Positions are **MuJoCo world XYZ** (see ``sim_object_placements_frame`` in session).
    Optional ``bounds`` are world axis-aligned ``[[min_xyz], [max_xyz]]`` from MuJoCo geoms.
    Same frame as ``camera_pose`` and Rerun ``world/`` logging; not nav-relative gps.
    """
    if not session:
        return None
    raw = session.get("sim_object_placements")
    if not isinstance(raw, dict) or not raw:
        return None
    out: dict[str, dict[str, Any]] = {}
    for body_name, info in raw.items():
        if body_name in _EMET_INTERNAL_KEYS or not isinstance(info, dict):
            continue
        pos = info.get("pos")
        if not pos:
            continue
        entry: dict[str, Any] = {
            "cat": str(info.get("cat") or body_name),
            "pos": np.asarray(pos, dtype=np.float64).reshape(3),
            "quat": np.asarray(info.get("quat") or [1.0, 0.0, 0.0, 0.0], dtype=np.float64).reshape(4),
        }
        bounds = info.get("bounds")
        if bounds is not None:
            b = np.asarray(bounds, dtype=np.float64).reshape(2, 3)
            entry["bounds"] = b
            half = extent_half_from_bounds(b)
            if half is not None:
                entry["extent_half"] = half
        out[str(body_name)] = entry
    return out or None


def deduplicate_placements(
    placements: dict[str, dict[str, Any]],
    *,
    merge_xy_m: float = 0.02,
) -> dict[str, dict[str, Any]]:
    """
    Return a clean placement dict: one entry per body, dropping near-duplicate scans.

    Keys remain MuJoCo body names. When two bodies share the same category label and
    are within ``merge_xy_m`` in XY, keep the lexicographically first body name.
    """
    if merge_xy_m <= 0 or len(placements) <= 1:
        return dict(placements)

    kept: dict[str, dict[str, Any]] = {}
    for body_name in sorted(placements.keys()):
        info = placements[body_name]
        cat = str(info.get("cat") or body_name).strip().lower()
        pos = np.asarray(info.get("pos"), dtype=np.float64).reshape(-1)[:3]
        duplicate = False
        for kept_name, kept_info in kept.items():
            kept_cat = str(kept_info.get("cat") or kept_name).strip().lower()
            if kept_cat != cat:
                continue
            kept_pos = np.asarray(kept_info.get("pos"), dtype=np.float64).reshape(-1)[:3]
            if float(np.linalg.norm(pos[:2] - kept_pos[:2])) <= merge_xy_m:
                duplicate = True
                break
        if not duplicate:
            kept[body_name] = info
    return kept


def upsert_graph_memory_from_placements(
    graph_memory: GraphEQAMemory,
    rgb: np.ndarray,
    placements: dict[str, dict[str, Any]],
    *,
    max_objects: int = 0,
    dedupe_placements: bool = False,
) -> int:
    """
    Upsert one graph node per GT body (labels from ``cat``, xyz from ``pos``).

    When ``bounds`` / ``extent_half`` are present, stores a 3D axis-aligned box on the node.
    ``max_objects=0`` means no limit (matches uncapped server fixture scan).
    Repeated calls refresh ``last_seen`` without duplicating nodes when poses are unchanged.
    Returns the number of placements processed this call.
    """
    if dedupe_placements:
        placements = deduplicate_placements(placements)
    processed = 0
    for body_name, info in placements.items():
        if body_name in _EMET_INTERNAL_KEYS:
            continue
        if max_objects > 0 and processed >= max_objects:
            break
        cat = str(info.get("cat") or body_name).strip()
        pos = np.asarray(info.get("pos"), dtype=np.float64).reshape(-1)
        if pos.size < 3 or not cat:
            continue
        extent_half = info.get("extent_half")
        if extent_half is None:
            extent_half = extent_half_from_bounds(info.get("bounds"))
        graph_memory.upsert_ground_truth_observation(
            body_name,
            rgb,
            pos[:3],
            [cat],
            extent_half=extent_half,
        )
        processed += 1
    return processed


def populate_graph_memory_from_placements(
    graph_memory: GraphEQAMemory,
    rgb: np.ndarray,
    placements: dict[str, dict[str, Any]],
    *,
    max_objects: int = 0,
) -> int:
    """Alias for :func:`upsert_graph_memory_from_placements` (deduplicated GT insert)."""
    return upsert_graph_memory_from_placements(
        graph_memory,
        rgb,
        placements,
        max_objects=max_objects,
    )


def count_ground_truth_nodes(graph_memory: GraphEQAMemory) -> int:
    """Number of graph nodes tagged with ``ground_truth:{body_name}``."""
    return sum(1 for n in graph_memory.get_nodes() if is_ground_truth_node(n))


def build_ground_truth_graph_from_session(
    graph_memory: GraphEQAMemory,
    rgb: np.ndarray,
    session: dict[str, Any] | None,
) -> tuple[int, dict[str, dict[str, Any]] | None]:
    """
    Upsert ``graph_memory`` from session GT.

    Returns ``(n_gt_nodes_in_graph, placements_dict)`` after upsert (stable count over time).
    """
    placements = read_sim_object_placements(session)
    if not placements:
        return 0, None
    upsert_graph_memory_from_placements(graph_memory, rgb, placements)
    return count_ground_truth_nodes(graph_memory), placements


def ground_truth_alignment_report(
    graph_memory: GraphEQAMemory,
    placements: dict[str, dict[str, Any]] | None,
    *,
    max_dist_xy: float = 1.2,
    perception_nodes_only: bool = False,
) -> str:
    """Printable dev report: built graph vs GT placements (identity check in GT mode)."""
    if not placements:
        return "Graph vs GT: (no sim_object_placements in emet_session)"
    nodes = graph_memory.get_nodes()
    if perception_nodes_only:
        nodes = [n for n in nodes if not is_ground_truth_node(n)]
        if not nodes:
            return "Graph vs GT: (no perception nodes in graph memory; run rotate / explore first)"
    return compare_graph_to_placements_report(
        nodes,
        placements,
        max_dist_xy=max_dist_xy,
    )


def ground_truth_body_key(node: Any) -> str | None:
    """MuJoCo body key from a GT graph node description (``ground_truth:{body_key}``)."""
    if not is_ground_truth_node(node):
        return None
    desc = str(getattr(node, "description", "") or "")
    if not desc.startswith(GT_BODY_DESC_PREFIX):
        return None
    key = desc[len(GT_BODY_DESC_PREFIX) :]
    if "|det:" in key:
        key = key.split("|det:", 1)[0]
    key = key.strip()
    return key or None


def associate_instance_detections_to_ground_truth(
    graph_memory: GraphEQAMemory,
    frame: Any,
    *,
    rgb: np.ndarray,
    voxel_map: Any,
    detection_model: Any | None = None,
    max_assoc_xy_m: float = 0.75,
) -> int:
    """
    Match YoloE/instance centroids to nearest GT nodes in XY; attach frame RGB + det label.

    Does not add new graph nodes — only refreshes GT observation images for dataset export.
    """
    from emet.memory.graph_eqa.ingest.instance_observations import frame_instances_to_detections

    gt_nodes = [(ground_truth_body_key(n), n) for n in graph_memory.get_nodes()]
    gt_nodes = [(k, n) for k, n in gt_nodes if k]
    if not gt_nodes:
        return 0

    dets = frame_instances_to_detections(
        frame,
        min_depth=float(voxel_map.min_depth),
        max_depth=float(voxel_map.max_depth),
        detection_model=detection_model,
    )
    if not dets:
        return 0

    matched = 0
    used_gt: set[str] = set()
    rgb_a = np.asarray(rgb, dtype=np.uint8)
    for det in dets:
        xyz = np.asarray(det["xyz"], dtype=np.float64).reshape(3)
        label = str(det.get("label_short") or "").strip()
        best_key: str | None = None
        best_d = float(max_assoc_xy_m)
        for body_key, node in gt_nodes:
            if body_key in used_gt:
                continue
            nxy = np.asarray(node.xyz, dtype=np.float64).reshape(-1)[:2]
            dxy = float(np.linalg.norm(nxy - xyz[:2]))
            if dxy < best_d:
                best_d = dxy
                best_key = body_key
        if best_key is None:
            continue
        if graph_memory.attach_detection_to_ground_truth_node(best_key, rgb_a, detection_label=label or None):
            used_gt.add(best_key)
            matched += 1
    return matched


def _parse_detection_label(description: str | None) -> str | None:
    desc = str(description or "")
    if "|det:" not in desc:
        return None
    return desc.split("|det:", 1)[1].strip() or None


def _aabb_corners(bounds: np.ndarray) -> np.ndarray:
    """Eight world XYZ corners from ``bounds`` shaped ``(2, 3)``."""
    b = np.asarray(bounds, dtype=np.float64).reshape(2, 3)
    lo, hi = b[0], b[1]
    return np.array(
        [
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [lo[0], hi[1], hi[2]],
            [hi[0], hi[1], hi[2]],
        ],
        dtype=np.float64,
    )


def project_world_points_to_image(
    world_xyz: np.ndarray,
    camera_pose: np.ndarray,
    camera_K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project world points to pixel coordinates.

    ``camera_pose`` is camera-to-world (4x4); returns ``(uv, valid)`` with ``uv`` (N, 2).
    """
    w2c = np.linalg.inv(np.asarray(camera_pose, dtype=np.float64).reshape(4, 4))
    pts = np.asarray(world_xyz, dtype=np.float64).reshape(-1, 3)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    cam = (w2c @ np.hstack([pts, ones]).T).T[:, :3]
    valid = cam[:, 2] > 1e-4
    K = np.asarray(camera_K, dtype=np.float64).reshape(3, 3)
    z = np.where(valid, cam[:, 2], 1.0)
    u = K[0, 0] * cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * cam[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=1), valid


def projected_aabb_2d(
    bounds: np.ndarray,
    camera_pose: np.ndarray,
    camera_K: np.ndarray,
    *,
    image_hw: tuple[int, int] | None = None,
) -> tuple[int, int, int, int] | None:
    """Axis-aligned 2D bbox (x0, y0, x1, y1) from a world AABB, or ``None`` if behind camera."""
    corners = _aabb_corners(bounds)
    uv, valid = project_world_points_to_image(corners, camera_pose, camera_K)
    if not np.any(valid):
        return None
    uv = uv[valid]
    x0, y0 = np.floor(uv.min(axis=0)).astype(int)
    x1, y1 = np.ceil(uv.max(axis=0)).astype(int)
    if image_hw is not None:
        h, w = image_hw
        x0 = int(np.clip(x0, 0, w - 1))
        x1 = int(np.clip(x1, 0, w - 1))
        y0 = int(np.clip(y0, 0, h - 1))
        y1 = int(np.clip(y1, 0, h - 1))
    if x1 <= x0 or y1 <= y0:
        return None
    return int(x0), int(y0), int(x1), int(y1)


def _bbox_mask(h: int, w: int, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    mask = np.zeros((h, w), dtype=bool)
    mask[y0 : y1 + 1, x0 : x1 + 1] = True
    return mask


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    if inter <= 0:
        return 0.0
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def associate_ground_truth_to_frame_instances(
    placements: dict[str, dict[str, Any]],
    frame: Any,
    *,
    min_iou: float = 0.05,
) -> list[dict[str, Any]]:
    """
    Match GT body AABBs (projected to image) to YoloE instance masks via IoU.

    Returns a list of ``{body_key, cat, instance_id, iou, det_label}`` dicts.
    """
    inst = getattr(frame, "instance", None)
    camera_pose = getattr(frame, "camera_pose", None)
    camera_K = getattr(frame, "camera_K", None)
    if inst is None or camera_pose is None or camera_K is None:
        return []
    inst = np.asarray(inst)
    if inst.ndim != 2:
        return []
    h, w = inst.shape[:2]
    instance_ids = [int(uid) for uid in np.unique(inst) if int(uid) > 0]
    if not instance_ids:
        return []

    from emet.memory.graph_eqa.ingest.instance_observations import frame_instances_to_detections

    dets = frame_instances_to_detections(frame, min_depth=0.05, max_depth=8.0, detection_model=None)
    det_by_id = {int(d["instance_id"]): d for d in dets if d.get("instance_id") is not None}

    associations: list[dict[str, Any]] = []
    used_instances: set[int] = set()
    for body_key, info in placements.items():
        if body_key in _EMET_INTERNAL_KEYS:
            continue
        bounds = info.get("bounds")
        if bounds is None:
            continue
        bbox = projected_aabb_2d(bounds, camera_pose, camera_K, image_hw=(h, w))
        if bbox is None:
            continue
        bbox_mask = _bbox_mask(h, w, bbox)
        best_iou = 0.0
        best_uid: int | None = None
        for uid in instance_ids:
            if uid in used_instances:
                continue
            inst_mask = inst == uid
            iou = _mask_iou(inst_mask, bbox_mask)
            if iou > best_iou:
                best_iou = iou
                best_uid = uid
        if best_uid is None or best_iou < min_iou:
            continue
        det = det_by_id.get(best_uid, {})
        associations.append(
            {
                "body_key": body_key,
                "cat": str(info.get("cat") or body_key),
                "instance_id": best_uid,
                "iou": float(best_iou),
                "det_label": str(det.get("label_short") or ""),
            }
        )
        used_instances.add(best_uid)
    return associations


def associate_ground_truth_to_voxel_observation(
    placements: dict[str, dict[str, Any]],
    obs: Any,
) -> list[dict[str, Any]]:
    """
    Tag world XYZ points inside GT AABBs (read-only metadata for export/eval).

    Returns ``[{body_key, cat, point_count}]`` for bodies with in-bounds points.
    """
    world_xyz = getattr(obs, "full_world_xyz", None)
    if world_xyz is None:
        return []
    pts = np.asarray(world_xyz, dtype=np.float64).reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1)
    pts = pts[valid]
    if pts.size == 0:
        return []
    hits: list[dict[str, Any]] = []
    for body_key, info in placements.items():
        if body_key in _EMET_INTERNAL_KEYS:
            continue
        bounds = info.get("bounds")
        if bounds is None:
            continue
        b = np.asarray(bounds, dtype=np.float64).reshape(2, 3)
        inside = np.all((pts >= b[0]) & (pts <= b[1]), axis=1)
        count = int(inside.sum())
        if count > 0:
            hits.append(
                {
                    "body_key": body_key,
                    "cat": str(info.get("cat") or body_key),
                    "point_count": count,
                }
            )
    return hits


def graph_node_export_fields(node: Any, placements: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Extra GraphNodeView fields for GT nodes (body_key, bounds, extent_half, detection_label)."""
    body_key = ground_truth_body_key(node)
    if body_key is None:
        return {}
    out: dict[str, Any] = {"body_key": body_key}
    desc = getattr(node, "description", None)
    det = _parse_detection_label(desc)
    if det:
        out["detection_label"] = det
    extent_half = getattr(node, "extent_half", None)
    if extent_half is not None:
        out["extent_half"] = list(np.ravel(extent_half).tolist())
    bounds = None
    if placements and body_key in placements:
        bounds = placements[body_key].get("bounds")
    if bounds is not None:
        b = np.asarray(bounds, dtype=np.float64).reshape(2, 3)
        out["bounds"] = [list(b[0]), list(b[1])]
    return out


def gt_graph_completeness(
    graph_memory: GraphEQAMemory,
    placements: dict[str, dict[str, Any]] | None,
) -> float:
    """Fraction of placement bodies present as GT graph nodes."""
    if not placements:
        return 0.0
    gt_keys = {ground_truth_body_key(n) for n in graph_memory.get_nodes()}
    gt_keys.discard(None)
    return len(gt_keys & set(placements.keys())) / float(len(placements))


def gt_localization_errors(
    graph_memory: GraphEQAMemory,
    placements: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, float]]:
    """Per-body XY and Z error between GT graph node xyz and placement pos."""
    if not placements:
        return {}
    node_by_key: dict[str, Any] = {}
    for n in graph_memory.get_nodes():
        key = ground_truth_body_key(n)
        if key:
            node_by_key[key] = n
    errors: dict[str, dict[str, float]] = {}
    for body_key, info in placements.items():
        if body_key not in node_by_key:
            continue
        gt_pos = np.asarray(info["pos"], dtype=np.float64).reshape(3)
        node_xyz = np.asarray(node_by_key[body_key].xyz, dtype=np.float64).reshape(3)
        errors[body_key] = {
            "err_xy_m": float(np.linalg.norm(node_xyz[:2] - gt_pos[:2])),
            "err_z_m": float(abs(node_xyz[2] - gt_pos[2])),
            "cat": str(info.get("cat") or body_key),
        }
    return errors


def instance_gt_association_recall(
    graph_memory: GraphEQAMemory,
    placements: dict[str, dict[str, Any]] | None,
    *,
    max_assoc_xy_m: float = 0.75,
) -> float:
    """Fraction of GT bodies with a detection label attached (forward assoc succeeded)."""
    if not placements:
        return 0.0
    matched = 0
    for n in graph_memory.get_nodes():
        key = ground_truth_body_key(n)
        if not key or key not in placements:
            continue
        if _parse_detection_label(getattr(n, "description", None)):
            matched += 1
    return matched / float(len(placements))


def projected_association_recall(
    frames: list[Any],
    placements: dict[str, dict[str, Any]] | None,
    *,
    min_iou: float = 0.05,
) -> float:
    """Fraction of GT bodies with bounds matched in at least one frame (reverse assoc)."""
    if not placements:
        return 0.0
    bodies_with_bounds = [k for k, v in placements.items() if v.get("bounds") is not None]
    if not bodies_with_bounds:
        return 0.0
    matched: set[str] = set()
    for fr in frames:
        assocs = getattr(fr, "gt_associations", None)
        if not assocs and hasattr(fr, "info") and isinstance(fr.info, dict):
            assocs = fr.info.get("gt_associations")
        if not assocs:
            continue
        for row in assocs:
            if float(row.get("iou", 0.0)) >= min_iou:
                matched.add(str(row["body_key"]))
    return len(matched) / float(len(bodies_with_bounds))


def placements_to_json_dict(placements: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """JSON-serializable snapshot of placements for ``sim_object_placements.json``."""
    out: dict[str, Any] = {}
    for body_key, info in placements.items():
        if body_key in _EMET_INTERNAL_KEYS:
            continue
        entry: dict[str, Any] = {
            "cat": str(info.get("cat") or body_key),
            "pos": np.asarray(info["pos"], dtype=np.float64).reshape(3).tolist(),
        }
        if info.get("quat") is not None:
            entry["quat"] = np.asarray(info["quat"], dtype=np.float64).reshape(4).tolist()
        if info.get("bounds") is not None:
            b = np.asarray(info["bounds"], dtype=np.float64).reshape(2, 3)
            entry["bounds"] = [b[0].tolist(), b[1].tolist()]
        if info.get("extent_half") is not None:
            entry["extent_half"] = np.asarray(info["extent_half"], dtype=np.float64).reshape(3).tolist()
        out[str(body_key)] = entry
    return out


def gt_pose_sanity_report(
    placements: dict[str, dict[str, Any]] | None,
    *,
    robot_world_xyt: np.ndarray | None = None,
    session: dict[str, Any] | None = None,
) -> str:
    """
    Short dev report: GT frame, robot world pose, distances to GT bodies.

    Helps catch nav-world vs MuJoCo-world mismatches in Rerun.
    """
    lines = ["GT pose sanity:"]
    frame = (session or {}).get("sim_object_placements_frame", "mujoco_world (assumed)")
    lines.append(f"  frame: {frame}")
    if session and session.get("navigation_origin_xyt") is not None:
        org = np.asarray(session["navigation_origin_xyt"], dtype=np.float64).reshape(-1)[:3]
        lines.append(f"  navigation_origin_xyt (world): ({org[0]:.2f}, {org[1]:.2f}, {org[2]:.2f})")
        lines.append("  note: gps/compass are nav-relative; GT + camera_pose are MuJoCo world")
    if robot_world_xyt is not None:
        r = np.asarray(robot_world_xyt, dtype=np.float64).reshape(-1)[:3]
        lines.append(f"  robot base (world x,y,θ): ({r[0]:.2f}, {r[1]:.2f}, {r[2]:.2f})")
    if not placements:
        lines.append("  (no placements)")
        return "\n".join(lines)
    if robot_world_xyt is not None:
        rxy = np.asarray(robot_world_xyt, dtype=np.float64).reshape(-1)[:2]
        for body_name in sorted(placements.keys()):
            pos = np.asarray(placements[body_name]["pos"], dtype=np.float64).reshape(-1)[:3]
            d = float(np.linalg.norm(pos[:2] - rxy))
            cat = placements[body_name].get("cat", body_name)
            lines.append(f"  {body_name} ({cat}): dist_xy={d:.2f}m  pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})")
    return "\n".join(lines)
