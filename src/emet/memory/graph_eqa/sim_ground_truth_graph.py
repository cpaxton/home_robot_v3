# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Build GraphEQA / Dynagraph scene graphs from sim ground-truth placements (no VLM)."""

from __future__ import annotations

from typing import Any

import numpy as np

from emet.memory.graph_eqa.graph_memory import GT_BODY_DESC_PREFIX, GraphEQAMemory, is_ground_truth_node
from emet.memory.graph_eqa.mujoco_align import compare_graph_to_placements_report

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
    from emet.memory.graph_eqa.instance_observations import frame_instances_to_detections

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
