# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Add/merge observations, ground-truth upsert, frontiers, viewpoints, and edges."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

import numpy as np
from PIL import Image

from emet.memory.graph_eqa.graph_types import (
    GT_BODY_DESC_PREFIX,
    GraphNavigationSample,
    GraphNode,
    GraphObservation,
    RelationBelief,
    _inside_bounds,
    _near,
    _node_is_room,
    _on,
    _on_floor,
    is_ground_truth_node,
)
from emet.utils.logger import Logger

_logger = Logger(__name__)



def add_observation(
    self,
    rgb: np.ndarray | Image.Image,
    xyz: np.ndarray,
    labels: list[str],
    description: str | None = None,
    *,
    viewer_xyz: np.ndarray | None = None,
    bbox_xyxy: tuple[int, int, int, int] | None = None,
    extent_half: np.ndarray | None = None,
    identity_key: str | None = None,
    countable_instance: bool = False,
) -> int:
    """
    Add one observation to the graph: create a node and update edges.

    Args:
        rgb: RGB image (H, W, 3) or PIL Image
        xyz: (3,) world position for this observation (e.g. camera or centroid)
        labels: list of object/region labels (e.g. from a VLM)
        description: optional text description of the scene (e.g. from VLM)
        viewer_xyz: optional (3,) robot base or head-camera position in world frame when captured
        bbox_xyxy: optional (x0, y0, x1, y1) crop in ``rgb`` for this object (instance mask bbox)
        identity_key: optional stable identity for this detected instance
        countable_instance: whether this observation is a FIND candidate for count questions

    Returns:
        obs_id: 1-based observation id (used as image id in EQA).
    """
    if isinstance(rgb, Image.Image):
        rgb = np.array(rgb)
    step = self._effective_timestep()
    xyz_a = np.asarray(xyz, dtype=float).reshape(-1)[:3]
    viewer_a: np.ndarray | None = None
    if viewer_xyz is not None:
        viewer_a = np.asarray(viewer_xyz, dtype=float).reshape(-1)[:3].copy()
    labels_norm = [str(l).strip() for l in labels if str(l).strip()]
    if not labels_norm:
        labels_norm = ["object"]
    primary = labels_norm[0].lower()
    identity_key_norm = str(identity_key).strip() if identity_key is not None else ""
    identity_key_norm = identity_key_norm or None

    bbox_i: tuple[int, int, int, int] | None = None
    if bbox_xyxy is not None:
        b = tuple(int(x) for x in bbox_xyxy)
        if len(b) == 4:
            bbox_i = (b[0], b[1], b[2], b[3])

    if self.spatial_merge_m > 0 or identity_key_norm is not None:
        from emet.memory.graph_eqa.graph_stats import labels_compatible_for_dedup

        for idx, existing in enumerate(self._nodes):
            if existing.is_viewpoint or existing.is_frontier or is_ground_truth_node(existing):
                continue
            existing_key = str(existing.identity_key).strip() if existing.identity_key else None
            same_identity = identity_key_norm is not None and existing_key == identity_key_norm
            el = [str(x).strip() for x in existing.labels if str(x).strip()]
            if not same_identity and (not el or not labels_compatible_for_dedup(primary, el[0])):
                continue
            if (
                identity_key_norm is not None
                and existing.countable_instance
                and existing_key is not None
                and not same_identity
            ):
                continue
            ex = np.asarray(existing.xyz, dtype=float).reshape(-1)[:3]
            spatial_match = (
                self.spatial_merge_m > 0 and float(np.linalg.norm(ex[:2] - xyz_a[:2])) <= self.spatial_merge_m
            )
            if not same_identity and not spatial_match:
                continue
            sc = int(existing.support_count) + 1
            new_xyz, covariance, history, changes, belief_confidence = self._position_update(
                existing, xyz_a, step=step
            )
            merged_labels = sorted({*(str(x).strip() for x in existing.labels if str(x).strip()), *labels_norm})
            new_desc = description if description else existing.description
            merged_bbox = bbox_i if bbox_i is not None else existing.bbox_xyxy
            self._nodes[idx] = replace(
                existing,
                xyz=new_xyz,
                labels=merged_labels,
                last_seen=step,
                support_count=sc,
                description=new_desc,
                bbox_xyxy=merged_bbox,
                position_covariance=covariance,
                position_history=history,
                change_events=changes,
                belief_confidence=belief_confidence,
                identity_key=(
                    identity_key_norm
                    if identity_key_norm is not None and countable_instance
                    else existing.identity_key or identity_key_norm
                ),
                countable_instance=bool(existing.countable_instance or countable_instance),
                close_look_label=existing.close_look_label,
            )
            # Keep the graph node's candidate image in sync with this revisit.
            self.refresh_observation_candidate(
                int(existing.obs_id),
                rgb,
                xyz=new_xyz,
                labels=merged_labels,
                description=new_desc if new_desc else None,
                viewer_xyz=viewer_a,
            )
            if viewer_a is not None:
                self._ensure_viewpoint_node(int(existing.obs_id), viewer_a)
            self._update_edges()
            return int(existing.obs_id)

    obs_id = self._next_obs_id
    self._next_obs_id += 1
    node_id = len(self._nodes) + 1
    ext = None
    if extent_half is not None:
        ext = np.asarray(extent_half, dtype=float).reshape(-1)[:3].copy()
    node = GraphNode(
        node_id=node_id,
        labels=labels_norm,
        xyz=xyz_a.copy(),
        obs_id=obs_id,
        description=description,
        last_seen=step,
        support_count=1,
        extent_half=ext,
        bbox_xyxy=bbox_i,
        belief_confidence=0.55,
        position_covariance=np.zeros((3, 3), dtype=float),
        position_history=[
            {
                "step": int(step),
                "xyz": xyz_a.tolist(),
                "confidence": 0.55,
            }
        ],
        identity_key=(
            identity_key_norm
            or (
                description[len(GT_BODY_DESC_PREFIX) :]
                if isinstance(description, str) and description.startswith(GT_BODY_DESC_PREFIX)
                else f"{re.sub(r'[^a-z0-9]+', '-', primary).strip('-')}:{obs_id}"
            )
        ),
        countable_instance=bool(countable_instance),
    )
    self._nodes.append(node)
    self._observations.append(
        GraphObservation(
            obs_id=obs_id,
            rgb=rgb,
            xyz=xyz_a.copy(),
            labels=list(labels_norm),
            description=description,
            viewer_xyz=viewer_a,
        )
    )
    self._obs_revisions[int(obs_id)] = 1
    self._last_obs_content_update_id = int(obs_id)
    self._record_world_view_for_obs(obs_id)
    if viewer_a is not None:
        self._ensure_viewpoint_node(obs_id, viewer_a)
    self._update_edges()
    return obs_id

def merge_object_detection(
    self,
    rgb: np.ndarray | Image.Image,
    candidate: Any,
    *,
    merge_into_node_id: int | None,
    viewer_xyz: np.ndarray | None = None,
) -> int:
    """
    Add or merge an instance detection (GraphObjectFusion path).

    ``candidate`` is a :class:`~emet.memory.graph_eqa.graph_object_fusion.fusion.GraphDetectionCandidate`
    or any object with ``label``, ``xyz``, optional ``bbox_xyxy``, ``bounds_3d``, ``embedding``.
    """
    if isinstance(rgb, Image.Image):
        rgb = np.array(rgb)
    label = str(getattr(candidate, "label", "object"))
    xyz_a = np.asarray(candidate.xyz, dtype=float).reshape(-1)[:3]
    bbox_xyxy = getattr(candidate, "bbox_xyxy", None)
    bounds_3d = getattr(candidate, "bounds_3d", None)
    embedding = getattr(candidate, "embedding", None)
    identity_key = getattr(candidate, "identity_key", None)
    identity_key_norm = str(identity_key).strip() if identity_key is not None else ""
    identity_key_norm = identity_key_norm or None
    semantic_only = bool(getattr(candidate, "semantic_only", False))
    countable_instance = bool(
        (getattr(candidate, "countable_instance", False) or identity_key_norm) and not semantic_only
    )
    if embedding is not None:
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1).copy()

    step = self._effective_timestep()
    viewer_a: np.ndarray | None = None
    if viewer_xyz is not None:
        viewer_a = np.asarray(viewer_xyz, dtype=float).reshape(-1)[:3].copy()

    bbox_i: tuple[int, int, int, int] | None = None
    if bbox_xyxy is not None:
        b = tuple(int(x) for x in bbox_xyxy)
        if len(b) == 4:
            bbox_i = (b[0], b[1], b[2], b[3])

    if merge_into_node_id is not None:
        for idx, existing in enumerate(self._nodes):
            if int(existing.node_id) != int(merge_into_node_id):
                continue
            if existing.is_viewpoint:
                break
            if (
                identity_key_norm is not None
                and existing.countable_instance
                and existing.identity_key
                and str(existing.identity_key) != identity_key_norm
            ):
                break
            if semantic_only and existing.countable_instance:
                break
            if existing.countable_instance or countable_instance:
                from emet.memory.graph_eqa.graph_stats import labels_compatible_for_dedup

                primary = str(existing.labels[0]).strip() if existing.labels else label
                if not labels_compatible_for_dedup(label, primary):
                    break
                merged_labels = sorted(
                    {
                        *[
                            str(x).strip()
                            for x in existing.labels
                            if str(x).strip() and labels_compatible_for_dedup(str(x).strip(), primary)
                        ],
                        label,
                    }
                )
            else:
                merged_labels = sorted({*(str(x).strip() for x in existing.labels if str(x).strip()), label})
            sc = int(existing.support_count) + 1
            new_xyz, covariance, history, changes, belief_confidence = self._position_update(
                existing, xyz_a, step=step
            )
            new_emb = embedding if embedding is not None else existing.embedding
            if embedding is not None and existing.embedding is not None:
                a = float(getattr(candidate, "embedding_blend_alpha", 0.35))
                new_emb = (1.0 - a) * np.asarray(existing.embedding, dtype=np.float32) + a * embedding
            new_bounds = bounds_3d if bounds_3d is not None else existing.bounds_3d
            if bounds_3d is not None and existing.bounds_3d is not None:
                mn = np.minimum(
                    np.asarray(existing.bounds_3d["min"], dtype=np.float64),
                    np.asarray(bounds_3d["min"], dtype=np.float64),
                )
                mx = np.maximum(
                    np.asarray(existing.bounds_3d["max"], dtype=np.float64),
                    np.asarray(bounds_3d["max"], dtype=np.float64),
                )
                c = 0.5 * (mn + mx)
                new_bounds = {
                    "min": mn.tolist(),
                    "max": mx.tolist(),
                    "center": c.tolist(),
                    "size": (mx - mn).tolist(),
                }
            self._nodes[idx] = replace(
                existing,
                xyz=new_xyz,
                labels=merged_labels,
                last_seen=step,
                support_count=sc,
                bbox_xyxy=bbox_i if bbox_i is not None else existing.bbox_xyxy,
                embedding=new_emb,
                bounds_3d=new_bounds,
                position_covariance=covariance,
                position_history=history,
                change_events=changes,
                belief_confidence=belief_confidence,
                identity_key=identity_key_norm or existing.identity_key,
                countable_instance=bool(existing.countable_instance or countable_instance),
                close_look_label=existing.close_look_label,
            )
            self.refresh_observation_candidate(
                int(existing.obs_id),
                rgb,
                xyz=new_xyz,
                labels=merged_labels,
                viewer_xyz=viewer_a,
            )
            if viewer_a is not None:
                self._ensure_viewpoint_node(int(existing.obs_id), viewer_a)
            self._update_edges()
            return int(existing.obs_id)

    obs_id = self.add_observation(
        rgb,
        xyz_a,
        [label],
        viewer_xyz=viewer_a,
        bbox_xyxy=bbox_i,
        identity_key=identity_key_norm,
        countable_instance=countable_instance,
    )
    for idx, n in enumerate(self._nodes):
        if int(n.obs_id) == int(obs_id) and not n.is_viewpoint:
            self._nodes[idx] = replace(
                n,
                embedding=embedding,
                bounds_3d=bounds_3d,
            )
            break
    return obs_id

def absorb_object_node(self, src_node_id: int, dst_node_id: int) -> bool:
    """Fold ``src`` object node into ``dst`` and remove ``src`` from the graph."""
    if int(src_node_id) == int(dst_node_id):
        return False
    src = dst = None
    for n in self._nodes:
        if int(n.node_id) == int(src_node_id):
            src = n
        elif int(n.node_id) == int(dst_node_id):
            dst = n
    if src is None or dst is None:
        return False
    if src.is_viewpoint or dst.is_viewpoint or src.is_frontier or dst.is_frontier:
        return False
    if (
        src.countable_instance
        and dst.countable_instance
        and src.identity_key
        and dst.identity_key
        and str(src.identity_key) != str(dst.identity_key)
    ):
        return False

    sc_src = int(src.support_count)
    sc_dst = int(dst.support_count)
    total = sc_src + sc_dst
    new_xyz, covariance, history, changes, belief_confidence = self._position_update(
        dst,
        np.asarray(src.xyz, dtype=float),
        step=max(int(dst.last_seen), int(src.last_seen)),
    )

    merged_labels = sorted(
        {
            *(str(x).strip() for x in dst.labels if str(x).strip()),
            *(str(x).strip() for x in src.labels if str(x).strip()),
        }
    )
    if not merged_labels:
        merged_labels = ["object"]

    new_emb = dst.embedding
    if src.embedding is not None and dst.embedding is not None:
        a = 0.35
        new_emb = (1.0 - a) * np.asarray(dst.embedding, dtype=np.float32) + a * np.asarray(
            src.embedding, dtype=np.float32
        )
    elif src.embedding is not None:
        new_emb = np.asarray(src.embedding, dtype=np.float32).copy()

    new_bounds = dst.bounds_3d
    if src.bounds_3d is not None and dst.bounds_3d is not None:
        mn = np.minimum(
            np.asarray(dst.bounds_3d["min"], dtype=np.float64),
            np.asarray(src.bounds_3d["min"], dtype=np.float64),
        )
        mx = np.maximum(
            np.asarray(dst.bounds_3d["max"], dtype=np.float64),
            np.asarray(src.bounds_3d["max"], dtype=np.float64),
        )
        c = 0.5 * (mn + mx)
        new_bounds = {
            "min": mn.tolist(),
            "max": mx.tolist(),
            "center": c.tolist(),
            "size": (mx - mn).tolist(),
        }
    elif src.bounds_3d is not None:
        new_bounds = src.bounds_3d

    dst_idx = next(i for i, n in enumerate(self._nodes) if int(n.node_id) == int(dst_node_id))
    self._nodes[dst_idx] = replace(
        dst,
        xyz=new_xyz,
        labels=merged_labels,
        support_count=total,
        embedding=new_emb,
        bounds_3d=new_bounds,
        bbox_xyxy=dst.bbox_xyxy or src.bbox_xyxy,
        last_seen=max(int(dst.last_seen), int(src.last_seen)),
        position_covariance=covariance,
        position_history=history,
        change_events=changes,
        belief_confidence=belief_confidence,
        identity_key=dst.identity_key or src.identity_key,
        countable_instance=bool(dst.countable_instance or src.countable_instance),
        close_look_label=dst.close_look_label or src.close_look_label,
    )
    for o in self._observations:
        if int(o.obs_id) == int(dst.obs_id):
            o.xyz = new_xyz.copy()
            o.labels = list(merged_labels)
            break

    src_obs_id = int(src.obs_id)
    self.world_evidence.absorb_entity(
        src_node_id=int(src_node_id),
        dst_node_id=int(dst_node_id),
    )
    self._nodes = [n for n in self._nodes if int(n.node_id) != int(src_node_id)]
    self._observations = [o for o in self._observations if int(o.obs_id) != src_obs_id]
    for i, n in enumerate(self._nodes, start=1):
        self._nodes[i - 1] = replace(n, node_id=i)
    self.world_evidence.reindex_entities(
        self._nodes,
        step=self._effective_timestep(),
    )
    self._rebuild_viewpoint_index()
    self._update_edges()
    return True

def upsert_ground_truth_observation(
    self,
    body_key: str,
    rgb: np.ndarray | Image.Image,
    xyz: np.ndarray,
    labels: list[str],
    *,
    extent_half: np.ndarray | None = None,
) -> int:
    """
    Insert or refresh one sim GT node keyed by MuJoCo ``body_key``.

    GT nodes use ``description=ground_truth:{body_key}`` so repeated updates
    deduplicate instead of adding duplicate detections over time.
    """
    if isinstance(rgb, Image.Image):
        rgb = np.array(rgb)
    step = self._effective_timestep()
    xyz_a = np.asarray(xyz, dtype=float).reshape(-1)[:3]
    labels_norm = [str(l).strip() for l in labels if str(l).strip()]
    if not labels_norm:
        labels_norm = ["object"]
    desc = f"{GT_BODY_DESC_PREFIX}{body_key}"
    ext = None
    if extent_half is not None:
        ext = np.asarray(extent_half, dtype=float).reshape(-1)[:3].copy()

    for idx, existing in enumerate(self._nodes):
        if existing.description != desc:
            continue
        same_pose = np.allclose(existing.xyz, xyz_a, atol=1e-4, rtol=0.0)
        same_labels = list(existing.labels) == labels_norm
        same_ext = ext is None or (
            existing.extent_half is not None and np.allclose(existing.extent_half, ext, atol=1e-4, rtol=0.0)
        )
        if same_pose and same_labels and same_ext:
            self._nodes[idx] = replace(existing, last_seen=step)
            self._update_edges()
            return int(existing.obs_id)
        sc = int(existing.support_count) + 1
        self._nodes[idx] = replace(
            existing,
            xyz=xyz_a.copy(),
            labels=labels_norm,
            last_seen=step,
            support_count=sc,
            extent_half=ext if ext is not None else existing.extent_half,
        )
        for o in self._observations:
            if o.obs_id == existing.obs_id:
                o.xyz = xyz_a.copy()
                o.labels = list(labels_norm)
                o.description = desc
                break
        self._update_edges()
        return int(existing.obs_id)

    return self.add_observation(
        rgb,
        xyz_a,
        labels_norm,
        description=desc,
        extent_half=ext,
    )

def attach_detection_to_ground_truth_node(
    self,
    body_key: str,
    rgb: np.ndarray | Image.Image,
    *,
    detection_label: str | None = None,
) -> bool:
    """Refresh a GT node's stored RGB when an instance detector sees it nearby."""
    if isinstance(rgb, Image.Image):
        rgb = np.array(rgb)
    rgb_a = np.asarray(rgb, dtype=np.uint8)
    desc = f"{GT_BODY_DESC_PREFIX}{body_key}"
    det_tag = f"|det:{detection_label.strip()}" if detection_label and detection_label.strip() else ""
    step = self._effective_timestep()
    for idx, existing in enumerate(self._nodes):
        if existing.description is None or not str(existing.description).startswith(desc):
            continue
        new_desc = f"{desc}{det_tag}" if det_tag else desc
        self._nodes[idx] = replace(existing, last_seen=step, description=new_desc)
        self.refresh_observation_candidate(
            int(existing.obs_id),
            rgb_a,
            description=new_desc,
        )
        self._update_edges()
        return True
    return False

def record_navigation_sample(
    self,
    rgb: np.ndarray | Image.Image,
    xyz: np.ndarray,
    *,
    base_xyz: np.ndarray | None = None,
    link_viewpoint_node: bool = True,
) -> None:
    """
    Record a navigation-time viewpoint without adding a scene-graph node.

    Used when perception returns no usable object labels (e.g. generic
    ``object`` fallback) so the trajectory is still available for debugging
    and optional EQA image context.
    """
    if not self._record_navigation:
        return
    if isinstance(rgb, Image.Image):
        rgb = np.array(rgb)
    rgb = np.asarray(rgb)
    xyz = np.asarray(xyz, dtype=float).reshape(-1)[:3]
    bx = None
    if base_xyz is not None:
        bx = np.asarray(base_xyz, dtype=float).reshape(-1)[:3]
    self._nav_samples.append(GraphNavigationSample(rgb=rgb, xyz=xyz, base_xyz=bx))
    if len(self._nav_samples) > self._nav_max:
        drop = len(self._nav_samples) - self._nav_max
        self._nav_samples = self._nav_samples[drop:]
    if bx is not None and link_viewpoint_node:
        nav_obs_id = self._next_obs_id
        self._next_obs_id += 1
        self._ensure_viewpoint_node(nav_obs_id, bx, labels=["viewpoint", "nav"])

def get_navigation_samples(self) -> list[GraphNavigationSample]:
    return list(self._nav_samples)

def _frontier_desc(self, cluster_id: str) -> str:
    from emet.memory.graph_eqa.spatial.frontier_nodes import FRONTIER_DESC_PREFIX

    return f"{FRONTIER_DESC_PREFIX}{cluster_id}"

def _find_frontier_node(self, cluster_id: str) -> GraphNode | None:
    desc = self._frontier_desc(cluster_id)
    for n in self._nodes:
        if n.is_frontier and n.description == desc:
            return n
    return None

def _remove_frontier_nodes(self, keep_cluster_ids: set[str]) -> None:
    from emet.memory.graph_eqa.spatial.frontier_nodes import FRONTIER_DESC_PREFIX

    drop_obs: set[int] = set()
    drop_nodes: set[int] = set()
    for n in self._nodes:
        if not n.is_frontier:
            continue
        desc = n.description or ""
        cid = desc[len(FRONTIER_DESC_PREFIX) :] if desc.startswith(FRONTIER_DESC_PREFIX) else ""
        if cid not in keep_cluster_ids:
            drop_obs.add(int(n.obs_id))
            drop_nodes.add(int(n.node_id))
    if not drop_nodes:
        return
    self._nodes = [n for n in self._nodes if int(n.node_id) not in drop_nodes]
    self._observations = [o for o in self._observations if int(o.obs_id) not in drop_obs]
    for i, n in enumerate(self._nodes, start=1):
        self._nodes[i - 1] = replace(n, node_id=i)
    self._reindex_world_entities()
    self._rebuild_viewpoint_index()
    self._update_edges()

def _frontier_support_attachment(
    self,
    *,
    grid_ij: tuple[int, int],
    xyz: np.ndarray,
    unexplored: np.ndarray,
    explored: np.ndarray,
    reachable: np.ndarray | None,
    step: int,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """Return real nearby RGB or an explicitly namespaced map crop."""
    nearest_view = None
    nearest_distance = float("inf")
    if self.world_evidence.enabled:
        for view in self.world_evidence.views.values():
            if view.rgb is None:
                continue
            anchor = view.base_pose_world or view.object_xyz
            distance = float(np.linalg.norm(np.asarray(anchor[:2], dtype=float) - np.asarray(xyz[:2], dtype=float)))
            if distance < nearest_distance:
                nearest_view, nearest_distance = view, distance
    if nearest_view is not None and nearest_distance <= 5.0:
        return (
            np.asarray(nearest_view.rgb, dtype=np.uint8).copy(),
            (nearest_view.view_id,),
            (f"view:{nearest_view.view_id}",),
        )

    row, col = int(grid_ij[0]), int(grid_ij[1])
    radius = 12
    r0, r1 = max(0, row - radius), min(unexplored.shape[0], row + radius + 1)
    c0, c1 = max(0, col - radius), min(unexplored.shape[1], col + radius + 1)
    crop = np.full((r1 - r0, c1 - c0, 3), 32, dtype=np.uint8)
    crop[np.asarray(explored[r0:r1, c0:c1], dtype=bool)] = (90, 90, 90)
    if reachable is not None:
        crop[np.asarray(reachable[r0:r1, c0:c1], dtype=bool)] = (70, 100, 160)
    crop[np.asarray(unexplored[r0:r1, c0:c1], dtype=bool)] = (40, 210, 100)
    if crop.size == 0:
        crop = np.full((8, 8, 3), (40, 210, 100), dtype=np.uint8)
    scale = max(1, int(np.ceil(64 / max(crop.shape[:2]))))
    crop = np.repeat(np.repeat(crop, scale, axis=0), scale, axis=1)[:96, :96]
    attachment_id = f"map:grid-{row}-{col}:step-{int(step)}"
    return crop, (), (attachment_id,)

def sync_frontier_nodes(
    self,
    voxel_map: Any,
    planner: Any,
    xyt: Any,
    *,
    question_keywords: list[str] | None = None,
) -> int:
    """Upsert/remove frontier graph nodes from the voxel unexplored-frontier mask."""
    if not self.frontier_nodes_enabled:
        return sum(1 for n in self._nodes if n.is_frontier)

    from emet.memory.graph_eqa.spatial.frontier_nodes import (
        _as_bool_numpy,
        cluster_frontier_mask,
        frontier_components,
        hint_labels_near_grid,
        keyword_overlap_score,
    )

    try:
        outside = voxel_map.get_outside_frontier(xyt, planner)
        _, explored = voxel_map.get_2d_map()
        reachable = None
        if hasattr(voxel_map, "get_reachable_map"):
            reachable = _as_bool_numpy(voxel_map.get_reachable_map(xyt, planner))
    except Exception as e:
        _logger.warning(f"Frontier upsert: could not read map frontiers ({e})")
        return sum(1 for n in self._nodes if n.is_frontier)

    unexplored = _as_bool_numpy(outside) & ~_as_bool_numpy(explored)
    components = (
        frontier_components(
            unexplored,
            min_cells=self._frontier_min_cluster_cells,
            reachable=reachable,
        )
        if self.world_evidence.enabled
        else None
    )
    clusters = (
        [(component.transient_id, component.goal_ij, component.cell_count) for component in components]
        if components is not None
        else cluster_frontier_mask(
            unexplored,
            min_cells=self._frontier_min_cluster_cells,
            reachable=reachable,
        )
    )
    component_by_id = {component.transient_id: component for component in (components or ())}
    image_descriptions = getattr(voxel_map, "image_descriptions", None) or []
    keywords = list(question_keywords or self._relevant_objects or self._enrich_object_hints or [])

    scored: list[tuple[float, str, tuple[int, int], int]] = []
    for cluster_id, grid_ij, cell_count in clusters:
        hints = hint_labels_near_grid(grid_ij, image_descriptions)
        kw_score = keyword_overlap_score(hints, keywords) if keywords else 0.0
        scored.append((kw_score, cluster_id, grid_ij, cell_count))
    scored.sort(key=lambda x: (-x[0], -x[3]))

    step = self._effective_timestep()
    prepared: list[dict[str, Any]] = []
    for kw_score, transient_id, grid_ij, cell_count in scored[: self._frontier_max_nodes]:
        gi, gj = grid_ij
        try:
            xy = voxel_map.grid_coords_to_xy(np.array([gi, gj], dtype=float))
        except Exception:
            continue
        xyz = np.array([float(xy[0]), float(xy[1]), 0.0], dtype=float)
        hints = hint_labels_near_grid(grid_ij, image_descriptions)
        support_rgb, support_view_ids, attachment_ids = self._frontier_support_attachment(
            grid_ij=grid_ij,
            xyz=xyz,
            unexplored=unexplored,
            explored=_as_bool_numpy(explored),
            reachable=reachable,
            step=step,
        )
        component = component_by_id.get(transient_id)
        prepared.append(
            {
                "transient_id": transient_id,
                "grid_ij": grid_ij,
                "cell_count": int(cell_count),
                "cells": (component.cells if component is not None else ((int(grid_ij[0]), int(grid_ij[1])),)),
                "centroid_xyz": xyz,
                "keyword_score": float(kw_score),
                "hints": hints,
                "support_rgb": support_rgb,
                "support_view_ids": support_view_ids,
                "attachment_ids": attachment_ids,
            }
        )

    tracks = self.world_evidence.update_frontier_tracks(prepared, step=step) if self.world_evidence.enabled else []
    keep_ids: set[str] = set()
    for index, item in enumerate(prepared):
        track = tracks[index] if index < len(tracks) else None
        cluster_id = track.frontier_id if track is not None else str(item["transient_id"])
        keep_ids.add(cluster_id)
        xyz = np.asarray(item["centroid_xyz"], dtype=float)
        hints = list(item["hints"])
        cell_count = int(item["cell_count"])
        kw_score = float(item["keyword_score"])
        labels = ["frontier"] + hints[:3]
        desc = self._frontier_desc(cluster_id)
        obs_desc = (
            f"frontier_id={cluster_id}; attachments="
            + ",".join(item["attachment_ids"])
            + ("; unexplored areas; " + ", ".join(hints) if hints else "; unexplored space")
            if hints
            else f"frontier_id={cluster_id}; attachments=" + ",".join(item["attachment_ids"]) + "; unexplored space"
        )

        existing = self._find_frontier_node(cluster_id)
        if existing is not None:
            idx = next(i for i, n in enumerate(self._nodes) if n.node_id == existing.node_id)
            self._nodes[idx] = replace(
                existing,
                xyz=xyz,
                labels=labels,
                last_seen=step,
                description=desc,
                frontier_cell_count=int(cell_count),
                frontier_keyword_score=float(kw_score),
            )
            for o in self._observations:
                if int(o.obs_id) == int(existing.obs_id):
                    o.xyz = xyz.copy()
                    o.labels = list(labels)
                    o.description = obs_desc
                    o.rgb = np.asarray(item["support_rgb"], dtype=np.uint8).copy()
                    break
            if track is not None:
                self.world_evidence.set_frontier_obs(track.frontier_id, int(existing.obs_id))
        else:
            obs_id = self._next_obs_id
            self._next_obs_id += 1
            node_id = len(self._nodes) + 1
            self._nodes.append(
                GraphNode(
                    node_id=node_id,
                    labels=labels,
                    xyz=xyz.copy(),
                    obs_id=obs_id,
                    description=desc,
                    last_seen=step,
                    is_frontier=True,
                    frontier_cell_count=int(cell_count),
                    frontier_keyword_score=float(kw_score),
                )
            )
            self._observations.append(
                GraphObservation(
                    obs_id=obs_id,
                    rgb=np.asarray(item["support_rgb"], dtype=np.uint8).copy(),
                    xyz=xyz.copy(),
                    labels=list(labels),
                    description=obs_desc,
                )
            )
            if track is not None:
                self.world_evidence.set_frontier_obs(track.frontier_id, obs_id)

    self._remove_frontier_nodes(keep_ids)
    self._update_edges()
    return sum(1 for n in self._nodes if n.is_frontier)

def _rebuild_viewpoint_index(self) -> None:
    self._viewpoint_by_obs_id = {int(n.obs_id): int(n.node_id) for n in self._nodes if n.is_viewpoint}

def _find_nearby_viewpoint_node(self, viewer_xyz: np.ndarray) -> GraphNode | None:
    """Nearest viewpoint within ``viewpoint_merge_m`` (stationary-stream dedup)."""
    radius = float(self.viewpoint_merge_m)
    if radius <= 0.0:
        return None
    vxyz = np.asarray(viewer_xyz, dtype=float).reshape(-1)[:3]
    best: GraphNode | None = None
    best_d = float("inf")
    for n in self._nodes:
        if not n.is_viewpoint:
            continue
        d = float(np.linalg.norm(np.asarray(n.xyz, dtype=float).reshape(3) - vxyz))
        if d <= radius and d < best_d:
            best_d = d
            best = n
    return best

def _ensure_viewpoint_node(
    self,
    obs_id: int,
    viewer_xyz: np.ndarray,
    *,
    labels: list[str] | None = None,
) -> int:
    """Create or refresh a graph node at the observation vantage (``seen_from`` target)."""
    vxyz = np.asarray(viewer_xyz, dtype=float).reshape(-1)[:3].copy()
    step = self._effective_timestep()
    vp_labels = labels or [f"view img {int(obs_id)}"]
    existing_id = self._viewpoint_by_obs_id.get(int(obs_id))
    if existing_id is not None:
        for idx, n in enumerate(self._nodes):
            if int(n.node_id) == int(existing_id):
                self._nodes[idx] = replace(
                    n,
                    xyz=vxyz,
                    labels=list(vp_labels),
                    last_seen=step,
                )
                return int(existing_id)
    nearby = self._find_nearby_viewpoint_node(vxyz)
    if nearby is not None:
        for idx, n in enumerate(self._nodes):
            if int(n.node_id) == int(nearby.node_id):
                self._nodes[idx] = replace(
                    n,
                    xyz=vxyz,
                    labels=list(vp_labels),
                    last_seen=step,
                )
                self._viewpoint_by_obs_id[int(obs_id)] = int(nearby.node_id)
                return int(nearby.node_id)
    node_id = len(self._nodes) + 1
    self._nodes.append(
        GraphNode(
            node_id=node_id,
            labels=list(vp_labels),
            xyz=vxyz,
            obs_id=int(obs_id),
            last_seen=step,
            is_viewpoint=True,
        )
    )
    self._viewpoint_by_obs_id[int(obs_id)] = int(node_id)
    return int(node_id)

def _ensure_seen_from_edge(self, node_id: int, obs_id: int) -> None:
    """Link object *node_id* to the viewpoint graph node for observation *obs_id*."""
    vp_id = self._viewpoint_by_obs_id.get(int(obs_id))
    if vp_id is None:
        return
    edge = (int(node_id), int(vp_id), "seen_from")
    if edge not in self._edges:
        self._edges.append(edge)

def _observation_by_id(self, obs_id: int) -> GraphObservation | None:
    for o in self._observations:
        if int(o.obs_id) == int(obs_id):
            return o
    return None

def _update_edges(self) -> None:
    """Compute spatial/context relations and timestamp their uncertain evidence."""
    self._edges.clear()
    objects = [n for n in self._nodes if not n.is_viewpoint and not n.is_frontier]
    viewpoints = [n for n in self._nodes if n.is_viewpoint]
    for i, na in enumerate(objects):
        if _on_floor(na.xyz):
            self._edges.append((na.node_id, -1, "on"))  # -1 = floor
        for j, nb in enumerate(objects):
            if i >= j:
                continue
            if _near(na.xyz, nb.xyz, self.max_near_distance):
                if (nb.node_id, na.node_id, "near") not in self._edges:
                    self._edges.append((na.node_id, nb.node_id, "near"))
            if _on(na.xyz, nb.xyz):
                self._edges.append((na.node_id, nb.node_id, "on"))
                self._edges.append((nb.node_id, na.node_id, "supports"))
            elif _on(nb.xyz, na.xyz):
                self._edges.append((nb.node_id, na.node_id, "on"))
                self._edges.append((na.node_id, nb.node_id, "supports"))
            if _node_is_room(na) and _inside_bounds(nb.xyz, na.bounds_3d):
                self._edges.append((na.node_id, nb.node_id, "contains"))
            elif _node_is_room(nb) and _inside_bounds(na.xyz, nb.bounds_3d):
                self._edges.append((nb.node_id, na.node_id, "contains"))
    for node in objects:
        self._ensure_seen_from_edge(node.node_id, int(node.obs_id))
        if viewpoints:
            nearest = min(
                viewpoints,
                key=lambda view: float(
                    np.linalg.norm(np.asarray(view.xyz, dtype=float)[:2] - np.asarray(node.xyz, dtype=float)[:2])
                ),
            )
            distance = float(
                np.linalg.norm(np.asarray(nearest.xyz, dtype=float)[:2] - np.asarray(node.xyz, dtype=float)[:2])
            )
            failure_risk = float(node.nav_failures) / max(1, int(node.nav_attempts))
            if distance <= max(2.0, self.max_near_distance) and failure_risk < 0.8:
                self._edges.append((node.node_id, nearest.node_id, "accessible_from"))

    step = self._effective_timestep()
    prior = self._relation_beliefs
    current: dict[tuple[int, int, str], RelationBelief] = {}
    confidence_by_relation = {
        "seen_from": 0.95,
        "contains": 0.85,
        "supports": 0.80,
        "on": 0.75,
        "near": 0.65,
        "accessible_from": 0.60,
    }
    for edge in self._edges:
        old = prior.get(edge)
        current[edge] = RelationBelief(
            source_id=edge[0],
            target_id=edge[1],
            relation=edge[2],
            confidence=max(
                confidence_by_relation.get(edge[2], 0.5),
                float(old.confidence) if old is not None else 0.0,
            ),
            last_evidence_step=step,
            contradiction_count=old.contradiction_count if old is not None else 0,
        )
    for edge, old in prior.items():
        if edge in current:
            continue
        decayed = float(old.confidence) * 0.5
        if decayed >= 0.1:
            current[edge] = replace(
                old,
                confidence=decayed,
                contradiction_count=int(old.contradiction_count) + 1,
            )
    self._relation_beliefs = current
    self.refresh_room_clusters()
