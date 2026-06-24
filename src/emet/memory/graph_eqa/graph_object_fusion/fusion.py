# Copyright (c) Hello Robot, Inc.
# All rights reserved.

"""Spatial + embedding fusion for graph instance detections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from emet.memory.graph_eqa.graph_memory import GraphEQAMemory, GraphNode
from emet.memory.graph_eqa.graph_object_fusion.config import GraphObjectFusionConfig


@dataclass
class GraphDetectionCandidate:
    label: str
    xyz: np.ndarray
    bbox_xyxy: tuple[int, int, int, int] | None = None
    bounds_3d: dict[str, list[float]] | None = None
    embedding: np.ndarray | None = None


def bounds_3d_from_points(points: np.ndarray) -> dict[str, list[float]]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    c = 0.5 * (mn + mx)
    size = (mx - mn).tolist()
    return {"center": c.tolist(), "size": size, "min": mn.tolist(), "max": mx.tolist()}


def bounds_3d_iou(a: dict[str, list[float]] | None, b: dict[str, list[float]] | None) -> float:
    if a is None or b is None:
        return 0.0
    amin = np.asarray(a["min"], dtype=np.float64).reshape(3)
    amax = np.asarray(a["max"], dtype=np.float64).reshape(3)
    bmin = np.asarray(b["min"], dtype=np.float64).reshape(3)
    bmax = np.asarray(b["max"], dtype=np.float64).reshape(3)
    lo = np.maximum(amin, bmin)
    hi = np.minimum(amax, bmax)
    if np.any(hi <= lo):
        return 0.0
    inter = float(np.prod(hi - lo))
    va = float(np.prod(amax - amin))
    vb = float(np.prod(bmax - bmin))
    union = va + vb - inter
    if union <= 0:
        return 0.0
    return inter / union


def cosine_similarity_np(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    va = np.asarray(a, dtype=np.float64).reshape(-1)
    vb = np.asarray(b, dtype=np.float64).reshape(-1)
    if va.size == 0 or vb.size == 0 or va.shape != vb.shape:
        return 0.0
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


class GraphObjectFusion:
    def __init__(self, config: GraphObjectFusionConfig | None = None):
        self.config = config or GraphObjectFusionConfig()

    def _label_match(self, node: GraphNode, label: str) -> bool:
        if not self.config.require_label_match:
            return True
        lb = label.strip().lower()
        for x in node.labels:
            if (x or "").strip().lower() == lb:
                return True
            if lb in (x or "").strip().lower():
                return True
        return False

    def _spatial_ok(self, node: GraphNode, xyz: np.ndarray, bounds: dict[str, list[float]] | None) -> bool:
        nxy = np.asarray(node.xyz, dtype=np.float64).reshape(3)
        dxy = float(np.linalg.norm(nxy[:2] - xyz[:2]))
        if dxy > self.config.spatial_merge_xy_m:
            return False
        d3 = float(np.linalg.norm(nxy - xyz))
        if d3 > self.config.min_centroid_dist_m:
            return False
        if bounds is not None and node.bounds_3d is not None:
            if bounds_3d_iou(bounds, node.bounds_3d) < self.config.bounds_3d_iou_min:
                return False
        return True

    def _embedding_ok(
        self,
        node: GraphNode,
        embedding: np.ndarray | None,
    ) -> bool:
        if embedding is None or node.embedding is None:
            return True
        return cosine_similarity_np(node.embedding, embedding) >= self.config.embedding_min_cosine

    def find_best_node(
        self,
        graph_memory: GraphEQAMemory,
        candidate: GraphDetectionCandidate,
    ) -> GraphNode | None:
        xyz = np.asarray(candidate.xyz, dtype=np.float64).reshape(3)
        best: GraphNode | None = None
        best_score = -1.0
        cfg = self.config

        for node in graph_memory.get_nodes():
            if node.is_viewpoint:
                continue
            if not self._label_match(node, candidate.label):
                continue
            if not self._spatial_ok(node, xyz, candidate.bounds_3d):
                continue
            if not self._embedding_ok(node, candidate.embedding):
                continue
            score = 1.0 - float(np.linalg.norm(node.xyz[:2] - xyz[:2])) / max(cfg.spatial_merge_xy_m, 1e-6)
            if candidate.bounds_3d is not None and node.bounds_3d is not None:
                score = max(score, bounds_3d_iou(candidate.bounds_3d, node.bounds_3d))
            if candidate.embedding is not None and node.embedding is not None:
                score = max(score, cosine_similarity_np(node.embedding, candidate.embedding))
            if score > best_score:
                best_score = score
                best = node
        return best

    def find_fallback_node(
        self,
        graph_memory: GraphEQAMemory,
        candidate: GraphDetectionCandidate,
    ) -> GraphNode | None:
        """Nearest-neighbor XY merge when strict spatial/embedding/bounds gates fail."""
        radius = float(self.config.fallback_spatial_merge_xy_m)
        if radius <= 0.0:
            return None
        xyz = np.asarray(candidate.xyz, dtype=np.float64).reshape(3)
        best: GraphNode | None = None
        best_dxy = float("inf")
        for node in graph_memory.get_nodes():
            if node.is_viewpoint:
                continue
            if not self._label_match(node, candidate.label):
                continue
            nxy = np.asarray(node.xyz, dtype=np.float64).reshape(3)
            dxy = float(np.linalg.norm(nxy[:2] - xyz[:2]))
            if dxy > radius or dxy >= best_dxy:
                continue
            best_dxy = dxy
            best = node
        return best

    def apply_detection(
        self,
        graph_memory: GraphEQAMemory,
        rgb: np.ndarray,
        candidate: GraphDetectionCandidate,
        *,
        viewer_xyz: np.ndarray | None = None,
    ) -> int:
        """Merge into an existing node or add a new observation."""
        match = self.find_best_node(graph_memory, candidate)
        if match is None:
            match = self.find_fallback_node(graph_memory, candidate)
        if match is not None:
            return graph_memory.merge_object_detection(
                rgb,
                candidate,
                merge_into_node_id=int(match.node_id),
                viewer_xyz=viewer_xyz,
            )
        return graph_memory.merge_object_detection(rgb, candidate, merge_into_node_id=None, viewer_xyz=viewer_xyz)
