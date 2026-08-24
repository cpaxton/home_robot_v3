# Copyright (c) Hello Robot, Inc.
# All rights reserved.

"""Spatial + embedding fusion for graph instance detections."""

from __future__ import annotations

from dataclasses import dataclass

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
    identity_key: str | None = None
    countable_instance: bool = False


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

    @staticmethod
    def _identity_compatible(node: GraphNode, candidate: GraphDetectionCandidate) -> bool:
        candidate_key = str(candidate.identity_key).strip() if candidate.identity_key else None
        node_key = str(node.identity_key).strip() if node.identity_key else None
        if candidate_key is None or not node.countable_instance or node_key is None:
            return True
        return candidate_key == node_key

    def _label_match(self, node: GraphNode, label: str) -> bool:
        if not self.config.require_label_match:
            return True
        from emet.memory.graph_eqa.graph_stats import labels_compatible_for_dedup

        for x in node.labels:
            if labels_compatible_for_dedup(label, str(x or "")):
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

        if candidate.identity_key:
            identity_key = str(candidate.identity_key).strip()
            for node in graph_memory.get_nodes():
                if (
                    not node.is_viewpoint
                    and not node.is_frontier
                    and node.countable_instance
                    and str(node.identity_key or "").strip() == identity_key
                ):
                    return node

        for node in graph_memory.get_nodes():
            if node.is_viewpoint:
                continue
            if not self._identity_compatible(node, candidate):
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

    def find_iou_merge_node(
        self,
        graph_memory: GraphEQAMemory,
        candidate: GraphDetectionCandidate,
    ) -> GraphNode | None:
        """Merge when axis-aligned 3D bounds mostly overlap (duplicate views / depth jitter)."""
        thr = float(self.config.bounds_3d_iou_merge_min)
        if thr <= 0.0 or candidate.bounds_3d is None:
            return None
        best: GraphNode | None = None
        best_iou = thr
        for node in graph_memory.get_nodes():
            if node.is_viewpoint or node.bounds_3d is None:
                continue
            if not self._identity_compatible(node, candidate):
                continue
            if not self._label_match(node, candidate.label):
                continue
            iou = bounds_3d_iou(candidate.bounds_3d, node.bounds_3d)
            if iou >= best_iou:
                best_iou = iou
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
            if not self._identity_compatible(node, candidate):
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
            match = self.find_iou_merge_node(graph_memory, candidate)
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

    def consolidate_high_iou_nodes(self, graph_memory: GraphEQAMemory) -> int:
        """Fold object pairs whose 3D bounds mostly overlap (returns nodes absorbed)."""
        thr = float(self.config.bounds_3d_iou_merge_min)
        if thr <= 0.0:
            return 0
        objs = [
            n
            for n in graph_memory.get_nodes()
            if not n.is_viewpoint and not getattr(n, "is_frontier", False) and n.bounds_3d is not None
        ]
        drop: set[int] = set()
        merged = 0
        for i, a in enumerate(objs):
            if int(a.node_id) in drop:
                continue
            for b in objs[i + 1 :]:
                if int(b.node_id) in drop:
                    continue
                if (
                    a.countable_instance
                    and b.countable_instance
                    and a.identity_key
                    and b.identity_key
                    and str(a.identity_key) != str(b.identity_key)
                ):
                    continue
                if bounds_3d_iou(a.bounds_3d, b.bounds_3d) < thr:
                    continue
                keep, lose = (a, b) if int(a.support_count) >= int(b.support_count) else (b, a)
                if graph_memory.absorb_object_node(int(lose.node_id), int(keep.node_id)):
                    drop.add(int(lose.node_id))
                    merged += 1
                    if int(keep.node_id) == int(a.node_id):
                        a = keep
        return merged


def max_pairwise_object_bounds_iou(graph_memory: GraphEQAMemory) -> float | None:
    """Max 3D bounds IoU across object nodes (None when <2 bounded objects)."""
    objs = [
        n
        for n in graph_memory.get_nodes()
        if not n.is_viewpoint and not getattr(n, "is_frontier", False) and n.bounds_3d is not None
    ]
    if len(objs) < 2:
        return None
    best = 0.0
    for i, a in enumerate(objs):
        for b in objs[i + 1 :]:
            best = max(best, bounds_3d_iou(a.bounds_3d, b.bounds_3d))
    return float(best)
