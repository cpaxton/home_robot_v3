# Copyright (c) Hello Robot, Inc.
# All rights reserved.

"""Spatial + embedding fusion for graph instance detections."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from emet.memory.graph_eqa.graph_memory import GraphEQAMemory, GraphNode
from emet.memory.graph_eqa.graph_object_fusion.config import GraphObjectFusionConfig
from emet.memory.graph_eqa.graph_stats import labels_compatible_for_dedup


@dataclass
class GraphDetectionCandidate:
    label: str
    xyz: np.ndarray
    bbox_xyxy: tuple[int, int, int, int] | None = None
    bounds_3d: dict[str, list[float]] | None = None
    embedding: np.ndarray | None = None
    identity_key: str | None = None
    countable_instance: bool = False
    detection_score: float | None = None
    mask_point_count: int = 0
    semantic_only: bool = False


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
    def _step(graph_memory: GraphEQAMemory) -> int:
        try:
            return int(graph_memory._effective_timestep())
        except Exception:
            return 0

    @staticmethod
    def _identity_compatible(node: GraphNode, candidate: GraphDetectionCandidate) -> bool:
        candidate_key = str(candidate.identity_key).strip() if candidate.identity_key else None
        node_key = str(node.identity_key).strip() if node.identity_key else None
        if candidate_key is None or not node.countable_instance or node_key is None:
            return True
        return candidate_key == node_key

    def _label_incompatible(self, a: str, b: str) -> bool:
        """True when a config ``labels.incompatible`` pair forbids merging a and b."""
        for pair in self.config.labels.incompatible:
            if a in pair and b in pair and a != b:
                return True
        return False

    def _label_match(self, node: GraphNode, label: str, candidate: GraphDetectionCandidate | None = None) -> bool:
        instance_guard = bool(self.config.require_label_match_for_instances) and (
            (candidate is not None and candidate.countable_instance) or bool(node.countable_instance)
        )
        if not self.config.require_label_match and not instance_guard:
            return True
        for x in node.labels:
            if self._label_incompatible(label, str(x or "")):
                return False
            if labels_compatible_for_dedup(label, str(x or "")):
                return True
        # Extra synonym groups from config (fixture dilution, terse queries).
        for group in self.config.labels.synonyms:
            if label in group and any(str(x or "") in group for x in node.labels):
                return True
        return False

    def _within_temporal_window(self, node: GraphNode, step: int) -> bool:
        """False when the growth window is on and the node went stale before it."""
        window = int(self.config.growth.temporal_window_steps)
        if window <= 0:
            return True
        return (int(step) - int(getattr(node, "last_seen", 0))) <= window

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

    def _embedding_cosine(self, node: GraphNode, candidate: GraphDetectionCandidate) -> float | None:
        if candidate.embedding is None or node.embedding is None:
            return None
        return cosine_similarity_np(node.embedding, candidate.embedding)

    def _appearance_ok(self, node: GraphNode, candidate: GraphDetectionCandidate) -> bool:
        """Appearance override: same-looking object near the candidate may merge even
        when the label gate fails (YoloE labels drift frame-to-frame)."""
        if not self.config.gates.embedding.on:
            return False
        thr = float(self.config.gates.embedding.appearance_merge_min_cosine)
        if thr <= 0.0:
            return False
        cos = self._embedding_cosine(node, candidate)
        return cos is not None and cos >= thr

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
            if not self._within_temporal_window(node, self._step(graph_memory)):
                continue
            if not self._identity_compatible(node, candidate):
                continue
            if not self._label_match(node, candidate.label, candidate) and not self._appearance_ok(node, candidate):
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
            if not self._within_temporal_window(node, self._step(graph_memory)):
                continue
            if not self._identity_compatible(node, candidate):
                continue
            if not self._label_match(node, candidate.label, candidate) and not self._appearance_ok(node, candidate):
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
            if not self._within_temporal_window(node, self._step(graph_memory)):
                continue
            if not self._identity_compatible(node, candidate):
                continue
            if not self._label_match(node, candidate.label, candidate) and not self._appearance_ok(node, candidate):
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
    ) -> int | None:
        """Merge into an existing node or add a new observation."""
        match = self.find_best_node(graph_memory, candidate)
        if match is None:
            match = self.find_iou_merge_node(graph_memory, candidate)
        if match is None:
            match = self.find_fallback_node(graph_memory, candidate)
        if match is not None:
            label_ok = self._label_match(match, candidate.label, candidate)
            obs_id = graph_memory.merge_object_detection(
                rgb,
                candidate,
                merge_into_node_id=int(match.node_id),
                viewer_xyz=viewer_xyz,
                blend=self.config.keep,
                allow_label_mismatch=not label_ok,
            )
            stats = getattr(graph_memory, "instance_ingest_stats", None)
            if isinstance(stats, dict):
                stats["merged"] = int(stats.get("merged", 0)) + 1
            return obs_id
        cap = int(getattr(self.config.growth, "max_object_nodes", 0) or 0)
        if cap > 0:
            n_objects = sum(
                1
                for node in graph_memory.get_nodes()
                if not node.is_viewpoint and not getattr(node, "is_frontier", False)
            )
            if n_objects >= cap:
                stats = getattr(graph_memory, "instance_ingest_stats", None)
                if isinstance(stats, dict):
                    stats["rejected_object_cap"] = int(stats.get("rejected_object_cap", 0)) + 1
                return None
        obs_id = graph_memory.merge_object_detection(
            rgb, candidate, merge_into_node_id=None, viewer_xyz=viewer_xyz, blend=self.config.keep
        )
        stats = getattr(graph_memory, "instance_ingest_stats", None)
        if isinstance(stats, dict):
            stats["created"] = int(stats.get("created", 0)) + 1
        return obs_id

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
                if self.config.require_label_match_for_instances and (a.countable_instance or b.countable_instance):
                    from emet.memory.graph_eqa.graph_stats import labels_compatible_for_dedup

                    la = str(a.labels[0]) if a.labels else ""
                    lb = str(b.labels[0]) if b.labels else ""
                    if la and lb and not labels_compatible_for_dedup(la, lb):
                        continue
                if bounds_3d_iou(a.bounds_3d, b.bounds_3d) < thr:
                    continue
                if bool(self.config.keep.prefer_support):
                    keep, lose = (a, b) if int(a.support_count) >= int(b.support_count) else (b, a)
                else:
                    keep, lose = (a, b) if int(a.last_seen) >= int(b.last_seen) else (b, a)
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
