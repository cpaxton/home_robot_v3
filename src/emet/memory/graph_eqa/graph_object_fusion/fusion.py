# Copyright (c) Hello Robot, Inc.
# All rights reserved.

"""Spatial + embedding fusion for graph instance detections.

Merges repeated YoloE/instance-mask detections into a single ``GraphNode`` when
centroids, 3D bounds, and optional embeddings agree within configured thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from emet.memory.graph_eqa.graph_memory import GraphEQAMemory, GraphNode
from emet.memory.graph_eqa.graph_object_fusion.config import GraphObjectFusionConfig


@dataclass
class GraphDetectionCandidate:
    """One instance detection lifted from a frame before graph merge.

    Attributes:
        label: Open-vocab category string (e.g. from YoloE).
        xyz: World centroid ``(3,)`` in meters.
        bbox_xyxy: Optional pixel crop ``(x0, y0, x1, y1)`` in the source RGB.
        bounds_3d: Optional axis-aligned world AABB
            ``{min, max, center, size}`` from instance point cloud.
        embedding: Optional visual embedding (e.g. SigLIP crop); ``None`` skips
            the embedding gate at merge time.
    """

    label: str
    xyz: np.ndarray
    bbox_xyxy: tuple[int, int, int, int] | None = None
    bounds_3d: dict[str, list[float]] | None = None
    embedding: np.ndarray | None = None


def bounds_3d_from_points(points: np.ndarray) -> dict[str, list[float]]:
    """Build an axis-aligned world bounds dict from an ``(N, 3)`` point cloud.

    Args:
        points: Instance or crop points in world coordinates.

    Returns:
        Dict with ``min``, ``max``, ``center``, and ``size`` (edge lengths in m).
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    c = 0.5 * (mn + mx)
    size = (mx - mn).tolist()
    return {"center": c.tolist(), "size": size, "min": mn.tolist(), "max": mx.tolist()}


def bounds_3d_iou(a: dict[str, list[float]] | None, b: dict[str, list[float]] | None) -> float:
    """Intersection-over-union of two axis-aligned 3D boxes.

    Args:
        a: First bounds dict with ``min`` and ``max`` keys (length-3 lists).
        b: Second bounds dict with the same schema.

    Returns:
        IoU in ``[0, 1]``, or ``0.0`` when either input is ``None`` or boxes
        do not overlap.
    """
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
    """Cosine similarity between two 1D embedding vectors.

    Args:
        a: First embedding vector, or ``None``.
        b: Second embedding vector, or ``None``.

    Returns:
        Dot product divided by L2 norms, or ``0.0`` when inputs are missing,
        empty, or shape-mismatched.
    """
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
    """Merge instance detections into existing graph nodes by geometry (+ optional embedding).

    When ``GraphObjectFusionConfig.enabled`` is true, Dynagraph routes instance
    detections through this class instead of legacy label-only dedup /
    ``dynagraph_merge_xy_m`` on the instance path.
    """

    def __init__(self, config: GraphObjectFusionConfig | None = None) -> None:
        """Args:
            config: Fusion thresholds; uses ``GraphObjectFusionConfig()`` defaults when omitted.
        """
        self.config = config or GraphObjectFusionConfig()

    def _label_match(self, node: GraphNode, label: str) -> bool:
        """Return whether ``label`` matches ``node`` per ``require_label_match``.

        When ``require_label_match`` is false, always returns ``True``. Otherwise
        requires exact or substring match against any entry in ``node.labels``.
        """
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
        """Return whether ``xyz`` / ``bounds`` are close enough to ``node`` to merge.

        Gates on planar distance (``spatial_merge_xy_m``), 3D centroid distance
        (``min_centroid_dist_m``), and optional 3D bounds IoU
        (``bounds_3d_iou_min``) when both sides have ``bounds_3d``.
        """
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
        """Return whether ``embedding`` is similar enough to ``node.embedding``.

        When either side has no embedding, the gate passes (spatial/bounds only).
        """
        if embedding is None or node.embedding is None:
            return True
        return cosine_similarity_np(node.embedding, embedding) >= self.config.embedding_min_cosine

    def find_best_node(
        self,
        graph_memory: GraphEQAMemory,
        candidate: GraphDetectionCandidate,
    ) -> GraphNode | None:
        """Pick the highest-scoring existing node to merge ``candidate`` into.

        Scans non-viewpoint nodes, applies label/spatial/embedding gates, and
        ranks survivors by XY proximity, bounds IoU, and embedding cosine.

        Args:
            graph_memory: Live graph to search.
            candidate: New detection to associate.

        Returns:
            Best matching ``GraphNode``, or ``None`` if no node passes all gates.
        """
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

    def apply_detection(
        self,
        graph_memory: GraphEQAMemory,
        rgb: np.ndarray,
        candidate: GraphDetectionCandidate,
        *,
        viewer_xyz: np.ndarray | None = None,
    ) -> int:
        """Merge ``candidate`` into the best node or create a new graph observation.

        Args:
            graph_memory: Graph to update.
            rgb: Source RGB frame (stored on the observation when creating/merging).
            candidate: Detection payload.
            viewer_xyz: Optional camera/world vantage for ``seen_from`` edges.

        Returns:
            ``node_id`` of the merged or newly created object node.
        """
        match = self.find_best_node(graph_memory, candidate)
        if match is not None:
            return graph_memory.merge_object_detection(
                rgb,
                candidate,
                merge_into_node_id=int(match.node_id),
                viewer_xyz=viewer_xyz,
            )
        return graph_memory.merge_object_detection(rgb, candidate, merge_into_node_id=None, viewer_xyz=viewer_xyz)
