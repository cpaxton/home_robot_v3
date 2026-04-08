# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Open-vocabulary scene graph: ConceptGraphs-style object-centric 3D memory.
# Each node is a discrete object with dual embeddings (SigLIP + DINOv3),
# 3D point cloud, bounding box, labels, and temporal tracking.

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class ObjectObservation:
    """A single observation of an object from one frame."""

    mask: np.ndarray  # (H, W) bool
    bbox_xyxy: np.ndarray  # (4,) x0, y0, x1, y1
    rgb_crop: np.ndarray  # (H_crop, W_crop, 3) uint8
    points_3d: Tensor  # (N, 3) world-frame xyz
    points_rgb: Optional[Tensor]  # (N, 3) colors
    camera_pose: np.ndarray  # (4, 4)
    label: str  # detected label / concept
    score: float
    timestep: int
    siglip_embedding: Optional[Tensor] = None  # (D_siglip,)
    dinov3_embedding: Optional[Tensor] = None  # (D_dino,)


@dataclass
class SceneGraphNode:
    """A persistent object in the scene graph."""

    node_id: int
    labels: List[str] = field(default_factory=list)
    label_counts: Dict[str, int] = field(default_factory=dict)

    # 3D geometry
    point_cloud: Optional[Tensor] = None  # (N, 3) world xyz, accumulated
    point_cloud_rgb: Optional[Tensor] = None  # (N, 3) colors
    bounds: Optional[Tensor] = None  # (3, 2) axis-aligned bbox [min, max]
    center: Optional[np.ndarray] = None  # (3,) mean position

    # Embeddings (running averages)
    siglip_embedding: Optional[Tensor] = None  # (D,) L2-normalized
    dinov3_embedding: Optional[Tensor] = None  # (D,) L2-normalized
    _siglip_sum: Optional[Tensor] = None
    _dinov3_sum: Optional[Tensor] = None

    # Temporal tracking
    first_seen: int = 0
    last_seen: int = 0
    observation_count: int = 0

    # Best crop for visualization
    best_crop: Optional[np.ndarray] = None
    best_score: float = 0.0

    @property
    def primary_label(self) -> str:
        if not self.label_counts:
            return "unknown"
        return max(self.label_counts, key=self.label_counts.get)

    @property
    def is_stable(self) -> bool:
        """An object is stable if observed from multiple viewpoints."""
        return self.observation_count >= 2

    def merge_observation(self, obs: ObjectObservation) -> None:
        """Incorporate a new observation into this node."""
        # Labels
        lbl = obs.label
        if lbl not in self.labels:
            self.labels.append(lbl)
        self.label_counts[lbl] = self.label_counts.get(lbl, 0) + 1

        # Point cloud
        if obs.points_3d is not None and obs.points_3d.shape[0] > 0:
            if self.point_cloud is None:
                self.point_cloud = obs.points_3d
                self.point_cloud_rgb = obs.points_rgb
            else:
                self.point_cloud = torch.cat([self.point_cloud, obs.points_3d], dim=0)
                if self.point_cloud_rgb is not None and obs.points_rgb is not None:
                    self.point_cloud_rgb = torch.cat(
                        [self.point_cloud_rgb, obs.points_rgb], dim=0
                    )

        # Update bounds and center
        if self.point_cloud is not None and self.point_cloud.shape[0] > 0:
            self.bounds = torch.stack(
                [self.point_cloud.min(dim=0).values, self.point_cloud.max(dim=0).values],
                dim=1,
            )
            self.center = self.point_cloud.mean(dim=0).cpu().numpy()

        # Embeddings (running mean)
        if obs.siglip_embedding is not None:
            if self._siglip_sum is None:
                self._siglip_sum = obs.siglip_embedding.clone()
            else:
                self._siglip_sum = self._siglip_sum + obs.siglip_embedding
            self.siglip_embedding = F.normalize(self._siglip_sum.unsqueeze(0), dim=-1).squeeze(0)

        if obs.dinov3_embedding is not None:
            if self._dinov3_sum is None:
                self._dinov3_sum = obs.dinov3_embedding.clone()
            else:
                self._dinov3_sum = self._dinov3_sum + obs.dinov3_embedding
            self.dinov3_embedding = F.normalize(self._dinov3_sum.unsqueeze(0), dim=-1).squeeze(0)

        # Temporal
        self.last_seen = obs.timestep
        if self.observation_count == 0:
            self.first_seen = obs.timestep
        self.observation_count += 1

        # Best crop
        if obs.score >= self.best_score:
            self.best_score = obs.score
            self.best_crop = obs.rgb_crop

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node for JSON (without large tensors)."""
        return {
            "node_id": self.node_id,
            "labels": self.labels,
            "label_counts": self.label_counts,
            "primary_label": self.primary_label,
            "center": self.center.tolist() if self.center is not None else None,
            "bounds_min": self.bounds[:, 0].tolist() if self.bounds is not None else None,
            "bounds_max": self.bounds[:, 1].tolist() if self.bounds is not None else None,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "observation_count": self.observation_count,
            "is_stable": self.is_stable,
        }


@dataclass
class SceneGraphEdge:
    """A spatial relationship between two nodes."""

    source_id: int
    target_id: int
    relation: str  # "near", "on", "on_floor", "above", "below"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "relation": self.relation,
        }


class OpenVocabSceneGraph:
    """Open-vocabulary 3D scene graph inspired by ConceptGraphs.

    Maintains a set of discrete object nodes, each with:
    - Dual embeddings: SigLIP (text-aligned) + DINOv3 (visual similarity)
    - 3D point cloud and axis-aligned bounding box
    - Temporal tracking (first/last seen, observation count)
    - Open-vocabulary labels from the segmenter

    Spatial edges (near, on, on_floor) are computed between nodes.
    Deduplication uses both 3D bbox IoU and DINOv3 visual similarity.
    """

    def __init__(
        self,
        dedup_visual_threshold: float = 0.85,
        dedup_iou_threshold: float = 0.3,
        max_near_distance: float = 1.5,
        min_on_height: float = 0.02,
        max_on_height: float = 0.3,
        floor_z_threshold: float = 0.05,
        staleness_horizon: int = 0,
        min_observations_stable: int = 2,
        min_points_per_object: int = 10,
        max_points_per_object: int = 5000,
    ):
        self.dedup_visual_threshold = dedup_visual_threshold
        self.dedup_iou_threshold = dedup_iou_threshold
        self.max_near_distance = max_near_distance
        self.min_on_height = min_on_height
        self.max_on_height = max_on_height
        self.floor_z_threshold = floor_z_threshold
        self.staleness_horizon = staleness_horizon
        self.min_observations_stable = min_observations_stable
        self.min_points_per_object = min_points_per_object
        self.max_points_per_object = max_points_per_object

        self.nodes: Dict[int, SceneGraphNode] = {}
        self.edges: List[SceneGraphEdge] = []
        self._next_id = 0
        self._current_step = 0

    @property
    def num_objects(self) -> int:
        return len(self.nodes)

    @property
    def stable_objects(self) -> List[SceneGraphNode]:
        return [n for n in self.nodes.values() if n.is_stable]

    def _allocate_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def add_observation(self, obs: ObjectObservation) -> int:
        """Add an object observation, either merging into an existing node or creating new.

        Returns the node_id it was assigned to.
        """
        self._current_step = max(self._current_step, obs.timestep)

        # Try to match to existing node
        best_match_id = self._find_best_match(obs)

        if best_match_id is not None:
            self.nodes[best_match_id].merge_observation(obs)
            self._downsample_node(best_match_id)
            return best_match_id
        else:
            nid = self._allocate_id()
            node = SceneGraphNode(node_id=nid)
            node.merge_observation(obs)
            self.nodes[nid] = node
            return nid

    def add_observations_batch(self, observations: List[ObjectObservation]) -> List[int]:
        """Add multiple observations from a single frame. Returns list of node IDs."""
        return [self.add_observation(obs) for obs in observations]

    def _find_best_match(self, obs: ObjectObservation) -> Optional[int]:
        """Find the best matching existing node for an observation.

        Uses a combination of:
        1. 3D bounding box IoU (geometric overlap)
        2. DINOv3 visual similarity (appearance)
        3. SigLIP similarity (semantic)
        """
        if not self.nodes:
            return None

        best_id = None
        best_score = 0.0

        obs_bounds = None
        if obs.points_3d is not None and obs.points_3d.shape[0] >= 3:
            obs_min = obs.points_3d.min(dim=0).values
            obs_max = obs.points_3d.max(dim=0).values
            obs_bounds = torch.stack([obs_min, obs_max], dim=1)  # (3, 2)

        for nid, node in self.nodes.items():
            score = 0.0

            # 3D IoU
            if obs_bounds is not None and node.bounds is not None:
                iou = _bbox3d_iou(obs_bounds, node.bounds)
                if iou > self.dedup_iou_threshold:
                    score += iou * 0.4

            # DINOv3 visual similarity
            if obs.dinov3_embedding is not None and node.dinov3_embedding is not None:
                vis_sim = F.cosine_similarity(
                    obs.dinov3_embedding.unsqueeze(0),
                    node.dinov3_embedding.unsqueeze(0),
                ).item()
                if vis_sim > self.dedup_visual_threshold:
                    score += vis_sim * 0.35

            # SigLIP semantic similarity
            if obs.siglip_embedding is not None and node.siglip_embedding is not None:
                sem_sim = F.cosine_similarity(
                    obs.siglip_embedding.unsqueeze(0),
                    node.siglip_embedding.unsqueeze(0),
                ).item()
                score += sem_sim * 0.25

            if score > best_score:
                best_score = score
                best_id = nid

        # Require minimum combined score to match
        if best_score < 0.3:
            return None
        return best_id

    def _downsample_node(self, node_id: int) -> None:
        """Keep point cloud size bounded by random subsampling."""
        node = self.nodes[node_id]
        if node.point_cloud is not None and node.point_cloud.shape[0] > self.max_points_per_object:
            indices = torch.randperm(node.point_cloud.shape[0])[: self.max_points_per_object]
            node.point_cloud = node.point_cloud[indices]
            if node.point_cloud_rgb is not None:
                node.point_cloud_rgb = node.point_cloud_rgb[indices]

    def update_edges(self) -> None:
        """Recompute all spatial edges between nodes."""
        self.edges = []
        node_ids = list(self.nodes.keys())

        for i, nid_a in enumerate(node_ids):
            node_a = self.nodes[nid_a]
            if node_a.center is None:
                continue

            # on_floor
            if node_a.center[2] < self.floor_z_threshold:
                self.edges.append(SceneGraphEdge(nid_a, -1, "on_floor"))

            for nid_b in node_ids[i + 1 :]:
                node_b = self.nodes[nid_b]
                if node_b.center is None:
                    continue

                dist_2d = float(np.linalg.norm(node_a.center[:2] - node_b.center[:2]))
                z_diff = node_a.center[2] - node_b.center[2]

                # near
                if dist_2d < self.max_near_distance:
                    self.edges.append(SceneGraphEdge(nid_a, nid_b, "near"))

                # on: a is on b (a is above b, close in xy)
                if (
                    dist_2d < 0.5
                    and self.min_on_height < z_diff < self.max_on_height
                ):
                    self.edges.append(SceneGraphEdge(nid_a, nid_b, "on"))
                elif (
                    dist_2d < 0.5
                    and self.min_on_height < -z_diff < self.max_on_height
                ):
                    self.edges.append(SceneGraphEdge(nid_b, nid_a, "on"))

    def prune_stale(self, current_step: Optional[int] = None) -> List[int]:
        """Remove objects not seen recently (if staleness_horizon > 0).

        Returns list of removed node IDs.
        """
        if self.staleness_horizon <= 0:
            return []
        step = current_step or self._current_step
        removed = []
        for nid in list(self.nodes.keys()):
            node = self.nodes[nid]
            if step - node.last_seen > self.staleness_horizon:
                del self.nodes[nid]
                removed.append(nid)
        if removed:
            self.edges = [
                e for e in self.edges if e.source_id not in removed and e.target_id not in removed
            ]
        return removed

    def prune_small(self) -> List[int]:
        """Remove objects with too few points."""
        removed = []
        for nid in list(self.nodes.keys()):
            node = self.nodes[nid]
            n_pts = node.point_cloud.shape[0] if node.point_cloud is not None else 0
            if n_pts < self.min_points_per_object:
                del self.nodes[nid]
                removed.append(nid)
        return removed

    def merge_duplicates(self) -> int:
        """Post-hoc deduplication: merge nodes that are very similar.

        Returns number of merges performed.
        """
        merges = 0
        node_ids = list(self.nodes.keys())
        absorbed = set()

        for i, nid_a in enumerate(node_ids):
            if nid_a in absorbed:
                continue
            node_a = self.nodes[nid_a]

            for nid_b in node_ids[i + 1 :]:
                if nid_b in absorbed:
                    continue
                node_b = self.nodes[nid_b]

                should_merge = False

                # Check IoU
                if node_a.bounds is not None and node_b.bounds is not None:
                    iou = _bbox3d_iou(node_a.bounds, node_b.bounds)
                    if iou > 0.5:
                        should_merge = True

                # Check DINOv3 similarity
                if (
                    not should_merge
                    and node_a.dinov3_embedding is not None
                    and node_b.dinov3_embedding is not None
                ):
                    sim = F.cosine_similarity(
                        node_a.dinov3_embedding.unsqueeze(0),
                        node_b.dinov3_embedding.unsqueeze(0),
                    ).item()
                    if sim > self.dedup_visual_threshold and node_a.bounds is not None and node_b.bounds is not None:
                        iou = _bbox3d_iou(node_a.bounds, node_b.bounds)
                        if iou > 0.1:
                            should_merge = True

                if should_merge:
                    # Absorb b into a
                    self._merge_nodes(nid_a, nid_b)
                    absorbed.add(nid_b)
                    merges += 1

        for nid in absorbed:
            self.nodes.pop(nid, None)

        return merges

    def _merge_nodes(self, keep_id: int, remove_id: int) -> None:
        """Merge remove_id node into keep_id node."""
        keeper = self.nodes[keep_id]
        donor = self.nodes[remove_id]

        for lbl, cnt in donor.label_counts.items():
            if lbl not in keeper.labels:
                keeper.labels.append(lbl)
            keeper.label_counts[lbl] = keeper.label_counts.get(lbl, 0) + cnt

        if donor.point_cloud is not None:
            if keeper.point_cloud is not None:
                keeper.point_cloud = torch.cat([keeper.point_cloud, donor.point_cloud], dim=0)
                if keeper.point_cloud_rgb is not None and donor.point_cloud_rgb is not None:
                    keeper.point_cloud_rgb = torch.cat(
                        [keeper.point_cloud_rgb, donor.point_cloud_rgb], dim=0
                    )
            else:
                keeper.point_cloud = donor.point_cloud
                keeper.point_cloud_rgb = donor.point_cloud_rgb

        if keeper.point_cloud is not None and keeper.point_cloud.shape[0] > 0:
            keeper.bounds = torch.stack(
                [keeper.point_cloud.min(dim=0).values, keeper.point_cloud.max(dim=0).values],
                dim=1,
            )
            keeper.center = keeper.point_cloud.mean(dim=0).cpu().numpy()

        # Merge embeddings
        if donor._siglip_sum is not None:
            if keeper._siglip_sum is None:
                keeper._siglip_sum = donor._siglip_sum.clone()
            else:
                keeper._siglip_sum = keeper._siglip_sum + donor._siglip_sum
            keeper.siglip_embedding = F.normalize(
                keeper._siglip_sum.unsqueeze(0), dim=-1
            ).squeeze(0)

        if donor._dinov3_sum is not None:
            if keeper._dinov3_sum is None:
                keeper._dinov3_sum = donor._dinov3_sum.clone()
            else:
                keeper._dinov3_sum = keeper._dinov3_sum + donor._dinov3_sum
            keeper.dinov3_embedding = F.normalize(
                keeper._dinov3_sum.unsqueeze(0), dim=-1
            ).squeeze(0)

        keeper.first_seen = min(keeper.first_seen, donor.first_seen)
        keeper.last_seen = max(keeper.last_seen, donor.last_seen)
        keeper.observation_count += donor.observation_count

        if donor.best_score > keeper.best_score:
            keeper.best_score = donor.best_score
            keeper.best_crop = donor.best_crop

        self._downsample_node(keep_id)

    # --- Query methods ---

    def localize_text(self, text: str, text_encoder: Any) -> Optional[np.ndarray]:
        """Find the 3D position of an object matching the text query.

        Args:
            text: natural language object description
            text_encoder: encoder with encode_text(str) -> Tensor

        Returns:
            (3,) world position or None
        """
        if not self.nodes:
            return None

        text_feat = text_encoder.encode_text(text)
        if text_feat.dim() == 2:
            text_feat = text_feat.squeeze(0)
        text_feat = F.normalize(text_feat.unsqueeze(0), dim=-1)

        best_score = -1.0
        best_center = None

        for node in self.nodes.values():
            if node.siglip_embedding is None or node.center is None:
                continue
            sim = F.cosine_similarity(
                text_feat, node.siglip_embedding.unsqueeze(0)
            ).item()
            if sim > best_score:
                best_score = sim
                best_center = node.center

        return best_center

    def check_for_object(self, text: str, text_encoder: Any) -> Tuple[float, Optional[np.ndarray]]:
        """Check if an object matching text is in the graph.

        Returns (confidence, location_xyz).
        """
        if not self.nodes:
            return 0.0, None

        text_feat = text_encoder.encode_text(text)
        if text_feat.dim() == 2:
            text_feat = text_feat.squeeze(0)
        text_feat = F.normalize(text_feat.unsqueeze(0), dim=-1)

        best_score = 0.0
        best_center = None

        for node in self.nodes.values():
            if node.siglip_embedding is None:
                continue
            sim = F.cosine_similarity(
                text_feat, node.siglip_embedding.unsqueeze(0)
            ).item()
            # Also check label string match
            label_boost = 0.0
            text_lower = text.lower().strip()
            for lbl in node.labels:
                if text_lower in lbl.lower() or lbl.lower() in text_lower:
                    label_boost = 0.2
                    break
            total = min(sim + label_boost, 1.0)
            if total > best_score:
                best_score = total
                best_center = node.center

        return best_score, best_center

    def list_objects(self) -> List[str]:
        """Return primary labels of all stable objects."""
        return [n.primary_label for n in self.nodes.values() if n.is_stable]

    def get_node_by_label(self, label: str) -> Optional[SceneGraphNode]:
        """Find the node whose primary label best matches."""
        label_lower = label.lower().strip()
        for node in self.nodes.values():
            if node.primary_label.lower() == label_lower:
                return node
        for node in self.nodes.values():
            for lbl in node.labels:
                if label_lower in lbl.lower() or lbl.lower() in label_lower:
                    return node
        return None

    # --- Serialization ---

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the scene graph to a JSON-compatible dict (no large tensors)."""
        self.update_edges()
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
            "num_objects": self.num_objects,
            "num_stable": len(self.stable_objects),
        }

    def to_string(self) -> str:
        """Human-readable text representation for LLM prompts."""
        self.update_edges()
        lines = [f"Scene graph: {self.num_objects} objects"]
        for node in self.nodes.values():
            lines.append(
                f"  [{node.node_id}] {node.primary_label} "
                f"(seen {node.observation_count}x, "
                f"pos={node.center.tolist() if node.center is not None else '?'})"
            )
        for edge in self.edges:
            src = self.nodes.get(edge.source_id)
            tgt = self.nodes.get(edge.target_id)
            src_lbl = src.primary_label if src else "floor"
            tgt_lbl = tgt.primary_label if tgt else "floor"
            lines.append(f"  {src_lbl} --{edge.relation}--> {tgt_lbl}")
        return "\n".join(lines)

    def save(self, path: str) -> None:
        """Save scene graph to a directory."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)

        # Save graph structure
        with open(out / "scene_graph.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        # Save embeddings and point clouds as tensors
        tensors = {}
        for nid, node in self.nodes.items():
            prefix = f"node_{nid}"
            if node.point_cloud is not None:
                tensors[f"{prefix}_points"] = node.point_cloud
            if node.point_cloud_rgb is not None:
                tensors[f"{prefix}_rgb"] = node.point_cloud_rgb
            if node.siglip_embedding is not None:
                tensors[f"{prefix}_siglip"] = node.siglip_embedding
            if node.dinov3_embedding is not None:
                tensors[f"{prefix}_dinov3"] = node.dinov3_embedding
        if tensors:
            torch.save(tensors, out / "node_tensors.pt")

        # Save best crops
        import cv2

        for nid, node in self.nodes.items():
            if node.best_crop is not None:
                cv2.imwrite(
                    str(out / f"node_{nid}_crop.jpg"),
                    node.best_crop[:, :, ::-1],
                )

    @classmethod
    def load(cls, path: str, **kwargs) -> "OpenVocabSceneGraph":
        """Load a scene graph from a directory."""
        p = Path(path)
        graph = cls(**kwargs)

        with open(p / "scene_graph.json") as f:
            data = json.load(f)

        tensors = {}
        tensor_path = p / "node_tensors.pt"
        if tensor_path.exists():
            tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)

        for node_data in data.get("nodes", []):
            nid = node_data["node_id"]
            node = SceneGraphNode(
                node_id=nid,
                labels=node_data.get("labels", []),
                label_counts=node_data.get("label_counts", {}),
                first_seen=node_data.get("first_seen", 0),
                last_seen=node_data.get("last_seen", 0),
                observation_count=node_data.get("observation_count", 0),
            )
            if node_data.get("center") is not None:
                node.center = np.array(node_data["center"])

            prefix = f"node_{nid}"
            if f"{prefix}_points" in tensors:
                node.point_cloud = tensors[f"{prefix}_points"]
                if f"{prefix}_rgb" in tensors:
                    node.point_cloud_rgb = tensors[f"{prefix}_rgb"]
                node.bounds = torch.stack(
                    [node.point_cloud.min(dim=0).values, node.point_cloud.max(dim=0).values],
                    dim=1,
                )
            if f"{prefix}_siglip" in tensors:
                node.siglip_embedding = tensors[f"{prefix}_siglip"]
                node._siglip_sum = node.siglip_embedding * node.observation_count
            if f"{prefix}_dinov3" in tensors:
                node.dinov3_embedding = tensors[f"{prefix}_dinov3"]
                node._dinov3_sum = node.dinov3_embedding * node.observation_count

            import cv2

            crop_path = p / f"node_{nid}_crop.jpg"
            if crop_path.exists():
                node.best_crop = cv2.imread(str(crop_path))[:, :, ::-1].copy()

            graph.nodes[nid] = node
            graph._next_id = max(graph._next_id, nid + 1)

        return graph


def _bbox3d_iou(bounds_a: Tensor, bounds_b: Tensor) -> float:
    """IoU between two (3, 2) axis-aligned bounding boxes."""
    mins_a, maxs_a = bounds_a[:, 0], bounds_a[:, 1]
    mins_b, maxs_b = bounds_b[:, 0], bounds_b[:, 1]
    inter_mins = torch.max(mins_a, mins_b)
    inter_maxs = torch.min(maxs_a, maxs_b)
    inter_extent = (inter_maxs - inter_mins).clamp(min=0)
    inter_vol = inter_extent.prod().item()
    vol_a = (maxs_a - mins_a).clamp(min=0).prod().item()
    vol_b = (maxs_b - mins_b).clamp(min=0).prod().item()
    union = vol_a + vol_b - inter_vol
    return inter_vol / union if union > 0 else 0.0
