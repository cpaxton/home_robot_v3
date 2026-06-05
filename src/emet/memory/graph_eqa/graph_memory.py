# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Graph-based EQA memory: re-implementation inspired by GraphEQA
# (https://arxiv.org/abs/2412.14480). Object-centric scene graph + task-relevant
# images for embodied question answering. No code copied from closed-source repos.

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from PIL import Image

from emet.core.parameters import Parameters
from emet.memory.graph_eqa.human_answer import format_human_eqa_answer


def labels_are_semantic_graph_hypothesis(labels: list[str] | None) -> bool:
    """
    Whether ``labels`` should become a scene-graph node (vs navigation-only sample).

    Generic VLM fallback ``["object"]`` is not a semantic hypothesis: it would clutter
    the graph with one node per controller step.
    """
    if not labels:
        return False
    if len(labels) == 1 and labels[0].strip().lower() == "object":
        return False
    return True


@dataclass
class GraphNavigationSample:
    """A viewpoint along the run without an object-level graph node (RGB + anchors)."""

    rgb: np.ndarray
    xyz: np.ndarray  # (3,) scene anchor (e.g. depth median in world frame)
    base_xyz: np.ndarray | None = None  # (3,) optional robot base x,y,z for trajectory context


GT_BODY_DESC_PREFIX = "ground_truth:"


@dataclass
class GraphNode:
    """Single node in the scene graph: an object or region with label and position."""

    node_id: int
    labels: list[str]
    xyz: np.ndarray  # (3,) world position
    obs_id: int  # 1-based index into observations list
    description: str | None = None  # optional VLM-generated description
    last_seen: int = 0
    support_count: int = 1
    extent_half: np.ndarray | None = None  # (3,) half-axis sizes in meters; None = point-like
    bbox_xyxy: tuple[int, int, int, int] | None = None  # pixel crop in obs RGB; None = full frame
    is_viewpoint: bool = False  # True = robot/camera vantage (``seen_from`` target), not a detected object
    embedding: np.ndarray | None = None  # optional visual embedding (e.g. SigLIP crop)
    bounds_3d: dict[str, list[float]] | None = None  # axis-aligned world bounds {min,max,center,size}


def is_ground_truth_node(node: GraphNode | None) -> bool:
    """True when ``node.description`` marks sim GT (stable ``body_name`` key)."""
    if node is None:
        return False
    desc = getattr(node, "description", None)
    return isinstance(desc, str) and desc.startswith(GT_BODY_DESC_PREFIX)


@dataclass
class GraphObservation:
    """One observation (image + pose + labels) used to build the graph."""

    obs_id: int  # 1-based
    rgb: np.ndarray  # (H, W, 3)
    xyz: np.ndarray  # (3,) e.g. mean of visible points or camera position
    labels: list[str]
    description: str | None = None  # optional VLM-generated description
    viewer_xyz: np.ndarray | None = None  # (3,) robot base or camera when the image was taken


def _near(p1: np.ndarray, p2: np.ndarray, max_dist: float = 1.5) -> bool:
    return float(np.linalg.norm(p1[:2] - p2[:2])) <= max_dist


def _on(p_lower: np.ndarray, p_upper: np.ndarray, z_thresh: float = 0.15) -> bool:
    """Heuristic: lower object is 'on' upper if roughly below and close in xy."""
    if p_lower[2] >= p_upper[2]:
        return False
    return abs(p_lower[2] - p_upper[2]) <= z_thresh + 0.2 and float(np.linalg.norm(p_lower[:2] - p_upper[:2])) < 0.5


def _on_floor(p: np.ndarray, floor_z: float = 0.05) -> bool:
    return float(p[2]) <= floor_z


class GraphEQAMemory:
    """
    Graph-based semantic memory for Embodied Question Answering (EQA).

    Maintains an object-centric scene graph (nodes = objects/regions with labels and
    3D positions; edges = spatial relations). Uses the same EQA query contract as
    the DynaMem voxel map: query_answer(question, xyt, planner) returns
    (reasoning, answer, confidence, confidence_reasoning, target_point, relevant_images).

    Optional **Dynagraph** behavior (parameters ``dynagraph_merge_xy_m``,
    ``dynagraph_staleness_horizon``): spatial merge of nodes with the same primary
    label within XY distance, and ``maintain(current_step)`` to drop stale nodes.
    """

    def __init__(
        self,
        parameters: Parameters | None = None,
        max_near_distance: float = 1.5,
        eqa_client: Callable[..., str] | None = None,
        image_description_client: Callable[..., str] | None = None,
        log_dir: str = "graph_eqa_log",
        defer_llm_clients: bool = False,
    ):
        self.parameters = parameters or {}
        self.max_near_distance = max_near_distance
        self._nodes: list[GraphNode] = []
        self._edges: list[tuple[int, int, str]] = []  # (id1, id2, relation)
        self._observations: list[GraphObservation] = []
        self._next_obs_id = 1
        self._question: str | None = None
        self._relevant_objects: list[str] | None = None
        self._history_outputs: list[str] = []

        self.log_dir = log_dir
        self.eqa_client = eqa_client
        self.image_description_client = image_description_client
        self._defer_llm_clients = defer_llm_clients
        self._nav_samples: list[GraphNavigationSample] = []
        self._viewpoint_by_obs_id: dict[int, int] = {}
        self._record_navigation = True
        self._nav_max = 256
        self._graph_timestep: int = 0
        self._fallback_timestep: int = 0
        self.spatial_merge_m: float = 0.0
        self.staleness_horizon: int = 0
        self._load_navigation_settings()
        self._load_dynagraph_settings()

        if not defer_llm_clients and (self.eqa_client is None or self.image_description_client is None):
            self._init_clients()

    def _parameters_dict(self) -> dict[str, Any]:
        p = self.parameters
        if isinstance(p, dict):
            return p
        if hasattr(p, "data") and isinstance(p.data, dict):
            return p.data
        return {}

    def _load_navigation_settings(self) -> None:
        d = self._parameters_dict()
        if not d:
            return
        v = d.get("graph_eqa_record_navigation")
        if v is not None:
            self._record_navigation = bool(v)
        blk = d.get("graph_eqa_extract")
        if isinstance(blk, dict) and blk.get("navigation_samples_max") is not None:
            self._nav_max = max(1, int(blk["navigation_samples_max"]))

    def _load_dynagraph_settings(self) -> None:
        d = self._parameters_dict()
        if not d:
            return
        if d.get("dynagraph_merge_xy_m") is not None:
            self.spatial_merge_m = float(d["dynagraph_merge_xy_m"])
        if d.get("dynagraph_staleness_horizon") is not None:
            self.staleness_horizon = max(0, int(d["dynagraph_staleness_horizon"]))

    def set_graph_timestep(self, step: int) -> None:
        """Set the discrete time index used for ``last_seen`` and staleness (e.g. controller ``obs_count``)."""
        self._graph_timestep = int(step)

    def set_navigation_samples_max(self, n: int) -> None:
        """Raise or lower the cap on stored navigation viewpoint samples (default from config)."""
        self._nav_max = max(1, int(n))

    @property
    def navigation_samples_max(self) -> int:
        return int(self._nav_max)

    def _effective_timestep(self) -> int:
        if self._graph_timestep > 0:
            return self._graph_timestep
        self._fallback_timestep += 1
        return self._fallback_timestep

    def maintain(self, current_step: int) -> int:
        """
        Drop stale nodes (and their observations) when ``staleness_horizon`` > 0,
        then renumber ``node_id`` to 1..N and rebuild edges.

        Returns:
            Number of nodes removed.
        """
        if self.staleness_horizon <= 0 or not self._nodes:
            return 0
        cur = int(current_step)
        to_drop: list[GraphNode] = [
            n for n in self._nodes if not is_ground_truth_node(n) and cur - int(n.last_seen) > self.staleness_horizon
        ]
        if not to_drop:
            return 0
        drop_obs = {n.obs_id for n in to_drop if not n.is_viewpoint}
        drop_node_ids = {n.node_id for n in to_drop}
        drop_node_ids |= {
            n.node_id for n in self._nodes if n.is_viewpoint and int(n.obs_id) in drop_obs
        }
        self._nodes = [n for n in self._nodes if n.node_id not in drop_node_ids]
        self._observations = [o for o in self._observations if o.obs_id not in drop_obs]
        for i, n in enumerate(self._nodes, start=1):
            self._nodes[i - 1] = replace(n, node_id=i)
        self._rebuild_viewpoint_index()
        self._update_edges()
        return len(to_drop)

    def _ensure_llm_clients(self) -> None:
        """Load shared Qwen3.5 multimodal on first use when defer_llm_clients=True."""
        if self.eqa_client is not None and self.image_description_client is not None:
            return
        self._init_clients()

    def _init_clients(self) -> None:
        """Initialize EQA + keyword helper on one shared Qwen3.5 multimodal load."""
        try:
            from emet.llms.eqa_qwen import build_shared_eqa_clients
            from emet.llms.eqa_vl_settings import apply_eqa_vl_runtime_settings, get_eqa_vl_int

            apply_eqa_vl_runtime_settings(self.parameters)
            kw = get_eqa_vl_int(self.parameters, "graph_keyword_max_tokens", 64)
            self.image_description_client, self.eqa_client = build_shared_eqa_clients(
                parameters=self.parameters,
                keyword_max_tokens=kw,
            )
        except ImportError as e:
            raise ImportError(
                "GraphEQA memory requires emet.llms (Qwen3.5 multimodal) for EQA. Install extras with GPU support."
            ) from e

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

        bbox_i: tuple[int, int, int, int] | None = None
        if bbox_xyxy is not None:
            b = tuple(int(x) for x in bbox_xyxy)
            if len(b) == 4:
                bbox_i = (b[0], b[1], b[2], b[3])

        if self.spatial_merge_m > 0:
            for idx, existing in enumerate(self._nodes):
                if existing.is_viewpoint or is_ground_truth_node(existing):
                    continue
                el = [(x or "").strip().lower() for x in existing.labels if str(x).strip()]
                if not el or el[0] != primary:
                    continue
                ex = np.asarray(existing.xyz, dtype=float).reshape(-1)[:3]
                if float(np.linalg.norm(ex[:2] - xyz_a[:2])) <= self.spatial_merge_m:
                    sc = int(existing.support_count) + 1
                    new_xyz = (existing.xyz * (sc - 1) + xyz_a) / sc
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
                    )
                    for o in self._observations:
                        if o.obs_id == existing.obs_id:
                            o.xyz = new_xyz
                            o.labels = merged_labels
                            if new_desc and not o.description:
                                o.description = new_desc
                            if viewer_a is not None:
                                o.viewer_xyz = viewer_a
                            break
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
        xyz_a = np.asarray(getattr(candidate, "xyz"), dtype=float).reshape(-1)[:3]
        bbox_xyxy = getattr(candidate, "bbox_xyxy", None)
        bounds_3d = getattr(candidate, "bounds_3d", None)
        embedding = getattr(candidate, "embedding", None)
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
                sc = int(existing.support_count) + 1
                alpha = 0.35
                new_xyz = (existing.xyz * (sc - 1) + xyz_a) / sc
                merged_labels = sorted(
                    {*(str(x).strip() for x in existing.labels if str(x).strip()), label}
                )
                new_emb = embedding
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
                )
                for o in self._observations:
                    if o.obs_id == existing.obs_id:
                        o.xyz = new_xyz
                        o.labels = merged_labels
                        if viewer_a is not None:
                            o.viewer_xyz = viewer_a
                        break
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
            for o in self._observations:
                if o.obs_id == existing.obs_id:
                    o.rgb = rgb_a.copy()
                    o.description = new_desc
                    break
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

    def _rebuild_viewpoint_index(self) -> None:
        self._viewpoint_by_obs_id = {
            int(n.obs_id): int(n.node_id) for n in self._nodes if n.is_viewpoint
        }

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
        """Compute spatial relations (near, on, on_floor) and ``seen_from`` viewpoint links."""
        self._edges.clear()
        objects = [n for n in self._nodes if not n.is_viewpoint]
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
                elif _on(nb.xyz, na.xyz):
                    self._edges.append((nb.node_id, na.node_id, "on"))
        for node in objects:
            self._ensure_seen_from_edge(node.node_id, int(node.obs_id))

    def to_string(self) -> str:
        """Serialize the scene graph to a string for mLLM prompts."""
        lines = []

        def _prompt_labels(labels: list[str], max_len: int = 120) -> str:
            s = ", ".join(labels) if labels else "object"
            return s if len(s) <= max_len else s[: max_len - 3] + "..."

        for n in self._nodes:
            lbl = _prompt_labels(n.labels)
            sup = f" n={n.support_count}" if getattr(n, "support_count", 1) != 1 else ""
            kind = "View" if n.is_viewpoint else "Node"
            lines.append(
                f"{kind} {n.node_id}: {lbl} at ({n.xyz[0]:.2f}, {n.xyz[1]:.2f}, {n.xyz[2]:.2f}) [Image {n.obs_id}]{sup}"
            )
        for a, b, rel in self._edges:
            b_str = "floor" if b == -1 else str(b)
            lines.append(f"  {rel}({a}, {b_str})")
        return "SCENE_GRAPH:\n" + "\n".join(lines) if lines else "SCENE_GRAPH: (empty)"

    def to_tree_string(self, indent: str = "  ") -> str:
        """
        Format the 3D spatial scene graph as an indented tree (text).

        Root = Scene; Floor is a virtual node; objects on floor are children of Floor;
        objects on other objects are nested. "Near" relations are listed at the end.
        Includes object labels, (x,y,z), and optional descriptions.
        """
        edge_set = set(self._edges)
        node_by_id = {n.node_id: n for n in self._nodes}
        object_nodes = [n for n in self._nodes if not n.is_viewpoint]

        def on_floor(nid: int) -> bool:
            return (nid, -1, "on") in edge_set

        def has_on_parent(nid: int) -> int | None:
            """Return node_id that this node is 'on', or None if on floor or no 'on' edge."""
            for a, b, rel in edge_set:
                if rel == "on" and a == nid and b != -1:
                    return b
            return None

        def children_of(nid: int | None) -> list[GraphNode]:
            if nid is None:
                # Floor children: explicitly on floor, or no "on" relation (in-scene)
                out = [
                    node_by_id[n.node_id]
                    for n in object_nodes
                    if on_floor(n.node_id) or has_on_parent(n.node_id) is None
                ]
            else:
                out = [node_by_id[a] for a, b, rel in edge_set if rel == "on" and b == nid and a in node_by_id]
            return sorted(out, key=lambda n: n.node_id)

        near_pairs = [(a, b) for a, b, rel in self._edges if rel == "near" and a < b]

        lines: list[str] = []
        lines.append("Scene (3D spatial graph)")
        lines.append(f"{indent}Floor")

        def visit(node: GraphNode, depth: int) -> None:
            pref = indent * (depth + 1)
            x, y, z = float(node.xyz[0]), float(node.xyz[1]), float(node.xyz[2])
            lbl = ", ".join(node.labels) if node.labels else "object"
            line = f"{pref}[{node.node_id}] {lbl}  at ({x:.2f}, {y:.2f}, {z:.2f})"
            if node.description:
                d = node.description
                if len(d) > 160:
                    d = d[:157] + "..."
                line += f"  — {d}"
            lines.append(line)
            for c in children_of(node.node_id):
                visit(c, depth + 1)

        for node in children_of(None):
            visit(node, 1)

        if near_pairs:
            lines.append("")
            lines.append("Near relations:")
            for a, b in near_pairs:
                na, nb = node_by_id.get(a), node_by_id.get(b)
                la = ", ".join(na.labels) if na and na.labels else str(a)
                lb = ", ".join(nb.labels) if nb and nb.labels else str(b)
                lines.append(f"{indent}{la} — {lb}")

        seen_from_edges = [(a, b) for a, b, rel in self._edges if rel == "seen_from"]
        if seen_from_edges:
            lines.append("")
            lines.append("Seen from (viewpoint node → object):")
            for a, b in seen_from_edges:
                na = node_by_id.get(a)
                nb = node_by_id.get(b)
                la = ", ".join(na.labels) if na and na.labels else str(a)
                if nb is not None:
                    vx, vy, vz = (float(nb.xyz[i]) for i in range(3))
                    lb = ", ".join(nb.labels) if nb.labels else str(b)
                    lines.append(f"{indent}{la} ← {lb} [{b}] at ({vx:.2f}, {vy:.2f}, {vz:.2f})")
                else:
                    lines.append(f"{indent}{la} ← node {b}")

        return "\n".join(lines) if lines else "Scene (3D spatial graph): (empty)"

    def extract_relevant_objects(self, question: str) -> None:
        """Extract keywords from the question for image selection (same idea as DynaMem)."""
        if self._question == question:
            return
        self._question = question
        prompt = (
            "Assume there is an agent doing Question Answering in an environment. "
            "When it receives a question, tell the agent few objects (preferably 1-3) to pay attention to. "
            "Example: Where is the pen? -> pen. Is there grey cloth on cloth hanger? -> grey cloth, cloth hanger"
        )
        out = self.image_description_client([prompt, question])
        self._relevant_objects = [s.strip() for s in out.split(",") if s.strip()]

    def _select_relevant_obs_ids(self, max_images: int = 6) -> list[int]:
        """Select observation IDs whose labels match relevant_objects (1-based)."""
        if not self._relevant_objects or not self._observations:
            return [o.obs_id for o in self._observations[:max_images]]
        seen: set = set()
        out: list[int] = []
        for obj in self._relevant_objects:
            obj_lower = obj.lower()
            for o in self._observations:
                if o.obs_id in seen:
                    continue
                if any(obj_lower in lab.lower() for lab in o.labels):
                    seen.add(o.obs_id)
                    out.append(o.obs_id)
                    if len(out) >= max_images:
                        return out
        # If few matches, add remaining up to max_images
        for o in self._observations:
            if o.obs_id not in seen:
                seen.add(o.obs_id)
                out.append(o.obs_id)
                if len(out) >= max_images:
                    break
        return out

    def _get_image_descriptions_str(self, obs_ids: list[int]) -> str:
        """Build IMAGE_DESCRIPTIONS string for the prompt (1-indexed image ids)."""
        options = []
        for i, obs in enumerate(self._observations, start=1):
            lbl = ", ".join(obs.labels) if obs.labels else "object"
            line = f"{i}. {lbl} at ({obs.xyz[0]:.2f}, {obs.xyz[1]:.2f});"
            if obs.obs_id in obs_ids:
                idx = obs_ids.index(obs.obs_id) + 1
                line += f" This observation is associated with Image {idx};"
            options.append(line)
        return "IMAGE_DESCRIPTIONS: " + "\n".join(options) if options else "IMAGE_DESCRIPTIONS: (none)"

    def parse_answer(self, answer_outputs: str) -> tuple[str, str, bool, str, str]:
        """Parse mLLM output into reasoning, answer, confidence, action, confidence_reasoning."""

        def extract_between(text: str, start: str, end: str) -> str:
            try:
                return text.split(start, 1)[1].split(end, 1)[0].strip().replace("\n", "").replace("\t", "")
            except IndexError:
                return ""

        def extract_after(text: str, start: str) -> str:
            try:
                return text.split(start, 1)[1].strip().replace("\n", "").replace("\t", "")
            except IndexError:
                return ""

        reasoning = extract_between(answer_outputs, "reasoning:", "answer:")
        answer = extract_between(answer_outputs, "answer:", "confidence:")
        confidence_text = extract_between(answer_outputs, "confidence:", "action:")
        confidence = "true" in confidence_text.replace(" ", "").lower()
        action = extract_between(answer_outputs, "action:", "confidence_reasoning:")
        confidence_reasoning = extract_after(answer_outputs, "confidence_reasoning:")
        return reasoning, answer, confidence, action, confidence_reasoning

    def _target_point_from_image_id(self, image_id: int) -> np.ndarray | None:
        """Return (x, y, 1) for the observation's position when mLLM suggests navigating to that image."""
        for obs in self._observations:
            if obs.obs_id == image_id:
                return np.array([obs.xyz[0], obs.xyz[1], 1.0], dtype=float)
        return None

    def query_answer(
        self,
        question: str,
        xyt: Any | np.ndarray | list | None = None,
        planner: Any = None,
    ) -> tuple[str, str, bool, str, np.ndarray | None, list[Image.Image]]:
        """
        Answer the question using the scene graph and task-relevant images.
        Same return contract as voxel_dynamem.SparseVoxelMap.query_answer.

        Returns:
            reasoning, answer, confidence, confidence_reasoning, target_point, relevant_images
        """
        self._ensure_llm_clients()
        self.extract_relevant_objects(question)
        obs_ids = self._select_relevant_obs_ids(max_images=6)
        graph_str = self.to_string()
        nav_fallback_tail: list[GraphNavigationSample] = []
        if self._observations:
            img_desc_str = self._get_image_descriptions_str(obs_ids)
        elif self._nav_samples:
            nav_fallback_tail = self._nav_samples[-6:]
            lines = [
                "IMAGE_DESCRIPTIONS (navigation-only views; no object graph nodes yet):",
            ]
            for i, nv in enumerate(nav_fallback_tail, start=1):
                tail = f" robot base (~{nv.base_xyz[0]:.2f}, {nv.base_xyz[1]:.2f})." if nv.base_xyz is not None else ""
                lines.append(
                    f"Image {i}. viewpoint anchor at ({nv.xyz[0]:.2f}, {nv.xyz[1]:.2f}, {nv.xyz[2]:.2f});{tail}"
                )
            img_desc_str = "\n".join(lines)
        else:
            img_desc_str = self._get_image_descriptions_str(obs_ids)

        commands: list[Any] = ["Question: " + question]
        commands.append("HISTORY: ")
        for i, h in enumerate(self._history_outputs):
            commands.append("Iteration_" + str(i) + ":" + h)
        commands.append(graph_str)
        commands.append(img_desc_str)

        relevant_images: list[Image.Image] = []
        for obs in self._observations:
            if obs.obs_id in obs_ids:
                relevant_images.append(Image.fromarray(obs.rgb.astype(np.uint8), mode="RGB"))
                commands.append(Image.fromarray(obs.rgb.astype(np.uint8), mode="RGB"))
        for nv in nav_fallback_tail:
            im = Image.fromarray(nv.rgb.astype(np.uint8), mode="RGB")
            relevant_images.append(im)
            commands.append(im)

        raw = self.eqa_client(commands)
        answer_outputs = raw.replace("*", "").replace("/", "").replace("#", "").lower()

        reasoning, answer, confidence, action, confidence_reasoning = self.parse_answer(answer_outputs)
        raw_answer = answer
        human = format_human_eqa_answer(
            question,
            answer,
            reasoning,
            self,
            confidence=confidence,
            confidence_reasoning=confidence_reasoning,
            selected_obs_ids=obs_ids,
        )
        answer = human.user_answer
        reasoning = human.debug_reasoning

        target_point = None
        if not confidence and action.strip():
            # Parse action as image id (integer)
            match = re.search(r"\d+", action.strip())
            if match:
                image_id = int(match.group())
                # image_id from mLLM is 1-based observation index
                if 1 <= image_id <= len(self._observations):
                    obs = self._observations[image_id - 1]
                    target_point = np.array([obs.xyz[0], obs.xyz[1], 1.0], dtype=float)
                elif nav_fallback_tail and 1 <= image_id <= len(nav_fallback_tail):
                    nv = nav_fallback_tail[image_id - 1]
                    target_point = np.array([nv.xyz[0], nv.xyz[1], 1.0], dtype=float)
                else:
                    target_point = self._target_point_from_image_id(image_id)
            self._history_outputs.append(
                "Answer:"
                + raw_answer
                + "\nReasoning:"
                + reasoning
                + "\nConfidence:"
                + str(confidence)
                + "\nAction: Navigate to Image "
                + action.strip()
                + "\nConfidence_reasoning:"
                + confidence_reasoning
            )
        else:
            self._history_outputs.append(
                "Answer:"
                + raw_answer
                + "\nReasoning:"
                + reasoning
                + "\nConfidence:"
                + str(confidence)
                + "\nAction:\nConfidence_reasoning: "
                + confidence_reasoning
            )

        return (
            reasoning,
            answer,
            confidence,
            confidence_reasoning,
            target_point,
            relevant_images,
        )

    def fill_descriptions_from_vlm(
        self,
        prompt: str | None = None,
        max_tokens: int = 80,
    ) -> None:
        """
        Fill missing node/observation descriptions using the VLM (e.g. Qwen 2.5-VL / 3.5).
        Skips observations that already have a description. Can be slow for many images.
        """
        if self.image_description_client is None:
            self._init_clients()
        default_prompt = (
            "In one short sentence, describe what is visible in this image: "
            "main objects, their arrangement, and any notable spatial relationships. "
            "Be concise."
        )
        prompt = prompt or default_prompt
        for obs in self._observations:
            if obs.description:
                continue
            try:
                # VLM accepts list of text + image(s)
                out = self.image_description_client(
                    [prompt, Image.fromarray(obs.rgb.astype(np.uint8), mode="RGB")],
                    verbose=False,
                )
                if isinstance(out, str) and out.strip():
                    desc = out.strip()
                    # Update observation (same object as stored)
                    obs.description = desc
                    # Update corresponding node
                    for n in self._nodes:
                        if n.obs_id == obs.obs_id:
                            n.description = desc
                            break
            except Exception:
                continue

    def get_observations(self) -> list[GraphObservation]:
        return list(self._observations)

    def get_nodes(self) -> list[GraphNode]:
        return list(self._nodes)

    def get_edges(self) -> list[tuple[int, int, str]]:
        return list(self._edges)

    def print_memory(self) -> str:
        """
        Return the 3D scene graph as a human-readable tree (same as to_tree_string).
        Use this as the canonical "print" output for the graph memory.
        """
        return self.to_tree_string()
