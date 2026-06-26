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
from emet.memory.graph_eqa.mcq_debias import (
    LETTERS,
    extract_single_letter,
    format_rotated_question,
    letter_to_original_index,
    match_freeform_to_choice,
    tally_choice_votes,
)

# Min SigLIP cosine similarity for an open-vocab text query to count as "present" in the
# observed point cloud. Matches DynaMem's verify_point default for SigLIP grounding.
SIGLIP_PRESENT_THRESHOLD = 0.21

# Min SigLIP cosine similarity for an open-vocab text query to count as "present" in the
# observed point cloud. Matches DynaMem's verify_point default for SigLIP grounding.
SIGLIP_PRESENT_THRESHOLD = 0.21

_QUESTION_STOPWORDS = frozenset(
    {
        "is",
        "the",
        "a",
        "an",
        "on",
        "in",
        "at",
        "to",
        "or",
        "and",
        "did",
        "i",
        "any",
        "there",
        "which",
        "where",
        "what",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "my",
        "me",
        "it",
        "its",
        "this",
        "that",
        "with",
        "for",
        "of",
        "by",
        "from",
        "left",
        "next",
        "under",
        "over",
        "one",
        "two",
        "not",
        "all",
        "can",
        "you",
        "your",
        "how",
        "when",
        "who",
        "why",
        "fold",
        "turned",
        "mounted",
        "standing",
        "covered",
        "color",
        "objects",
        "object",
        "see",
        "things",
        "thing",
        "room",
        "area",
        "items",
        "item",
    }
)


def _object_match_tokens(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(tok) >= 3 and tok not in _QUESTION_STOPWORDS
    }


def label_matches_relevant_object(obj: str, label: str) -> bool:
    """True when ``label`` plausibly names ``obj`` (handles ``standing fan`` vs ``fan``)."""
    obj_l = (obj or "").strip().lower()
    lab_l = (label or "").strip().lower()
    if not obj_l or not lab_l:
        return False
    if obj_l in lab_l or lab_l in obj_l:
        return True
    obj_tok = _object_match_tokens(obj_l)
    lab_tok = _object_match_tokens(lab_l)
    if not obj_tok or not lab_tok:
        return False
    if obj_tok <= lab_tok or lab_tok <= obj_tok:
        return True
    return bool(obj_tok & lab_tok)


def heuristic_relevant_objects(question: str, *, max_objects: int = 4) -> list[str]:
    """Cheap noun-like tokens from the question stem (before MCQ options)."""
    head = question.strip().split("?")[0]
    out: list[str] = []
    for tok in re.findall(r"[a-z]{3,}", head.lower()):
        if tok in _QUESTION_STOPWORDS:
            continue
        if tok not in out:
            out.append(tok)
        if len(out) >= max_objects:
            break
    return out


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
    is_frontier: bool = False  # True = unexplored map frontier cluster (managed by sync_frontier_nodes)
    embedding: np.ndarray | None = None  # optional visual embedding (e.g. SigLIP crop)
    bounds_3d: dict[str, list[float]] | None = None  # axis-aligned world bounds {min,max,center,size}
    nav_attempts: int = 0
    nav_failures: int = 0
    last_nav_note: str | None = None
    last_nav_at_step: int = 0


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
        self.last_eqa_raw: str = ""
        self.last_eqa_parsed: tuple[str, str, bool, str, str] = ("", "", False, "", "")
        self.last_eqa_obs_ids: list[int] = []
        self.last_eqa_action_obs_id: int | None = None
        self.last_nav_result_note: str = ""
        self.last_eqa_nav_fallback_count: int = 0
        # Model's own confidence before the graph-coverage gate suppresses it (for early-stop).
        self.last_eqa_model_confident: bool = False
        self._nodes: list[GraphNode] = []
        self._edges: list[tuple[int, int, str]] = []  # (id1, id2, relation)
        self._observations: list[GraphObservation] = []
        self._next_obs_id = 1
        self._question: str | None = None
        self._relevant_objects: list[str] | None = None
        # Dynagraph improvements (kept OFF here so GraphEQA stays a clean baseline; the
        # DynagraphController turns them on):
        #  * memory_summary_enabled: prepend the CONFIRMED_MEMORY block to the planner prompt.
        #  * _text_grounder: open-vocab visual grounder (text -> (similarity, xyz)) backed by
        #    the voxel map's SigLIP features, decoupling grounding from brittle caption labels
        #    (e.g. a "woven basket" captioned as "decorative plant").
        self.memory_summary_enabled: bool = False
        #  * mcq_debias_enabled: choice-rotation vote at episode end (see mcq_debias.py).
        self.mcq_debias_enabled: bool = False
        self.last_mcq_debias: dict[str, Any] = {}
        self._text_grounder: Callable[[str], tuple[float, np.ndarray] | None] | None = None
        self._obs_id_grounder: Callable[[str], int | None] | None = None
        self._enrich_object_hints: list[str] = []
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
        self.frontier_nodes_enabled: bool = True
        self._frontier_max_nodes: int = 12
        self._frontier_min_cluster_cells: int = 3
        self._frontier_keyword_score_weight: float = 1.0
        self._load_navigation_settings()
        self._load_dynagraph_settings()
        self._load_frontier_settings()

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

    def _load_frontier_settings(self) -> None:
        d = self._parameters_dict()
        blk = d.get("graph_eqa_frontier_nodes")
        if not isinstance(blk, dict):
            eqa = d.get("graph_eqa")
            if isinstance(eqa, dict):
                blk = eqa.get("frontier_nodes")
        if not isinstance(blk, dict):
            return
        if blk.get("enabled") is not None:
            self.frontier_nodes_enabled = bool(blk["enabled"])
        if blk.get("max_nodes") is not None:
            self._frontier_max_nodes = max(1, int(blk["max_nodes"]))
        if blk.get("min_cluster_cells") is not None:
            self._frontier_min_cluster_cells = max(1, int(blk["min_cluster_cells"]))
        if blk.get("keyword_score_weight") is not None:
            self._frontier_keyword_score_weight = max(0.0, float(blk["keyword_score_weight"]))

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
            n
            for n in self._nodes
            if not is_ground_truth_node(n)
            and not n.is_frontier
            and cur - int(n.last_seen) > self.staleness_horizon
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
        """Initialize EQA + keyword helper (one shared VLM: gemma4 / Qwen-VL / Qwen3.5)."""
        try:
            from emet.llms.eqa_vl_settings import get_eqa_vl_int
            from emet.llms.graph_eqa_vlm import build_graph_eqa_vlm_clients

            kw = get_eqa_vl_int(self.parameters, "graph_keyword_max_tokens", 64)
            self.image_description_client, self.eqa_client = build_graph_eqa_vlm_clients(
                parameters=self.parameters,
                keyword_max_tokens=kw,
            )
        except ImportError as e:
            raise ImportError(
                "GraphEQA memory requires emet.llms for EQA. Install GPU extras (torch, transformers)."
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
                if existing.is_viewpoint or existing.is_frontier or is_ground_truth_node(existing):
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
        xyz_a = np.asarray(candidate.xyz, dtype=float).reshape(-1)[:3]
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

    def _frontier_desc(self, cluster_id: str) -> str:
        from emet.memory.graph_eqa.frontier_nodes import FRONTIER_DESC_PREFIX

        return f"{FRONTIER_DESC_PREFIX}{cluster_id}"

    def _find_frontier_node(self, cluster_id: str) -> GraphNode | None:
        desc = self._frontier_desc(cluster_id)
        for n in self._nodes:
            if n.is_frontier and n.description == desc:
                return n
        return None

    def _remove_frontier_nodes(self, keep_cluster_ids: set[str]) -> None:
        from emet.memory.graph_eqa.frontier_nodes import FRONTIER_DESC_PREFIX

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
        self._rebuild_viewpoint_index()
        self._update_edges()

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

        from emet.memory.graph_eqa.frontier_nodes import (
            _as_bool_numpy,
            cluster_frontier_mask,
            hint_labels_near_grid,
            keyword_overlap_score,
        )

        try:
            outside = voxel_map.get_outside_frontier(xyt, planner)
            _, explored = voxel_map.get_2d_map()
        except Exception:
            return sum(1 for n in self._nodes if n.is_frontier)

        unexplored = _as_bool_numpy(outside) & ~_as_bool_numpy(explored)
        clusters = cluster_frontier_mask(
            unexplored,
            min_cells=self._frontier_min_cluster_cells,
        )
        image_descriptions = getattr(voxel_map, "image_descriptions", None) or []
        keywords = list(question_keywords or self._relevant_objects or self._enrich_object_hints or [])

        scored: list[tuple[float, str, tuple[int, int], int]] = []
        for cluster_id, grid_ij, cell_count in clusters:
            hints = hint_labels_near_grid(grid_ij, image_descriptions)
            kw_score = keyword_overlap_score(hints, keywords) if keywords else 0.0
            scored.append((kw_score, cluster_id, grid_ij, cell_count))
        scored.sort(key=lambda x: (-x[0], -x[3]))

        keep_ids: set[str] = set()
        step = self._effective_timestep()
        placeholder_rgb = np.zeros((8, 8, 3), dtype=np.uint8)

        for _score, cluster_id, grid_ij, _count in scored[: self._frontier_max_nodes]:
            keep_ids.add(cluster_id)
            gi, gj = grid_ij
            try:
                xy = voxel_map.grid_coords_to_xy(np.array([gi, gj], dtype=float))
            except Exception:
                continue
            xyz = np.array([float(xy[0]), float(xy[1]), 0.0], dtype=float)
            hints = hint_labels_near_grid(grid_ij, image_descriptions)
            labels = ["frontier"] + hints[:3]
            desc = self._frontier_desc(cluster_id)
            obs_desc = (
                "unexplored areas; " + ", ".join(hints) if hints else "This observation corresponds to unexplored space;"
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
                )
                for o in self._observations:
                    if int(o.obs_id) == int(existing.obs_id):
                        o.xyz = xyz.copy()
                        o.labels = list(labels)
                        o.description = obs_desc
                        break
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
                    )
                )
                self._observations.append(
                    GraphObservation(
                        obs_id=obs_id,
                        rgb=placeholder_rgb.copy(),
                        xyz=xyz.copy(),
                        labels=list(labels),
                        description=obs_desc,
                    )
                )

        self._remove_frontier_nodes(keep_ids)
        self._update_edges()
        return sum(1 for n in self._nodes if n.is_frontier)

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
        objects = [n for n in self._nodes if not n.is_viewpoint and not n.is_frontier]
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

    def _node_nav_status_suffix(self, node: GraphNode) -> str:
        failures = int(getattr(node, "nav_failures", 0) or 0)
        if failures <= 0:
            return ""
        note = (getattr(node, "last_nav_note", None) or "").strip()
        tail = f", last: {note}" if note else ""
        return f"; unreachable ({failures} nav failure(s){tail})"

    def record_nav_attempt(
        self,
        obs_id: int | None,
        *,
        success: bool,
        note: str,
        dist_m: float = 0.0,
        step: int | None = None,
    ) -> None:
        """Update graph node(s) tied to ``obs_id`` after an EQA navigation attempt."""
        if obs_id is None:
            self.last_nav_result_note = note
            return
        oid = int(obs_id)
        st = int(step if step is not None else self._effective_timestep())
        moved = float(dist_m) >= 0.12
        ok = bool(success) and moved
        for idx, node in enumerate(self._nodes):
            if int(node.obs_id) != oid:
                continue
            failures = int(getattr(node, "nav_failures", 0)) + (0 if ok else 1)
            self._nodes[idx] = replace(
                node,
                nav_attempts=int(getattr(node, "nav_attempts", 0)) + 1,
                nav_failures=failures,
                last_nav_note=str(note or "")[:120] or None,
                last_nav_at_step=st,
            )
        self.last_nav_result_note = note

    def append_nav_outcome_to_last_history(self, *, dist_m: float, success: bool, note: str) -> None:
        if not self._history_outputs:
            return
        status = "ok" if success else "failed"
        self._history_outputs[-1] += (
            f"\nNav_result: moved {float(dist_m):.2f}m ({status}; {note})"
        )

    def alternate_nav_target_for_failed_action(
        self,
        question: str,
        blocked_obs_id: int,
        planner: Any,
        base_xyt: Any,
    ) -> np.ndarray | None:
        """Pick a different frontier/fluid goal when the VLM re-picks a failed image action."""
        frontier_nodes = [
            n
            for n in self._nodes
            if getattr(n, "is_frontier", False) and int(n.obs_id) != int(blocked_obs_id)
        ]
        if frontier_nodes:
            frontier_nodes.sort(key=lambda n: (int(getattr(n, "nav_failures", 0)), -int(n.last_seen)))
            pick = frontier_nodes[0]
            return np.array([float(pick.xyz[0]), float(pick.xyz[1]), 1.0], dtype=float)
        return None

    def to_string(self) -> str:
        """Serialize the scene graph to a string for mLLM prompts."""
        lines = []

        def _prompt_labels(labels: list[str], max_len: int = 120) -> str:
            s = ", ".join(labels) if labels else "object"
            return s if len(s) <= max_len else s[: max_len - 3] + "..."

        for n in self._nodes:
            lbl = _prompt_labels(n.labels)
            sup = f" n={n.support_count}" if getattr(n, "support_count", 1) != 1 else ""
            if n.is_frontier:
                kind = "Frontier"
            elif n.is_viewpoint:
                kind = "View"
            else:
                kind = "Node"
            lines.append(
                f"{kind} {n.node_id}: {lbl} at ({n.xyz[0]:.2f}, {n.xyz[1]:.2f}, {n.xyz[2]:.2f}) [Image {n.obs_id}]{sup}{self._node_nav_status_suffix(n)}"
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

    def seed_object_hints(self, labels: str) -> None:
        """GraphEQA HM-EQA enrich labels (per-question object hints for planning)."""
        from emet.habitat.hmeqa_enrich_labels import parse_enrich_label_text

        self._enrich_object_hints = parse_enrich_label_text(labels)

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
        merged: list[str] = []
        enrich_hints = getattr(self, "_enrich_object_hints", None) or []
        for obj in (
            list(enrich_hints)
            + [s.strip() for s in out.split(",") if s.strip()]
            + heuristic_relevant_objects(question)
        ):
            key = obj.strip().lower()
            if key and key not in merged:
                merged.append(key)
        self._relevant_objects = merged[:4]

    def _select_relevant_obs_ids(self, max_images: int = 6) -> list[int]:
        """Select a diverse set of observation IDs for the EQA prompt (1-based).

        P2 diversification: instead of "all keyword matches then fill", build a
        prioritized pool so the VLM sees question-relevant views *and* a frontier
        view *and* a recent view *and* spatially spread context, capped at
        ``max_images``. Falls back to the most recent observations when there are
        no keyword objects.
        """
        if not self._observations:
            return []
        if max_images <= 0:
            return []
        if not self._relevant_objects:
            return [o.obs_id for o in self._observations[-max_images:]]

        by_id = {int(o.obs_id): o for o in self._observations}
        selected: list[int] = []

        def take(oid: int) -> bool:
            oid = int(oid)
            if oid in selected or oid not in by_id:
                return False
            selected.append(oid)
            return len(selected) >= max_images

        # 0) SigLIP-matched observation per relevant object (caption-independent). Guarantees the
        #    VLM is shown the best view of the target object even when it was captioned as
        #    something else, instead of reasoning over whatever furniture happens to be in frame.
        obs_grounder = getattr(self, "_obs_id_grounder", None)
        if obs_grounder is not None:
            for obj in self._relevant_objects:
                try:
                    oid = obs_grounder(obj)
                except Exception:
                    oid = None
                if oid is not None and take(int(oid)):
                    return selected

        # 1) Keyword matches (question-relevant objects), most recent first.
        keyword_hits: list[int] = []
        for obj in self._relevant_objects:
            obj_lower = obj.lower()
            for o in reversed(self._observations):
                if int(o.obs_id) in keyword_hits:
                    continue
                if any(obj_lower in lab.lower() for lab in o.labels):
                    keyword_hits.append(int(o.obs_id))
        # Reserve at least one slot each for frontier + recent when budget allows.
        reserved = 0
        if max_images >= 3:
            reserved = min(2, max_images - 1)
        keyword_budget = max(1, max_images - reserved)
        for oid in keyword_hits[:keyword_budget]:
            if take(oid):
                return selected

        # 2) One frontier-tagged observation (prefer lowest nav_failures).
        frontier_candidates = [
            int(o.obs_id)
            for o in reversed(self._observations)
            if self._obs_is_frontier(int(o.obs_id)) and int(o.obs_id) not in selected
        ]
        if frontier_candidates:
            frontier_by_id = {int(n.obs_id): n for n in self._nodes if getattr(n, "is_frontier", False)}
            frontier_candidates.sort(
                key=lambda oid: (
                    int(getattr(frontier_by_id.get(oid), "nav_failures", 0)) if oid in frontier_by_id else 0,
                    -oid,
                )
            )
            if take(frontier_candidates[0]):
                return selected

        # 3) Most recent observation (fresh context).
        for o in reversed(self._observations):
            if take(int(o.obs_id)):
                return selected
            break

        # 4) Spatial spread: greedily add observations farthest from those chosen.
        remaining = [int(o.obs_id) for o in self._observations if int(o.obs_id) not in selected]
        while remaining and len(selected) < max_images:
            best_oid = None
            best_dist = -1.0
            for oid in remaining:
                cand = by_id[oid].xyz[:2]
                if selected:
                    d = min(
                        float(np.linalg.norm(cand - by_id[s].xyz[:2]))
                        for s in selected
                        if s in by_id
                    )
                else:
                    d = 0.0
                if d > best_dist:
                    best_dist = d
                    best_oid = oid
            if best_oid is None:
                break
            selected.append(best_oid)
            remaining.remove(best_oid)

        return selected

    def set_text_grounder(
        self, grounder: Callable[[str], tuple[float, np.ndarray] | None] | None
    ) -> None:
        """Register an open-vocab visual grounder: ``text -> (similarity, xyz) | None``.

        Backed by the voxel map's SigLIP features so existence/location can be grounded in
        pixels rather than the VLM's caption-derived node labels.
        """
        self._text_grounder = grounder

    def set_obs_id_grounder(self, grounder: Callable[[str], int | None] | None) -> None:
        """Register an open-vocab ``text -> obs_id`` selector (SigLIP-backed).

        Used by image selection to force the best-aligned observation of each relevant object
        into the VLM prompt regardless of its caption label.
        """
        self._obs_id_grounder = grounder

    def _relevant_memory_summary(self) -> str:
        """Surface question-relevant objects as 'confirmed memory' for the VLM.

        Combines two grounding signals so the model does not have to rely on the attached
        images (it otherwise reports 'none' for objects it cannot currently see):
          * graph nodes whose caption-derived labels match the object, and
          * a SigLIP visual match over all observed points (independent of captions) — this
            catches objects that were seen but mislabeled (e.g. a woven basket captioned as
            a "decorative plant").
        """
        if not self._relevant_objects:
            return ""
        object_nodes = [n for n in self._nodes if not n.is_frontier and not n.is_viewpoint]
        grounder = self._text_grounder
        present_thresh = SIGLIP_PRESENT_THRESHOLD
        lines: list[str] = []
        for obj in self._relevant_objects:
            obj_l = obj.lower()
            matches = [
                n
                for n in object_nodes
                if any(label_matches_relevant_object(obj, lab) for lab in n.labels)
            ]
            sig: tuple[float, np.ndarray] | None = None
            if grounder is not None:
                try:
                    sig = grounder(obj)
                except Exception:
                    sig = None
            parts: list[str] = []
            if matches:
                positions = ", ".join(f"({n.xyz[0]:.1f}, {n.xyz[1]:.1f})" for n in matches[:4])
                parts.append(f"{len(matches)} graph node(s) at {positions}")
            sig_present = sig is not None and float(sig[0]) >= present_thresh
            if sig is not None:
                sim, xyz = float(sig[0]), sig[1]
                if sig_present:
                    parts.append(f"SigLIP visual match sim={sim:.2f} near ({xyz[0]:.1f}, {xyz[1]:.1f})")
                else:
                    parts.append(f"no strong SigLIP match (sim={sim:.2f})")
            present = bool(matches) or sig_present
            if present:
                lines.append(f"- {obj}: PRESENT — " + "; ".join(parts))
            elif parts:
                lines.append(f"- {obj}: likely NOT present — " + "; ".join(parts))
            else:
                lines.append(f"- {obj}: not observed during exploration")
        if not lines:
            return ""
        header = (
            "CONFIRMED_MEMORY (grounded in observed graph nodes + SigLIP visual matches; "
            "trust these for existence/counting/location even if they are not in the "
            "attached images):"
        )
        return header + "\n" + "\n".join(lines)

    def _graph_covers_relevant_objects(self) -> bool:
        """True when every keyword object appears in at least one graph node label."""
        eqa_cfg = self.parameters.get("eqa", {}) if hasattr(self.parameters, "get") else {}
        if isinstance(eqa_cfg, dict) and eqa_cfg.get("sqa3d_allow_partial_graph"):
            return True
        if not self._relevant_objects or not self._observations:
            return True
        for obj in self._relevant_objects:
            if not any(
                label_matches_relevant_object(obj, lab)
                for o in self._observations
                for lab in o.labels
            ):
                return False
        return True

    def _obs_is_frontier(self, obs_id: int) -> bool:
        for n in self._nodes:
            if int(n.obs_id) == int(obs_id) and n.is_frontier:
                return True
        return False

    def _get_image_descriptions_str(self, obs_ids: list[int]) -> str:
        """Build IMAGE_DESCRIPTIONS for attached EQA images only (Image 1..N)."""
        if not obs_ids:
            return "IMAGE_DESCRIPTIONS: (none)"
        id_to_obs = {int(o.obs_id): o for o in self._observations}
        options: list[str] = []
        for img_idx, oid in enumerate(obs_ids, start=1):
            obs = id_to_obs.get(int(oid))
            if obs is None:
                continue
            lbl = ", ".join(obs.labels) if obs.labels else "object"
            line = f"Image {img_idx}. {lbl} at ({obs.xyz[0]:.2f}, {obs.xyz[1]:.2f});"
            node = next((n for n in self._nodes if int(n.obs_id) == int(obs.obs_id)), None)
            if node is not None:
                line += self._node_nav_status_suffix(node)
            if self._obs_is_frontier(obs.obs_id):
                line += " unexplored frontier;"
            elif obs.description and "unexplored" in obs.description.lower():
                line += f" {obs.description.strip()};"
            options.append(line)
        return "IMAGE_DESCRIPTIONS: " + "\n".join(options) if options else "IMAGE_DESCRIPTIONS: (none)"

    def parse_answer(self, answer_outputs: str) -> tuple[str, str, bool, str, str]:
        """Parse mLLM output into reasoning, answer, confidence, action, confidence_reasoning."""
        text = answer_outputs or ""
        lowered = text.lower()

        def extract_between(src: str, start: str, end: str) -> str:
            pattern = re.compile(
                rf"{re.escape(start)}\s*(.*?)\s*{re.escape(end)}",
                flags=re.IGNORECASE | re.DOTALL,
            )
            m = pattern.search(src)
            if not m:
                return ""
            return m.group(1).strip().replace("\n", " ").replace("\t", " ")

        def extract_after(src: str, start: str) -> str:
            pattern = re.compile(rf"{re.escape(start)}\s*(.*)", flags=re.IGNORECASE | re.DOTALL)
            m = pattern.search(src)
            if not m:
                return ""
            return m.group(1).strip().replace("\n", " ").replace("\t", " ")

        reasoning = extract_between(lowered, "reasoning:", "answer:")
        answer = extract_between(lowered, "answer:", "confidence:")
        confidence_text = extract_between(lowered, "confidence:", "action:")
        confidence = "true" in confidence_text.replace(" ", "").lower()
        action = extract_between(lowered, "action:", "confidence_reasoning:")
        confidence_reasoning = extract_after(lowered, "confidence_reasoning:")
        if not answer.strip():
            m = re.search(r"answer\s*:\s*([a-d])\b", lowered)
            if m:
                answer = m.group(1).upper()
        if not answer.strip():
            m = re.search(r"(?:^|\n)\s*([a-d])\s*(?:\n|$)", lowered)
            if m:
                answer = m.group(1).upper()
        return reasoning, answer, confidence, action, confidence_reasoning

    def _salvage_answer_letter(self, question: str, commands: list[Any]) -> str:
        """Terse follow-up when the main EQA output never produced an ``answer:`` field.

        Reuses the attached images from ``commands`` and asks for only a letter, which
        recovers runaway-caption episodes (the small VLM loops before emitting answer).
        """
        if self.eqa_client is None:
            return ""
        images = [c for c in commands if isinstance(c, Image.Image)]
        directive = (
            "Answer the multiple-choice question with ONLY a single letter (A, B, C, or D). "
            "Do not caption images. Do not explain. Output just the letter.\n"
            f"Question: {question}"
        )
        try:
            salvage_raw = self.eqa_client([directive, *images])
        except Exception:
            return ""
        text = (salvage_raw or "").strip()
        m = re.search(r"\b([A-D])\b", text)
        if m:
            return m.group(1)
        m = re.search(r"([A-D])", text)
        return m.group(1) if m else ""

    def vote_mcq_letter(self, question: str, choices: list[str]) -> str:
        """Debiased final MCQ letter (see mcq_debias.py).

        Two stages, both letter-token-free at the selection step:
          1. Free-form ask ("answer in a few words", no choices shown) matched to the
             closest choice by token overlap — immune to MCQ position bias.
          2. Fallback: choice-rotation voting — re-ask with cyclically rotated choice
             orders, map each reply back to the original choice index, majority-vote.
        Returns the winning original letter, or ``""`` when neither stage finds a
        clear signal (caller keeps its main answer). Details in ``self.last_mcq_debias``.
        """
        self.last_mcq_debias = {}
        if self.eqa_client is None or len(choices) < 2:
            return ""
        n = min(4, len(choices))
        images = [
            Image.fromarray(o.rgb.astype(np.uint8), mode="RGB")
            for o in self._observations
            if o.obs_id in set(self.last_eqa_obs_ids)
        ]

        freeform_directive = (
            "Look at the images and answer the question in a few words. "
            "Do not use option letters. Do not caption images. Do not explain.\n"
            f"Question: {question}"
        )
        try:
            freeform = (self.eqa_client([freeform_directive, *images]) or "").strip()
        except Exception:
            freeform = ""
        ff_idx = match_freeform_to_choice(freeform, choices[:n])
        if ff_idx is not None:
            letter = LETTERS[ff_idx]
            self.last_mcq_debias = {
                "letter": letter,
                "freeform": freeform[:300],
                "freeform_match": letter,
                "votes": [],
                "prior": None,
                "replies": [],
            }
            return letter

        prior_index = letter_to_original_index(
            extract_single_letter(self.last_eqa_parsed[1], n), rotation=0, n_choices=n
        )
        votes: list[int | None] = []
        replies: list[str] = []
        for r in range(n):
            formatted = format_rotated_question(question, choices[:n], r)
            directive = (
                "Answer the multiple-choice question with ONLY a single letter "
                f"(one of {', '.join(LETTERS[:n])}). Do not caption images. Do not "
                f"explain. Output just the letter.\nQuestion: {formatted}"
            )
            try:
                reply = self.eqa_client([directive, *images])
            except Exception:
                reply = ""
            replies.append((reply or "").strip()[:200])
            votes.append(letter_to_original_index(extract_single_letter(reply, n), r, n))
        win = tally_choice_votes(votes, choices[:n], prior_index=prior_index)
        letter = LETTERS[win] if win is not None else ""
        self.last_mcq_debias = {
            "letter": letter,
            "freeform": freeform[:300],
            "freeform_match": None,
            "votes": [None if v is None else LETTERS[v] for v in votes],
            "prior": None if prior_index is None else LETTERS[prior_index],
            "replies": replies,
        }
        return letter

    def _target_point_from_image_id(self, image_id: int) -> np.ndarray | None:
        """Return (x, y, 1) for the observation's position when mLLM suggests navigating to that image."""
        for obs in self._observations:
            if obs.obs_id == image_id:
                return np.array([obs.xyz[0], obs.xyz[1], 1.0], dtype=float)
        return None

    def _target_point_from_display_image_index(
        self,
        display_index: int,
        *,
        obs_ids: list[int],
        nav_fallback_tail: list[GraphNavigationSample],
    ) -> np.ndarray | None:
        """Map 1-based ``Image N`` from the EQA prompt to a navigation waypoint."""
        if display_index < 1:
            return None
        if self._observations and obs_ids:
            if display_index > len(obs_ids):
                return None
            oid = int(obs_ids[display_index - 1])
            for obs in self._observations:
                if int(obs.obs_id) == oid:
                    return np.array([obs.xyz[0], obs.xyz[1], 1.0], dtype=float)
            return self._target_point_from_image_id(oid)
        if nav_fallback_tail and display_index <= len(nav_fallback_tail):
            nv = nav_fallback_tail[display_index - 1]
            return np.array([nv.xyz[0], nv.xyz[1], 1.0], dtype=float)
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
        from emet.llms.eqa_vl_settings import get_eqa_vl_int

        self._ensure_llm_clients()
        self.extract_relevant_objects(question)
        max_images = get_eqa_vl_int(self.parameters, "eqa_max_images", 4)
        obs_ids = self._select_relevant_obs_ids(max_images=max_images)
        self.last_eqa_obs_ids = list(obs_ids)
        graph_str = self.to_string()
        nav_fallback_tail: list[GraphNavigationSample] = []
        if self._observations:
            img_desc_str = self._get_image_descriptions_str(obs_ids)
        elif self._nav_samples:
            nav_fallback_tail = self._nav_samples[-max_images:]
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
        if self.memory_summary_enabled:
            memory_summary = self._relevant_memory_summary()
            if memory_summary:
                commands.append(memory_summary)
        commands.append("HISTORY: ")
        # Only include the most recent iterations: unbounded history bloats the prompt
        # and feeds the model its own repeated outputs, which drives caption/action loops.
        max_history = get_eqa_vl_int(self.parameters, "eqa_max_history", 4)
        history = self._history_outputs
        start = max(0, len(history) - max_history) if max_history > 0 else 0
        for i, h in enumerate(history[start:], start=start):
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
        self.last_eqa_nav_fallback_count = len(nav_fallback_tail)

        try:
            raw = self.eqa_client(commands)
        except Exception as exc:
            raw = f"Error: {exc}"
            self.last_eqa_raw = raw
            self.last_eqa_parsed = ("", "Unknown", False, "", str(exc))
            self._history_outputs.append(
                "Answer:Unknown\nReasoning:"
                + str(exc)
                + "\nConfidence:False\nAction:\nConfidence_reasoning:"
                + str(exc)
            )
            return (
                str(exc),
                "Unknown",
                False,
                str(exc),
                None,
                relevant_images,
            )
        self.last_eqa_raw = raw
        answer_outputs = raw.replace("*", "").replace("/", "").replace("#", "").lower()

        reasoning, answer, confidence, action, confidence_reasoning = self.parse_answer(answer_outputs)
        # Salvage: small VLMs sometimes run away captioning and never emit ``answer:``.
        # Re-ask tersely for just the choice letter using the same images/question.
        if not answer.strip():
            salvage = self._salvage_answer_letter(question, commands)
            if salvage:
                answer = salvage
                raw = (raw or "") + f"\n[salvage]\nanswer:\n{salvage}\n"
                self.last_eqa_raw = raw
        self.last_eqa_model_confident = bool(confidence)
        if confidence and not self._graph_covers_relevant_objects():
            confidence = False
            confidence_reasoning = (
                confidence_reasoning
                + " The scene graph does not yet include all question-relevant objects; explore further."
            ).strip()
        raw_answer = answer
        self.last_eqa_parsed = (reasoning, raw_answer, confidence, action, confidence_reasoning)
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
        self.last_eqa_action_obs_id = None
        if not confidence and action.strip():
            match = re.search(r"\d+", action.strip())
            if match:
                display_index = int(match.group())
                if self._observations and obs_ids and 1 <= display_index <= len(obs_ids):
                    self.last_eqa_action_obs_id = int(obs_ids[display_index - 1])
                target_point = self._target_point_from_display_image_index(
                    display_index,
                    obs_ids=obs_ids,
                    nav_fallback_tail=nav_fallback_tail,
                )
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
