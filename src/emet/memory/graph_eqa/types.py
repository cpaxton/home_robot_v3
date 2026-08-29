# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Graph-memory dataclasses and recall-tier constants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

_WEAK_SIGLIP_FIND_TOKENS = frozenset({"time", "hour", "now", "today", "moment"})

_RECALL_SOURCE_TIER: dict[str, float] = {
    "voxel": 400.0,
    "graph": 300.0,
    "confirmed": 200.0,
    "siglip": 100.0,
    "frontier": 0.0,
}

GT_BODY_DESC_PREFIX = "ground_truth:"


@dataclass(frozen=True)
class NavHypothesis:
    """Retrieved navigation evidence card (voxel / graph / CONFIRMED_MEMORY / SigLIP / frontier).

    ``score`` is an internal recall rank key for top-K packing and ROUTER=0 fallback
    order — not a policy signal for the VLM router. ``source=voxel`` /
    ``obs_id <= -3_000_000`` is a localize_text handle, not a camera observation.
    """

    phrase: str
    obs_id: int
    xyz: np.ndarray
    score: float
    source: str  # "voxel" | "graph" | "confirmed" | "siglip" | "frontier"
    answerability_gain: float = 0.0
    belief_reduction: float = 0.0
    revisit_change_value: float = 0.0
    path_cost: float = 0.0
    failure_risk: float = 0.0
    siglip_sim: float | None = None
    # Voxel/detector detections: 1.0 = YoloE, else gated cosine. Not a camera view.
    confidence: float | None = None
    yoloe_hit: bool | None = None


@dataclass(frozen=True)
class RelationBelief:
    """Timestamped uncertain context relation."""

    source_id: int
    target_id: int
    relation: str
    confidence: float
    last_evidence_step: int
    contradiction_count: int = 0


@dataclass(frozen=True)
class VerifyResult:
    """SigLIP (+ optional graph-label) verification of a phrase at an observation."""

    # "UNAVAILABLE" means the encoder was gone (released for the VLM), not evidence of absence.
    status: str  # "PRESENT" | "CANDIDATE" | "ABSENT" | "UNAVAILABLE"
    sim: float
    obs_id: int
    phrase: str
    ok: bool = False
    text_feat: np.ndarray | None = None
    img_feat: np.ndarray | None = None


@dataclass
class GraphNavigationSample:
    """A viewpoint along the run without an object-level graph node (RGB + anchors)."""

    rgb: np.ndarray
    xyz: np.ndarray  # (3,) scene anchor (e.g. depth median in world frame)
    base_xyz: np.ndarray | None = None


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
    frontier_cell_count: int = 0  # frontier only: unexplored cells in this cluster (area gain)
    frontier_keyword_score: float = 0.0  # frontier only: question-keyword affinity of nearby hints
    embedding: np.ndarray | None = None  # optional visual embedding (e.g. SigLIP crop)
    bounds_3d: dict[str, list[float]] | None = None  # axis-aligned world bounds {min,max,center,size}
    nav_attempts: int = 0
    nav_failures: int = 0
    last_nav_note: str | None = None
    last_nav_at_step: int = 0
    belief_confidence: float = 0.5
    position_covariance: np.ndarray | None = None
    position_history: list[dict[str, Any]] = field(default_factory=list)
    identity_key: str | None = None
    # True only when this node came from instance-level evidence. Label-only
    # frame summaries are not safe inputs for exact count hints.
    countable_instance: bool = False
    # Qwen caption after a close look / vlm_assess. Preferred over detector class names.
    close_look_label: str | None = None
    change_events: list[dict[str, Any]] = field(default_factory=list)
    expected_absence_count: int = 0
    last_absence_step: int = -1


@dataclass
class GraphObservation:
    """One observation (image + pose + labels) used to build the graph."""

    obs_id: int  # 1-based
    rgb: np.ndarray  # (H, W, 3)
    xyz: np.ndarray  # (3,) e.g. mean of visible points or camera position
    labels: list[str]
    description: str | None = None  # optional VLM-generated description
    viewer_xyz: np.ndarray | None = None


def is_ground_truth_node(node: GraphNode | None) -> bool:
    """True when ``node.description`` marks sim GT (stable ``body_name`` key)."""
    if node is None:
        return False
    desc = getattr(node, "description", None)
    return isinstance(desc, str) and desc.startswith(GT_BODY_DESC_PREFIX)
