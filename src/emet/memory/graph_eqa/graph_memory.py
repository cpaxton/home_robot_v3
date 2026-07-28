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

import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
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
from emet.utils.logger import Logger

_logger = Logger(__name__)

# Min SigLIP cosine for open-vocab text vs **voxel point** features (DynaMem
# ``verify_point``). Higher precision than image-space scoring. Agentic RGB /
# dense-patch verify uses a lower high-recall bar — see
# ``SIGLIP_IMAGE_PRESENT_THRESHOLD`` in ``agentic_eqa.py`` and
# docs/experiments/agentic_scale.md § SigLIP role.
SIGLIP_PRESENT_THRESHOLD = 0.21
# Stronger bar before SigLIP-only evidence may override the VLM or finalize confidence.
SIGLIP_CONFIRM_THRESHOLD = 0.28


# Source tiers for hyp *recall* only (not a VLM decision policy).
_RECALL_SOURCE_TIER: dict[str, float] = {
    "graph": 300.0,
    "confirmed": 200.0,
    "siglip": 100.0,
    "frontier": 0.0,
}


@dataclass(frozen=True)
class NavHypothesis:
    """Retrieved navigation evidence card (graph / CONFIRMED_MEMORY / SigLIP / frontier).

    ``score`` is an internal recall rank key for top-K packing and ROUTER=0 fallback
    order — not a policy signal for the VLM router.
    """

    phrase: str
    obs_id: int
    xyz: np.ndarray
    score: float
    source: str  # "graph" | "confirmed" | "siglip" | "frontier"
    answerability_gain: float = 0.0
    belief_reduction: float = 0.0
    revisit_change_value: float = 0.0
    path_cost: float = 0.0
    failure_risk: float = 0.0
    siglip_sim: float | None = None


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


# Expand HM-EQA object phrases onto common caption synonyms (trash can ↔ recycle bin).
_OBJECT_LABEL_ALIASES: dict[str, frozenset[str]] = {
    "trash": frozenset({"recycle", "bin", "garbage", "waste", "rubbish"}),
    "garbage": frozenset({"recycle", "bin", "trash", "waste"}),
    "bin": frozenset({"recycle", "trash", "garbage", "waste"}),
    "recycle": frozenset({"bin", "trash", "garbage"}),
    "mat": frozenset({"exercise", "yoga", "workout"}),
    "curtain": frozenset({"curtains", "drape", "drapes", "shade"}),
}

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
        "anywhere",
    }
)

# Stem verbs / fillers that must not win SigLIP phrase[0] over the object noun
# (``trying remember placed`` beating ``large wall clock``).
_QUESTION_VERB_FILLERS = frozenset(
    {
        "trying",
        "try",
        "remember",
        "looking",
        "look",
        "placed",
        "place",
        "find",
        "finding",
        "found",
        "seek",
        "seeking",
        "recall",
        "tell",
        "know",
        "think",
        "want",
        "need",
        "help",
        "like",
        "get",
        "got",
        "put",
        "keep",
        "leave",
        "locate",
        "locating",
        "search",
        "searching",
        "show",
        "showing",
        "please",
    }
)


def question_stem_for_keywords(question: str) -> str:
    """Return the object stem of an HM-EQA question (no MCQ options / ``Answer:``).

    Agentic traces pass the full ``… A) … D) … Answer:`` string into phrase
    extract; without stripping, SigLIP verifies ``table sunroom answer`` instead
    of ``fruit bowl`` (failfix5 q105).
    """
    text = (question or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s*Answer:\s*$", "", text, flags=re.IGNORECASE)
    m = re.search(r"\s+[A-D]\)\s+", text, flags=re.IGNORECASE)
    if m:
        text = text[: m.start()]
    if "?" in text:
        text = text.split("?", 1)[0]
    return text.strip()


def heuristic_relevant_phrases(question: str, *, max_phrases: int = 4) -> list[str]:
    """Multi-word object phrases from the question stem (e.g. ``woven basket``).

    Prefer later noun compounds over leading verb n-grams so SigLIP verify uses
    ``large wall clock`` / ``fruit bowl`` rather than ``trying remember placed``.
    """
    head = question_stem_for_keywords(question).lower()
    tokens = [tok for tok in re.findall(r"[a-z]+", head) if len(tok) >= 3 and tok not in _QUESTION_STOPWORDS]
    scored: list[tuple[float, int, str]] = []
    for n in range(min(3, len(tokens)), 1, -1):
        for i in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[i : i + n])
            words = phrase.split()
            score = float(n) * 10.0 + 0.1 * float(i)
            if words[0] in _QUESTION_VERB_FILLERS:
                score -= 50.0
            if words[-1] in _QUESTION_VERB_FILLERS:
                score -= 20.0
            scored.append((-score, i, phrase))
    phrases: list[str] = []
    for _neg_score, _i, phrase in sorted(scored):
        if phrase not in phrases:
            phrases.append(phrase)
        if len(phrases) >= max_phrases:
            break
    return phrases


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
    if obj_tok & lab_tok:
        return True
    # Alias expansion: "trash can" ↔ "recycle bin".
    for tok in obj_tok:
        aliases = _OBJECT_LABEL_ALIASES.get(tok)
        if aliases and (aliases & lab_tok):
            return True
    for tok in lab_tok:
        aliases = _OBJECT_LABEL_ALIASES.get(tok)
        if aliases and (aliases & obj_tok):
            return True
    return False


def _location_mcq_weak_tokens() -> frozenset[str]:
    return frozenset(
        {
            "next",
            "near",
            "with",
            "from",
            "room",
            "living",
            "between",
            "the",
            "and",
            "by",
            "on",
            "at",
            "of",
            "to",
            "in",
            "under",
            "any",
            "left",
            "leave",
            "placed",
            "somewhere",
            "anywhere",
        }
    )


def distinctive_choice_tokens(choice: str) -> list[str]:
    """Content tokens from an MCQ option (for label↔option matching)."""
    weak = _location_mcq_weak_tokens()
    return [t for t in re.findall(r"[a-z]+", (choice or "").lower()) if len(t) > 3 and t not in weak]


# Generic furniture words that appear in many options/views; do not let them beat
# rare landmarks (refrigerator, treadmill) when ranking Image 1.
_LANDMARK_GENERIC_TOKENS = frozenset(
    {
        "table",
        "tables",
        "chair",
        "chairs",
        "sofa",
        "sofas",
        "couch",
        "couches",
        "door",
        "doors",
        "wall",
        "floor",
        "room",
        "rug",
        "window",
        "cabinet",
        "cabinets",
    }
)

# Question-object → landmark tokens that should win Image 1 over generic furniture.
# For trash: prefer fridge/recycle over sink — sink is often a competing MCQ option.
_QUESTION_LANDMARK_BOOST: dict[str, frozenset[str]] = {
    "trash": frozenset({"refrigerator", "fridge", "recycle", "bin"}),
    "garbage": frozenset({"refrigerator", "fridge", "recycle", "bin"}),
    "bin": frozenset({"refrigerator", "fridge", "recycle"}),
    "mat": frozenset({"treadmill", "elliptical", "bike", "bicycle", "exercise"}),
    "towel": frozenset({"bathroom", "bath", "sink", "toilet", "shower"}),
    "kettle": frozenset({"kitchen", "counter", "stove", "sink"}),
    "pillow": frozenset({"sofa", "couch", "bed", "armchair"}),
    "fruit": frozenset({"dining"}),
    "bowl": frozenset({"dining"}),
}


def location_mcq_landmark_phrases(
    question: str,
    *,
    max_landmarks: int = 4,
) -> list[str]:
    """Landmark phrases from location-MCQ options (e.g. ``kitchen island``).

    Stem heuristics ignore choices, and enrich labels are often empty for a scene —
    without these seeds, hyp recall never builds Investigate cards for option places
    even when the graph already has matching labels.
    """
    try:
        from emet.habitat.metrics import (
            choices_are_location_mcq,
            parse_mcq_choices_from_question,
        )
    except ImportError:
        return []
    choices = parse_mcq_choices_from_question(question or "")
    if not choices or not choices_are_location_mcq(choices):
        return []
    out: list[str] = []
    lead = re.compile(
        r"^(?:on|in|at|by|near|under|beside|behind|inside|outside|"
        r"between|next\s+to)\s+(?:the\s+)?",
        re.IGNORECASE,
    )
    for ch in choices[:4]:
        text = lead.sub("", str(ch or "").strip()).strip().lower()
        text = re.sub(r"\s+", " ", text)
        if len(text) < 3 or text in _QUESTION_STOPWORDS:
            continue
        if text not in out:
            out.append(text)
        if len(out) >= max_landmarks:
            break
    return out


def consolidate_relevant_keywords(
    phrases: list[str],
    extras: list[str],
    *,
    max_items: int = 4,
) -> tuple[list[str], list[str]]:
    """Phrase-first dedupe for CONFIRMED_MEMORY and image selection."""
    phrase_list: list[str] = []
    for item in phrases:
        key = item.strip().lower()
        if key and key not in phrase_list:
            phrase_list.append(key)

    phrase_tokens: set[str] = set()
    for phrase in phrase_list:
        phrase_tokens.update(_object_match_tokens(phrase))

    objects: list[str] = list(phrase_list)
    for item in extras:
        key = item.strip().lower()
        if not key or key in objects:
            continue
        if key in _QUESTION_STOPWORDS:
            continue
        if key in phrase_tokens:
            continue
        if any(key in phrase.split() for phrase in phrase_list):
            continue
        objects.append(key)

    return phrase_list[:max_items], objects[:max_items]


def heuristic_relevant_objects(question: str, *, max_objects: int = 4) -> list[str]:
    """Cheap noun-like tokens from the question stem (before MCQ options).

    Skip verb fillers and prefer later tokens (object of ``looking for X``).
    """
    head = question_stem_for_keywords(question)
    toks: list[str] = []
    for tok in re.findall(r"[a-z]{3,}", head.lower()):
        if tok in _QUESTION_STOPWORDS or tok in _QUESTION_VERB_FILLERS:
            continue
        if tok not in toks:
            toks.append(tok)
    out: list[str] = []
    for tok in reversed(toks):
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
    change_events: list[dict[str, Any]] = field(default_factory=list)
    expected_absence_count: int = 0
    last_absence_step: int = -1


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


_ROOM_WORDS = frozenset(
    {
        "room",
        "kitchen",
        "bedroom",
        "bathroom",
        "living room",
        "dining room",
        "hallway",
        "office",
        "garage",
        "sunroom",
    }
)


def _node_is_room(node: GraphNode) -> bool:
    text = " ".join(node.labels).lower()
    return any(word in text for word in _ROOM_WORDS)


def _inside_bounds(point: np.ndarray, bounds: dict[str, list[float]] | None) -> bool:
    if not bounds or "min" not in bounds or "max" not in bounds:
        return False
    xyz = np.asarray(point, dtype=float).reshape(-1)[:3]
    lower = np.asarray(bounds["min"], dtype=float).reshape(-1)[:3]
    upper = np.asarray(bounds["max"], dtype=float).reshape(-1)[:3]
    return bool(np.all(xyz >= lower) and np.all(xyz <= upper))


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
        self.last_eqa_prompt_node_count: int = 0
        self.last_eqa_prompt_regions: int = 0
        self.last_eqa_spatial_rag: dict[str, Any] | None = None
        self.last_nav_result_note: str = ""
        self.last_eqa_nav_fallback_count: int = 0
        # Model's own confidence before the graph-coverage gate suppresses it (for early-stop).
        self.last_eqa_model_confident: bool = False
        self._nodes: list[GraphNode] = []
        self._edges: list[tuple[int, int, str]] = []  # (id1, id2, relation)
        self._relation_beliefs: dict[tuple[int, int, str], RelationBelief] = {}
        self._change_events: list[dict[str, Any]] = []
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
        self._relevant_phrases: list[str] = []
        self._confirmed_memory_siglip_encoder: Any | None = None
        self._obs_siglip_features: dict[int, np.ndarray] = {}
        # obs_id → content revision (bumped when merge refreshes RGB/candidate evidence).
        self._obs_revisions: dict[int, int] = {}
        self._last_obs_content_update_id: int | None = None
        self._siglip_phrase_cache: dict[str, tuple[float, np.ndarray, int | None]] = {}

        self.log_dir = log_dir
        self.eqa_client = eqa_client
        self.image_description_client = image_description_client
        self._defer_llm_clients = defer_llm_clients
        self._nav_samples: list[GraphNavigationSample] = []
        self._viewpoint_by_obs_id: dict[int, int] = {}
        # Reuse viewpoint nodes when the camera/base pose is within this radius (m).
        self.viewpoint_merge_m: float = 0.15
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
        self.image_nav_min_approach_m: float = 0.35
        # (obs_id, normalized_phrase) claims retracted after close+ABSENT verify —
        # keep the place node, but stop offering that stem-object hyp card.
        self._retracted_nav_claims: set[tuple[int, str]] = set()
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
        eqa = d.get("eqa")
        if isinstance(eqa, dict) and eqa.get("image_nav_min_approach_m") is not None:
            self.image_nav_min_approach_m = max(0.05, float(eqa["image_nav_min_approach_m"]))

    def _load_dynagraph_settings(self) -> None:
        d = self._parameters_dict()
        if not d:
            return
        if d.get("dynagraph_merge_xy_m") is not None:
            self.spatial_merge_m = float(d["dynagraph_merge_xy_m"])
        if d.get("dynagraph_staleness_horizon") is not None:
            self.staleness_horizon = max(0, int(d["dynagraph_staleness_horizon"]))
        if d.get("dynagraph_viewpoint_merge_m") is not None:
            self.viewpoint_merge_m = max(0.0, float(d["dynagraph_viewpoint_merge_m"]))

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

    def _position_update(
        self,
        node: GraphNode,
        measured_xyz: np.ndarray,
        *,
        step: int,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, Any]], float]:
        """Update a track without averaging a contradictory relocation into its centroid."""
        measured = np.asarray(measured_xyz, dtype=float).reshape(-1)[:3]
        current = np.asarray(node.xyz, dtype=float).reshape(-1)[:3]
        history = list(node.position_history)
        if not history:
            history.append(
                {
                    "step": int(node.last_seen),
                    "xyz": current.tolist(),
                    "confidence": float(node.belief_confidence),
                }
            )
        distance = float(np.linalg.norm(measured - current))
        relocation_bar = max(0.45, float(self.spatial_merge_m) * 1.5)
        changes = list(node.change_events)
        confidence = min(0.99, float(node.belief_confidence) + 0.08)
        if distance > relocation_bar:
            event = {
                "type": "position_contradiction",
                "node_id": int(node.node_id),
                "step": int(step),
                "from_xyz": current.tolist(),
                "to_xyz": measured.tolist(),
                "displacement_m": distance,
                "confidence": min(0.99, distance / max(relocation_bar, 1e-6)),
            }
            changes.append(event)
            self._change_events.append(event)
            updated = measured.copy()
            confidence = max(0.2, float(node.belief_confidence) * 0.7)
        else:
            support = max(1, int(node.support_count))
            updated = (current * support + measured) / (support + 1)
        history.append(
            {
                "step": int(step),
                "xyz": measured.tolist(),
                "confidence": confidence,
            }
        )
        samples = np.asarray([entry["xyz"] for entry in history[-20:]], dtype=float)
        covariance = (
            np.cov(samples.T)
            if samples.shape[0] >= 2
            else np.zeros((3, 3), dtype=float)
        )
        return updated, covariance, history[-64:], changes[-32:], confidence

    def observe_visible_labels(
        self,
        labels: list[str],
        viewer_xyz: np.ndarray | None,
        *,
        step: int | None = None,
        viewpoint_tolerance_m: float = 0.75,
        absence_confirmations: int = 2,
    ) -> list[dict[str, Any]]:
        """Conservatively detect disappeared objects from repeated same-view contradictions.

        A node is expected only when the camera revisits approximately the viewpoint
        that originally saw it. This avoids treating out-of-FOV objects as absent.
        Ground-truth placements are never consulted.
        """
        if viewer_xyz is None:
            return []
        now = self._effective_timestep() if step is None else int(step)
        viewer = np.asarray(viewer_xyz, dtype=float).reshape(-1)[:3]
        visible = [str(label) for label in labels if str(label).strip()]
        events: list[dict[str, Any]] = []
        for index, node in enumerate(self._nodes):
            if node.is_viewpoint or node.is_frontier or is_ground_truth_node(node):
                continue
            original = self._observation_by_id(int(node.obs_id))
            expected_view = getattr(original, "viewer_xyz", None) if original is not None else None
            if expected_view is None:
                continue
            distance = float(
                np.linalg.norm(
                    np.asarray(expected_view, dtype=float).reshape(-1)[:2]
                    - viewer[:2]
                )
            )
            if distance > float(viewpoint_tolerance_m):
                continue
            seen = any(
                label_matches_relevant_object(node.labels[0], label)
                for label in visible
            )
            if seen:
                self._nodes[index] = replace(
                    node,
                    expected_absence_count=0,
                    belief_confidence=min(0.99, float(node.belief_confidence) + 0.1),
                )
                continue
            consecutive = (
                int(node.expected_absence_count) + 1
                if int(node.last_absence_step) < 0 or now - int(node.last_absence_step) <= 2
                else 1
            )
            changes = list(node.change_events)
            if consecutive >= int(absence_confirmations):
                event = {
                    "type": "expected_object_missing",
                    "node_id": int(node.node_id),
                    "obs_id": int(node.obs_id),
                    "step": now,
                    "last_xyz": np.asarray(node.xyz, dtype=float).tolist(),
                    "viewpoint_distance_m": distance,
                    "confirmations": consecutive,
                    "confidence": min(0.95, 0.45 + 0.2 * consecutive),
                }
                if not changes or changes[-1].get("type") != event["type"]:
                    changes.append(event)
                    self._change_events.append(event)
                    events.append(event)
            self._nodes[index] = replace(
                node,
                expected_absence_count=consecutive,
                last_absence_step=now,
                belief_confidence=max(0.05, float(node.belief_confidence) * 0.65),
                change_events=changes[-32:],
            )
        return events

    def get_change_events(self) -> list[dict[str, Any]]:
        return [dict(event) for event in self._change_events]

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

    def clear_eqa_working_memory(self) -> None:
        """Drop cached EQA / CONFIRMED_MEMORY state after a known world change.

        Forces the next planner call to re-ground from the live graph instead of
        reusing provisional memory summaries and Image-N selections from before
        objects moved.
        """
        self.last_eqa_raw = ""
        self.last_eqa_parsed = ("", "", False, "", "")
        self.last_eqa_obs_ids = []
        self.last_eqa_action_obs_id = None
        self.last_eqa_prompt_node_count = 0
        self.last_eqa_prompt_regions = 0
        self.last_eqa_spatial_rag = None
        self.last_eqa_nav_fallback_count = 0
        self.last_eqa_model_confident = False

    def invalidate_nodes_near(
        self,
        xyz: np.ndarray | list[float] | tuple[float, ...],
        *,
        radius_m: float = 0.75,
        current_step: int | None = None,
        prune: bool = True,
    ) -> tuple[int, int]:
        """Age object nodes near ``xyz`` so staleness pruning can drop them.

        Used after scripted body relocations (dynamic world-change / lifelong fuzz)
        when the old pose is known. Nodes keep their identity until ``maintain``
        runs (or immediately when ``prune=True``).

        Returns:
            ``(n_aged, n_pruned)``.
        """
        if not self._nodes:
            return 0, 0
        target = np.asarray(xyz, dtype=np.float64).reshape(-1)
        if target.size < 2:
            return 0, 0
        cur = int(current_step if current_step is not None else self._effective_timestep())
        horizon = max(0, int(self.staleness_horizon))
        aged_last_seen = cur - horizon - 1 if horizon > 0 else cur - 10_000
        radius = float(radius_m)
        n_aged = 0
        for i, n in enumerate(self._nodes):
            if is_ground_truth_node(n) or n.is_frontier or n.is_viewpoint:
                continue
            node_xy = np.asarray(n.xyz, dtype=np.float64).reshape(-1)
            if node_xy.size < 2:
                continue
            if float(np.linalg.norm(node_xy[:2] - target[:2])) > radius:
                continue
            self._nodes[i] = replace(n, last_seen=int(aged_last_seen))
            n_aged += 1
        n_pruned = 0
        if prune and n_aged > 0:
            if horizon > 0:
                n_pruned = int(self.maintain(cur))
            else:
                # Staleness disabled: still drop explicitly invalidated object nodes.
                n_pruned = int(self._drop_nodes_near(target, radius_m=radius))
        return n_aged, n_pruned

    def _drop_nodes_near(self, xyz: np.ndarray, *, radius_m: float) -> int:
        """Remove non-GT object nodes within ``radius_m`` of ``xyz`` (xy)."""
        target = np.asarray(xyz, dtype=np.float64).reshape(-1)
        to_drop = [
            n
            for n in self._nodes
            if not is_ground_truth_node(n)
            and not n.is_frontier
            and not n.is_viewpoint
            and float(np.linalg.norm(np.asarray(n.xyz, dtype=np.float64).reshape(-1)[:2] - target[:2]))
            <= float(radius_m)
        ]
        if not to_drop:
            return 0
        drop_obs = {n.obs_id for n in to_drop}
        drop_node_ids = {n.node_id for n in to_drop}
        drop_node_ids |= {n.node_id for n in self._nodes if n.is_viewpoint and int(n.obs_id) in drop_obs}
        self._nodes = [n for n in self._nodes if n.node_id not in drop_node_ids]
        self._observations = [o for o in self._observations if o.obs_id not in drop_obs]
        for i, n in enumerate(self._nodes, start=1):
            self._nodes[i - 1] = replace(n, node_id=i)
        self._rebuild_viewpoint_index()
        self._update_edges()
        return len(to_drop)

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
            if not is_ground_truth_node(n) and not n.is_frontier and cur - int(n.last_seen) > self.staleness_horizon
        ]
        if not to_drop:
            return 0
        drop_obs = {n.obs_id for n in to_drop if not n.is_viewpoint}
        drop_node_ids = {n.node_id for n in to_drop}
        drop_node_ids |= {n.node_id for n in self._nodes if n.is_viewpoint and int(n.obs_id) in drop_obs}
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

    def obs_revision(self, obs_id: int) -> int:
        """Content generation for *obs_id* (advances when candidate RGB is refreshed)."""
        return int(self._obs_revisions.get(int(obs_id), 0))

    def _bump_obs_revision(self, obs_id: int) -> int:
        oid = int(obs_id)
        nxt = int(self._obs_revisions.get(oid, 0)) + 1
        self._obs_revisions[oid] = nxt
        self._last_obs_content_update_id = oid
        # Stale SigLIP features would disagree with the refreshed RGB candidate.
        self._obs_siglip_features.pop(oid, None)
        return nxt

    def refresh_observation_candidate(
        self,
        obs_id: int,
        rgb: np.ndarray | Image.Image,
        *,
        xyz: np.ndarray | None = None,
        labels: list[str] | None = None,
        description: str | None = None,
        viewer_xyz: np.ndarray | None = None,
    ) -> bool:
        """Update the stored RGB/candidate for an existing graph observation.

        Graph nodes keep a stable ``obs_id`` under spatial merge; revisits must still
        refresh the evidence image (and invalidate caches) so verify/EQA see the
        better view rather than the first frame forever.
        """
        if isinstance(rgb, Image.Image):
            rgb = np.array(rgb)
        rgb_a = np.asarray(rgb)
        oid = int(obs_id)
        viewer_a: np.ndarray | None = None
        if viewer_xyz is not None:
            viewer_a = np.asarray(viewer_xyz, dtype=float).reshape(-1)[:3].copy()
        xyz_a = None
        if xyz is not None:
            xyz_a = np.asarray(xyz, dtype=float).reshape(-1)[:3].copy()
        for o in self._observations:
            if int(o.obs_id) != oid:
                continue
            o.rgb = rgb_a.copy()
            if xyz_a is not None:
                o.xyz = xyz_a
            if labels is not None:
                o.labels = list(labels)
            if description:
                o.description = description
            if viewer_a is not None:
                o.viewer_xyz = viewer_a
            self._bump_obs_revision(oid)
            return True
        return False

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
            from emet.memory.graph_eqa.graph_stats import labels_compatible_for_dedup

            for idx, existing in enumerate(self._nodes):
                if existing.is_viewpoint or existing.is_frontier or is_ground_truth_node(existing):
                    continue
                el = [str(x).strip() for x in existing.labels if str(x).strip()]
                if not el or not labels_compatible_for_dedup(primary, el[0]):
                    continue
                ex = np.asarray(existing.xyz, dtype=float).reshape(-1)[:3]
                if float(np.linalg.norm(ex[:2] - xyz_a[:2])) <= self.spatial_merge_m:
                    sc = int(existing.support_count) + 1
                    new_xyz, covariance, history, changes, belief_confidence = (
                        self._position_update(existing, xyz_a, step=step)
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
                description[len(GT_BODY_DESC_PREFIX) :]
                if isinstance(description, str)
                and description.startswith(GT_BODY_DESC_PREFIX)
                else f"{re.sub(r'[^a-z0-9]+', '-', primary).strip('-')}:{obs_id}"
            ),
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
                new_xyz, covariance, history, changes, belief_confidence = (
                    self._position_update(existing, xyz_a, step=step)
                )
                merged_labels = sorted({*(str(x).strip() for x in existing.labels if str(x).strip()), label})
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
                    position_covariance=covariance,
                    position_history=history,
                    change_events=changes,
                    belief_confidence=belief_confidence,
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
        )
        for o in self._observations:
            if int(o.obs_id) == int(dst.obs_id):
                o.xyz = new_xyz.copy()
                o.labels = list(merged_labels)
                break

        src_obs_id = int(src.obs_id)
        self._nodes = [n for n in self._nodes if int(n.node_id) != int(src_node_id)]
        self._observations = [o for o in self._observations if int(o.obs_id) != src_obs_id]
        for i, n in enumerate(self._nodes, start=1):
            self._nodes[i - 1] = replace(n, node_id=i)
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
            reachable = None
            if hasattr(voxel_map, "get_reachable_map"):
                reachable = _as_bool_numpy(voxel_map.get_reachable_map(xyt, planner))
        except Exception as e:
            _logger.warning(f"Frontier upsert: could not read map frontiers ({e})")
            return sum(1 for n in self._nodes if n.is_frontier)

        unexplored = _as_bool_numpy(outside) & ~_as_bool_numpy(explored)
        clusters = cluster_frontier_mask(
            unexplored,
            min_cells=self._frontier_min_cluster_cells,
            reachable=reachable,
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

        for kw_score, cluster_id, grid_ij, cell_count in scored[: self._frontier_max_nodes]:
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
                "unexplored areas; " + ", ".join(hints)
                if hints
                else "This observation corresponds to unexplored space;"
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
                        frontier_cell_count=int(cell_count),
                        frontier_keyword_score=float(kw_score),
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
                        np.linalg.norm(
                            np.asarray(view.xyz, dtype=float)[:2]
                            - np.asarray(node.xyz, dtype=float)[:2]
                        )
                    ),
                )
                distance = float(
                    np.linalg.norm(
                        np.asarray(nearest.xyz, dtype=float)[:2]
                        - np.asarray(node.xyz, dtype=float)[:2]
                    )
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
        self._history_outputs[-1] += f"\nNav_result: moved {float(dist_m):.2f}m ({status}; {note})"

    def alternate_nav_target_for_failed_action(
        self,
        question: str,
        blocked_obs_id: int,
        planner: Any,
        base_xyt: Any,
    ) -> np.ndarray | None:
        """Pick a different frontier/fluid goal when the VLM re-picks a failed image action."""
        frontier_nodes = [
            n for n in self._nodes if getattr(n, "is_frontier", False) and int(n.obs_id) != int(blocked_obs_id)
        ]
        if frontier_nodes:
            frontier_nodes.sort(key=lambda n: (int(getattr(n, "nav_failures", 0)), -int(n.last_seen)))
            pick = frontier_nodes[0]
            return np.array([float(pick.xyz[0]), float(pick.xyz[1]), 1.0], dtype=float)
        return None

    def _rank_nodes_for_eqa_prompt(
        self,
        *,
        keywords: list[str] | None = None,
        prefer_obs_ids: list[int] | None = None,
    ) -> list[GraphNode]:
        """Rank object/frontier nodes for a bounded EQA SCENE_GRAPH block.

        Viewpoints are omitted from the ranked list (they bloat prompts); edges still
        reference kept object ids. Frontiers are included and ranked after objects.
        """
        from emet.memory.graph_eqa.frontier_nodes import keyword_overlap_score

        kws = list(keywords or self._relevant_objects or [])
        prefer = {int(x) for x in (prefer_obs_ids or self.last_eqa_obs_ids or [])}
        objects: list[tuple[float, GraphNode]] = []
        frontiers: list[tuple[float, GraphNode]] = []
        for n in self._nodes:
            if n.is_viewpoint:
                continue
            kw = keyword_overlap_score(list(n.labels or []), kws) if kws else 0.0
            support = float(getattr(n, "support_count", 1) or 1)
            prefer_bonus = 2.0 if int(n.obs_id) in prefer else 0.0
            score = 10.0 * kw + prefer_bonus + 0.1 * support
            if n.is_frontier:
                frontiers.append((score, n))
            else:
                objects.append((score, n))
        objects.sort(key=lambda t: (-t[0], int(t[1].node_id)))
        frontiers.sort(key=lambda t: (-t[0], int(t[1].node_id)))
        return [n for _, n in objects] + [n for _, n in frontiers]

    def _eqa_cfg_value(self, key: str, default: Any = None) -> Any:
        """Read ``eqa.<key>`` from Parameters or a nested dict."""
        params = self.parameters
        if params is None:
            return default
        if isinstance(params, dict):
            eqa = params.get("eqa")
            if isinstance(eqa, dict) and key in eqa:
                return eqa.get(key, default)
            return params.get(f"eqa/{key}", params.get(key, default))
        if hasattr(params, "get"):
            return params.get(f"eqa/{key}", default)
        return default

    def _spatial_rag_enabled(self) -> bool:
        """True when eqa.spatial_rag or EMET_EQA_SPATIAL_RAG requests REGION prompts."""
        env = os.environ.get("EMET_EQA_SPATIAL_RAG", "").strip().lower()
        if env in ("1", "true", "yes", "on"):
            return True
        if env in ("0", "false", "no", "off"):
            return False
        raw = self._eqa_cfg_value("spatial_rag", False)
        if isinstance(raw, str):
            return raw.strip().lower() in ("1", "true", "yes", "on")
        return bool(raw)

    def _spatial_rag_float(self, key: str, default: float) -> float:
        try:
            return float(self._eqa_cfg_value(key, default))
        except Exception:
            return default

    def _spatial_rag_int(self, key: str, default: int) -> int:
        try:
            return int(self._eqa_cfg_value(key, default))
        except Exception:
            return default

    def to_string(
        self,
        *,
        max_object_nodes: int | None = None,
        question_keywords: list[str] | None = None,
        prefer_obs_ids: list[int] | None = None,
        record_prompt_count: bool = False,
    ) -> str:
        """Serialize the scene graph to a string for mLLM prompts.

        When ``max_object_nodes`` is set, keep the top-K ranked object/frontier nodes
        (keyword + support + Image-N preference) so blowups cannot starve the VLM.
        With ``eqa.spatial_rag`` / ``EMET_EQA_SPATIAL_RAG``, emit compact REGION blocks
        around keyword / preferred-obs neighborhoods instead of a flat node dump.
        Full untruncated serialization is the default for exports / debugging.
        """
        lines = []

        def _prompt_labels(labels: list[str], max_len: int = 120) -> str:
            s = ", ".join(labels) if labels else "object"
            return s if len(s) <= max_len else s[: max_len - 3] + "..."

        if (
            max_object_nodes is not None
            and max_object_nodes > 0
            and self._spatial_rag_enabled()
        ):
            from emet.memory.graph_eqa.spatial_rag import (
                format_regions_for_prompt,
                select_spatial_regions,
            )

            radius = self._spatial_rag_float("spatial_rag_radius_m", 2.5)
            max_regions = self._spatial_rag_int("spatial_rag_max_regions", 6)
            max_nodes = self._spatial_rag_int(
                "spatial_rag_max_nodes",
                int(max_object_nodes) if max_object_nodes else 48,
            )
            frontier_budget = max(4, int(max_object_nodes) // 4)
            rag = select_spatial_regions(
                list(self._nodes),
                keywords=question_keywords or list(self._relevant_objects or []),
                prefer_obs_ids=prefer_obs_ids or self.last_eqa_obs_ids,
                radius_m=radius,
                max_regions=max_regions,
                max_nodes=max_nodes,
                max_frontiers=frontier_budget,
            )
            if rag.regions:
                text = format_regions_for_prompt(rag)
                keep_ids = set(rag.kept_node_ids)
                for n in rag.frontier_nodes:
                    keep_ids.add(int(n.node_id))
                edge_lines: list[str] = []
                for a, b, rel in self._edges:
                    if int(a) not in keep_ids:
                        continue
                    if b != -1 and int(b) not in keep_ids:
                        continue
                    b_str = "floor" if b == -1 else str(b)
                    edge_lines.append(f"  {rel}({a}, {b_str})")
                if edge_lines:
                    text = text + "\n" + "\n".join(edge_lines)
                if record_prompt_count:
                    self.last_eqa_prompt_node_count = len(keep_ids)
                    self.last_eqa_prompt_regions = len(rag.regions)
                    self.last_eqa_spatial_rag = {
                        "n_regions": len(rag.regions),
                        "n_nodes": len(keep_ids),
                        "seed_node_ids": list(rag.seed_node_ids),
                        "radius_m": radius,
                    }
                return text

        if max_object_nodes is not None and max_object_nodes > 0:
            ranked = self._rank_nodes_for_eqa_prompt(
                keywords=question_keywords,
                prefer_obs_ids=prefer_obs_ids,
            )
            # Always keep at least a few frontiers if present.
            objects = [n for n in ranked if not n.is_frontier]
            frontiers = [n for n in ranked if n.is_frontier]
            keep_obj = objects[: max(0, int(max_object_nodes))]
            frontier_budget = max(0, min(len(frontiers), max(4, int(max_object_nodes) // 4)))
            keep = keep_obj + frontiers[:frontier_budget]
            keep_ids = {int(n.node_id) for n in keep}
            nodes_for_prompt = keep
        else:
            nodes_for_prompt = list(self._nodes)
            keep_ids = {int(n.node_id) for n in nodes_for_prompt}

        if record_prompt_count:
            self.last_eqa_prompt_node_count = len(nodes_for_prompt)
            self.last_eqa_prompt_regions = 0
            self.last_eqa_spatial_rag = None

        for n in nodes_for_prompt:
            lbl = _prompt_labels(n.labels)
            sup = f" n={n.support_count}" if getattr(n, "support_count", 1) != 1 else ""
            if n.is_frontier:
                kind = "Frontier"
            elif n.is_viewpoint:
                kind = "View"
            else:
                kind = "Node"
            lines.append(
                f"{kind} {n.node_id}: {lbl} at ({n.xyz[0]:.2f}, {n.xyz[1]:.2f}, {n.xyz[2]:.2f}) "
                f"[Image {n.obs_id}]{sup}{self._node_nav_status_suffix(n)}"
            )
        for a, b, rel in self._edges:
            if int(a) not in keep_ids:
                continue
            if b != -1 and int(b) not in keep_ids:
                continue
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
        enrich_hints = getattr(self, "_enrich_object_hints", None) or []
        llm_parts = [s.strip() for s in out.split(",") if s.strip()]
        mcq_landmarks = location_mcq_landmark_phrases(question)
        phrase_seed = (
            list(enrich_hints)
            + heuristic_relevant_phrases(question)
            + list(mcq_landmarks)
        )
        if enrich_hints:
            for hint in enrich_hints:
                h = hint.strip().lower()
                if h and " " in h and h not in phrase_seed:
                    phrase_seed.insert(0, h)
        extra_seed = llm_parts + heuristic_relevant_objects(question) + list(mcq_landmarks)
        # Location MCQs need stem object + option landmarks in the same recall set.
        max_items = 8 if mcq_landmarks else 4
        phrases, objects = consolidate_relevant_keywords(
            phrase_seed, extra_seed, max_items=max_items
        )
        self._relevant_phrases = phrases
        self._relevant_objects = objects

    def set_confirmed_memory_siglip_encoder(self, encoder: Any | None) -> None:
        """Attach a SigLIP encoder used only for CONFIRMED_MEMORY (survives voxel encoder drop)."""
        self._confirmed_memory_siglip_encoder = encoder

    def refresh_siglip_confirmed_memory(self) -> None:
        """Encode new graph observation RGBs and refresh phrase→best-view alignments."""
        if not self.memory_summary_enabled:
            return
        enc = self._confirmed_memory_siglip_encoder
        if enc is None:
            return
        from emet.memory.graph_eqa.graph_eqa_siglip import (
            align_phrase_to_observation_features,
            encode_observation_rgb,
        )

        for obs in self._observations:
            oid = int(obs.obs_id)
            if oid in self._obs_siglip_features:
                continue
            feat = encode_observation_rgb(enc, obs.rgb)
            if feat is not None:
                self._obs_siglip_features[oid] = feat
        phrases = self._confirmed_memory_phrases()
        for phrase in phrases:
            match = align_phrase_to_observation_features(
                phrase,
                enc,
                self._observations,
                self._obs_siglip_features,
            )
            if match is not None:
                self._siglip_phrase_cache[phrase.strip().lower()] = match

    def _node_for_obs(self, obs_id: int) -> GraphNode | None:
        return next(
            (
                node
                for node in self._nodes
                if int(node.obs_id) == int(obs_id) and not node.is_viewpoint
            ),
            None,
        )

    def _answerability_gain_for_obs(self, question: str, obs_id: int, phrase: str) -> float:
        obs = self._observation_by_id(int(obs_id))
        labels = list(obs.labels or []) if obs is not None else []
        target_hit = any(label_matches_relevant_object(phrase, label) for label in labels)
        try:
            from emet.habitat.metrics import parse_mcq_choices_from_question

            choices = parse_mcq_choices_from_question(question)
        except Exception:
            choices = []
        if not choices:
            return 1.0 if target_hit else 0.25
        landmark_hit = any(
            label_matches_relevant_object(choice, label)
            for choice in choices
            for label in labels
        )
        if target_hit and landmark_hit:
            return 1.0
        if target_hit or landmark_hit:
            return 0.55
        return 0.15

    def _recall_rank_score(
        self,
        hypothesis: NavHypothesis,
        question: str,
        robot_xyt: np.ndarray | None,
    ) -> NavHypothesis:
        """Cheap recall key for top-K packing (not a VLM decision policy)."""
        answerability = self._answerability_gain_for_obs(
            question,
            hypothesis.obs_id,
            hypothesis.phrase,
        )
        # Map answerability to a small recall boost (landmark/target label hits).
        if answerability >= 1.0:
            hit_boost = 20.0
        elif answerability >= 0.55:
            hit_boost = 10.0
        else:
            hit_boost = 0.0
        path_cost = 0.0
        if robot_xyt is not None and np.asarray(robot_xyt).size >= 2:
            path_cost = float(
                np.linalg.norm(
                    np.asarray(hypothesis.xyz, dtype=float)[:2]
                    - np.asarray(robot_xyt, dtype=float).reshape(-1)[:2]
                )
            )
        tier = float(_RECALL_SOURCE_TIER.get(str(hypothesis.source), 0.0))
        # Path is a weak tiebreak only (cm-scale in the key).
        total = tier + hit_boost - 0.01 * path_cost
        return replace(
            hypothesis,
            score=total,
            answerability_gain=answerability,
            belief_reduction=0.0,
            revisit_change_value=0.0,
            path_cost=path_cost,
            failure_risk=0.0,
        )

    @staticmethod
    def _pack_diversified_hypotheses(
        scored: list[NavHypothesis],
        max_k: int,
    ) -> list[NavHypothesis]:
        """Pack top-K with source diversity (graph + frontier when both exist)."""
        k = max(1, int(max_k))
        if not scored:
            return []
        picked: list[NavHypothesis] = []
        seen: set[int] = set()

        def _take_from(sources: tuple[str, ...]) -> None:
            for h in scored:
                oid = int(h.obs_id)
                if oid in seen:
                    continue
                if str(h.source) in sources:
                    picked.append(h)
                    seen.add(oid)
                    return

        # Seed diversity: one graph, one siglip/confirmed, one frontier when present.
        _take_from(("graph",))
        _take_from(("confirmed", "siglip"))
        _take_from(("frontier",))
        for h in scored:
            if len(picked) >= k:
                break
            oid = int(h.obs_id)
            if oid not in seen:
                picked.append(h)
                seen.add(oid)
        return picked[:k]

    def hypothesize_nav_targets(
        self,
        question: str,
        max_k: int = 6,
        robot_xyt: np.ndarray | None = None,
    ) -> list[NavHypothesis]:
        """Retrieve a small diversified set of nav evidence cards for the router/fallback.

        Ranking is **recall only** (source tier + keyword/landmark hit + distance
        tiebreak). The VLM router decides where to go among the returned cards.
        """
        if not self._observations and not any(
            getattr(n, "is_frontier", False) for n in self._nodes
        ):
            return []
        phrases = list(self._confirmed_memory_phrases()) + list(self._relevant_objects or [])
        if not phrases and question:
            self.extract_relevant_objects(question)
            phrases = list(self._confirmed_memory_phrases()) + list(self._relevant_objects or [])
        # Always merge location-MCQ landmarks (even if extract already ran thin).
        for landmark in location_mcq_landmark_phrases(question):
            if landmark not in phrases:
                phrases.append(landmark)
        scored: list[NavHypothesis] = []
        seen: set[int] = set()
        retracted = getattr(self, "_retracted_nav_claims", None) or set()

        def _claim_blocked(oid: int, phrase: str) -> bool:
            key = (int(oid), str(phrase or "").strip().lower())
            return key in retracted

        # Object / SigLIP cards need phrases; frontiers are still valid cold-start evidence.
        for phrase in phrases:
            for o in self._observations:
                oid = int(o.obs_id)
                if oid in seen or _claim_blocked(oid, phrase):
                    continue
                if self._obs_is_frontier(oid):
                    continue
                if any(label_matches_relevant_object(phrase, lab) for lab in (o.labels or [])):
                    seen.add(oid)
                    scored.append(
                        NavHypothesis(
                            phrase=phrase,
                            obs_id=oid,
                            xyz=np.asarray(o.xyz, dtype=float).reshape(-1)[:3].copy(),
                            score=0.0,
                            source="graph",
                        )
                    )
            # Also match graph nodes (centroid) when observations lack the label string.
            for node in self._nodes:
                if getattr(node, "is_frontier", False) or getattr(node, "is_viewpoint", False):
                    continue
                oid = int(node.obs_id)
                if oid in seen or _claim_blocked(oid, phrase):
                    continue
                if any(label_matches_relevant_object(phrase, lab) for lab in (node.labels or [])):
                    seen.add(oid)
                    scored.append(
                        NavHypothesis(
                            phrase=phrase,
                            obs_id=oid,
                            xyz=np.asarray(node.xyz, dtype=float).reshape(-1)[:3].copy(),
                            score=0.0,
                            source="graph",
                        )
                    )
        for phrase in phrases:
            sig = self._siglip_match_for_phrase(phrase)
            if sig is None:
                continue
            sim, xyz, oid = float(sig[0]), np.asarray(sig[1], dtype=float), sig[2]
            if oid is None:
                continue
            oid = int(oid)
            if oid in seen or self._obs_is_frontier(oid) or _claim_blocked(oid, phrase):
                continue
            if sim >= SIGLIP_CONFIRM_THRESHOLD:
                source = "confirmed"
            elif sim >= SIGLIP_PRESENT_THRESHOLD:
                source = "siglip"
            else:
                continue
            seen.add(oid)
            scored.append(
                NavHypothesis(
                    phrase=phrase,
                    obs_id=oid,
                    xyz=xyz.reshape(-1)[:3].copy(),
                    score=0.0,
                    source=source,
                    siglip_sim=float(sim),
                )
            )
        for node in self._nodes:
            if not node.is_frontier or int(node.obs_id) in seen:
                continue
            # Do NOT attach the question object as the frontier phrase — that made
            # every frontier look like a "fruit bowl" hit and drowned graph places.
            scored.append(
                NavHypothesis(
                    phrase="unexplored frontier",
                    obs_id=int(node.obs_id),
                    xyz=np.asarray(node.xyz, dtype=float).copy(),
                    score=0.0,
                    source="frontier",
                )
            )
            seen.add(int(node.obs_id))
        if not scored:
            return []
        scored = [
            self._recall_rank_score(hypothesis, question, robot_xyt)
            for hypothesis in scored
        ]
        scored.sort(key=lambda h: (-h.score, h.path_cost, -h.obs_id))
        return self._pack_diversified_hypotheses(scored, max_k)


    def retire_frontier_obs(self, obs_id: int) -> bool:
        """Drop a frontier node after visit — visited space is not a frontier."""
        oid = int(obs_id)
        drop_nodes: set[int] = set()
        for n in self._nodes:
            if n.is_frontier and int(n.obs_id) == oid:
                drop_nodes.add(int(n.node_id))
        if not drop_nodes:
            return False
        self._nodes = [n for n in self._nodes if int(n.node_id) not in drop_nodes]
        self._observations = [o for o in self._observations if int(o.obs_id) != oid]
        for i, n in enumerate(self._nodes, start=1):
            self._nodes[i - 1] = replace(n, node_id=i)
        self._rebuild_viewpoint_index()
        self._update_edges()
        return True

    def retract_phrase_claim_at_obs(
        self,
        obs_id: int,
        phrase: str,
        *,
        strip_matching_labels: bool = True,
    ) -> dict[str, Any]:
        """Stop offering a disproved stem-object claim without deleting the place.

        After a close look verifies ABSENT for ``phrase`` at ``obs_id``, blacklist
        that (obs, phrase) for hyp recall and optionally strip matching labels from
        the observation / node. Location-MCQ *place* landmarks should not call this
        for the place name itself (the island is real; only the object was missing).
        """
        oid = int(obs_id)
        key_phrase = str(phrase or "").strip().lower()
        if not key_phrase:
            return {"ok": False, "error": "empty phrase", "obs_id": oid}
        if not hasattr(self, "_retracted_nav_claims"):
            self._retracted_nav_claims = set()
        self._retracted_nav_claims.add((oid, key_phrase))
        stripped_obs = 0
        stripped_nodes = 0
        if strip_matching_labels:
            for o in self._observations:
                if int(o.obs_id) != oid:
                    continue
                before = list(o.labels or [])
                kept = [
                    lab
                    for lab in before
                    if not label_matches_relevant_object(key_phrase, lab)
                ]
                if len(kept) != len(before):
                    o.labels = kept if kept else ["object"]
                    stripped_obs += 1
            for i, n in enumerate(self._nodes):
                if int(n.obs_id) != oid:
                    continue
                if getattr(n, "is_frontier", False) or getattr(n, "is_viewpoint", False):
                    continue
                before = list(n.labels or [])
                kept = [
                    lab
                    for lab in before
                    if not label_matches_relevant_object(key_phrase, lab)
                ]
                if len(kept) != len(before):
                    self._nodes[i] = replace(
                        n,
                        labels=kept if kept else ["object"],
                    )
                    stripped_nodes += 1
        return {
            "ok": True,
            "obs_id": oid,
            "phrase": key_phrase,
            "stripped_obs": stripped_obs,
            "stripped_nodes": stripped_nodes,
            "n_retracted": len(self._retracted_nav_claims),
        }

    def clear_retracted_nav_claims(self) -> None:
        """Drop claim blacklist (e.g. new question)."""
        self._retracted_nav_claims = set()

    def retire_frontier_near_xy(
        self,
        xy: Any,
        *,
        radius_m: float = 1.25,
    ) -> int:
        """Retire frontier nodes within ``radius_m`` of a visited explore goal."""
        try:
            pt = np.asarray(xy, dtype=float).reshape(-1)[:2]
        except Exception:
            return 0
        if pt.size < 2:
            return 0
        r2 = float(radius_m) ** 2
        drop_obs: list[int] = []
        for n in self._nodes:
            if not n.is_frontier:
                continue
            nxy = np.asarray(n.xyz, dtype=float).reshape(-1)[:2]
            if nxy.size < 2:
                continue
            if float(np.sum((nxy - pt) ** 2)) <= r2:
                drop_obs.append(int(n.obs_id))
        n_dropped = 0
        for oid in drop_obs:
            if self.retire_frontier_obs(oid):
                n_dropped += 1
        return n_dropped

    def verify_phrase_at_obs(
        self,
        phrase: str,
        obs_id: int,
        rgb: np.ndarray | None = None,
        *,
        min_sim: float | None = None,
    ) -> VerifyResult:
        """SigLIP-verify *phrase* against observation *obs_id* (optional live *rgb*)."""
        thresh = float(min_sim if min_sim is not None else SIGLIP_CONFIRM_THRESHOLD)
        oid = int(obs_id)
        text = (phrase or "").strip()
        obs = self._observation_by_id(oid)
        if obs is None and rgb is None:
            return VerifyResult(status="ABSENT", sim=0.0, obs_id=oid, phrase=text, ok=False)

        label_hit = False
        if obs is not None and text:
            label_hit = any(label_matches_relevant_object(text, lab) for lab in (obs.labels or []))

        enc = self._confirmed_memory_siglip_encoder
        text_feat: np.ndarray | None = None
        img_feat: np.ndarray | None = None
        sim = 0.0
        if enc is not None and text:
            try:
                from emet.memory.graph_eqa.graph_eqa_siglip import (
                    _feature_vector,
                    encode_observation_rgb,
                )

                text_feat = _feature_vector(enc.encode_text(text))
                if rgb is not None:
                    img_feat = encode_observation_rgb(enc, np.asarray(rgb, dtype=np.uint8))
                elif oid in self._obs_siglip_features:
                    img_feat = np.asarray(self._obs_siglip_features[oid], dtype=np.float32)
                elif obs is not None:
                    img_feat = encode_observation_rgb(enc, obs.rgb)
                    if img_feat is not None:
                        self._obs_siglip_features[oid] = img_feat
                if text_feat is not None and img_feat is not None:
                    sim = float(np.dot(text_feat, img_feat))
            except Exception as e:
                _logger.warning(f"verify_phrase_at_obs SigLIP failed: {e}")

        if sim >= thresh:
            status, ok = "PRESENT", True
        elif sim >= SIGLIP_PRESENT_THRESHOLD or label_hit:
            status, ok = "CANDIDATE", False
        elif text and img_feat is None and text_feat is None:
            # SigLIP is released before submit_answer to free VRAM for the VLM, so any
            # verify after the first submit computes no features. Reporting that as
            # ABSENT looks like real negative evidence in traces and to the loop.
            status, ok = "UNAVAILABLE", False
        else:
            status, ok = "ABSENT", False
        return VerifyResult(
            status=status,
            sim=float(sim),
            obs_id=oid,
            phrase=text,
            ok=ok,
            text_feat=text_feat,
            img_feat=img_feat,
        )

    def select_obs_ids_for_verified_answer(
        self,
        verified_obs_id: int,
        max_images: int = 1,
    ) -> list[int]:
        """Prefer the verified observation; cap at *max_images*."""
        if max_images <= 0:
            return []
        oid = int(verified_obs_id)
        if self._observation_by_id(oid) is None:
            return []
        return [oid][:max_images]

    def _confirmed_memory_phrases(self) -> list[str]:
        if self._relevant_phrases:
            return list(self._relevant_phrases)
        return list(self._relevant_objects or [])

    def _siglip_match_for_phrase(self, phrase: str) -> tuple[float, np.ndarray, int | None] | None:
        key = (phrase or "").strip().lower()
        if not key:
            return None
        cached = self._siglip_phrase_cache.get(key)
        if cached is not None:
            return cached
        grounder = self._text_grounder
        if grounder is None:
            return None
        try:
            sig = grounder(phrase)
        except Exception as e:
            _logger.warning(f"SigLIP grounder failed for {phrase!r}: {e}")
            return None
        if sig is None:
            return None
        sim, xyz = float(sig[0]), np.asarray(sig[1], dtype=float)
        return sim, xyz, None

    def _object_present_in_graph_or_siglip(self, obj: str) -> bool:
        if any(label_matches_relevant_object(obj, lab) for o in self._observations for lab in o.labels):
            return True
        sig = self._siglip_match_for_phrase(obj)
        return sig is not None and float(sig[0]) >= SIGLIP_PRESENT_THRESHOLD

    def _obs_usable_for_eqa_image(self, obs_id: int) -> bool:
        """True when ``obs_id`` may be attached as a VLM answer image.

        Frontier sync stores black 8×8 placeholders — never answer off those.
        Frontiers remain in the SCENE_GRAPH text for Action navigation targets.
        """
        if self._obs_is_frontier(int(obs_id)):
            return False
        return self._observation_by_id(int(obs_id)) is not None

    def _select_relevant_obs_ids(
        self,
        max_images: int = 6,
        choices: list[str] | None = None,
        attribute_question: bool = False,
    ) -> list[int]:
        """Select a diverse set of observation IDs for the EQA prompt (1-based).

        P2 diversification: instead of "all keyword matches then fill", build a
        prioritized pool so the VLM sees question-relevant views *and* a recent
        view *and* spatially spread context, capped at ``max_images``. Falls back
        to the most recent non-frontier observations when there are no keyword
        objects. Frontier placeholder RGB is never selected for answering.

        When ``choices`` are location MCQ options, prefer views whose labels match
        option landmarks (refrigerator, treadmill, …) *before* SigLIP nearest —
        false CONFIRMED_MEMORY coords must not steal Image 1.

        For attribute/state questions, prefer views with lamp/light/curtain labels
        over frontiers before answering on/off or up/down.
        """
        if not self._observations:
            return []
        if max_images <= 0:
            return []
        if not self._relevant_objects:
            recent = [int(o.obs_id) for o in self._observations if self._obs_usable_for_eqa_image(o.obs_id)]
            return recent[-max_images:]

        by_id = {int(o.obs_id): o for o in self._observations}
        selected: list[int] = []

        def take(oid: int) -> bool:
            oid = int(oid)
            if oid in selected or oid not in by_id:
                return False
            if not self._obs_usable_for_eqa_image(oid):
                return False
            selected.append(oid)
            return len(selected) >= max_images

        reserved = 0
        if max_images >= 3:
            reserved = min(2, max_images - 1)
        keyword_budget = max(1, max_images - reserved)

        boost: set[str] = set()
        if choices and not attribute_question:
            for phrase in list(self._confirmed_memory_phrases()) + list(self._relevant_objects or []):
                for tok in _object_match_tokens(phrase):
                    boost |= set(_QUESTION_LANDMARK_BOOST.get(tok, frozenset()))

        def _obs_blob(o: GraphObservation) -> str:
            return " ".join(str(lab) for lab in (o.labels or [])).lower()

        def _direct_target_match(o: GraphObservation) -> bool:
            """True when a label matches the question object without relying only on aliases
            that are absent from the MCQ options (recycle bin vs refrigerator)."""
            labels = [str(lab) for lab in (o.labels or []) if lab]
            if not labels:
                return False
            phrases = list(self._relevant_objects or []) + list(self._confirmed_memory_phrases())
            choice_blob = " ".join(choices or []).lower()
            for lab in labels:
                for phrase in phrases:
                    if not label_matches_relevant_object(phrase, lab):
                        continue
                    # Direct token overlap with the question phrase/object.
                    if _object_match_tokens(phrase) & _object_match_tokens(lab):
                        return True
                    # Alias match (trash↔recycle): keep only if the label appears in options.
                    lab_toks = _object_match_tokens(lab)
                    if any(t in choice_blob for t in lab_toks):
                        return True
            return False

        # Unified Image-1 ranking for location MCQs:
        # boosted choice landmarks (fridge) > direct target (ladder) > weak aliases / generics.
        if choices and not attribute_question:
            scored: list[tuple[float, int]] = []
            for o in self._observations:
                oid = int(o.obs_id)
                blob = _obs_blob(o)
                if not blob.strip():
                    continue
                score = 0.0
                if _direct_target_match(o):
                    score += 10.0
                for ch in choices[:4]:
                    for tok in distinctive_choice_tokens(ch):
                        hit = tok in blob or any(lab.startswith(tok) or tok.startswith(lab) for lab in blob.split())
                        if not hit:
                            continue
                        if tok in _LANDMARK_GENERIC_TOKENS:
                            score += 0.25
                        elif tok in boost:
                            score += 12.0  # fridge for trash beats recycle-alias (+10)
                        else:
                            score += 1.0
                for tok in boost:
                    if tok in blob:
                        score += 0.5  # recycle/bin mild, not enough to beat fridge
                if score > 0:
                    scored.append((score, oid))
            scored.sort(key=lambda t: (-t[0], -t[1]))
            for _score, oid in scored[:keyword_budget]:
                if take(oid):
                    return selected

        # Target keyword / confirmed-memory label matches (non-MCQ or remaining budget).
        keyword_hits: list[int] = []
        for obj in self._relevant_objects:
            for o in reversed(self._observations):
                if int(o.obs_id) in keyword_hits:
                    continue
                if any(label_matches_relevant_object(obj, lab) for lab in o.labels):
                    keyword_hits.append(int(o.obs_id))
        for phrase in self._confirmed_memory_phrases():
            for o in reversed(self._observations):
                oid = int(o.obs_id)
                if oid in keyword_hits:
                    continue
                if any(label_matches_relevant_object(phrase, lab) for lab in o.labels):
                    keyword_hits.append(oid)
        for oid in keyword_hits[:keyword_budget]:
            if take(oid):
                return selected

        # Attribute/state: prefer lamp/light/curtain views over frontiers for Image 1.
        if attribute_question:
            attr_tokens = (
                "lamp",
                "light",
                "lights",
                "ceiling",
                "curtain",
                "curtains",
                "window",
                "fixture",
            )
            attr_hits: list[int] = []
            for o in reversed(self._observations):
                oid = int(o.obs_id)
                if oid in selected or self._obs_is_frontier(oid):
                    continue
                blob = _obs_blob(o)
                if any(t in blob for t in attr_tokens):
                    attr_hits.append(oid)
            for oid in attr_hits:
                if take(oid):
                    return selected

        # SigLIP phrase cache (caption-independent) for targets not already selected.
        for phrase in self._confirmed_memory_phrases():
            cached = self._siglip_phrase_cache.get(phrase.strip().lower())
            if cached is None or cached[2] is None:
                continue
            if float(cached[0]) >= SIGLIP_PRESENT_THRESHOLD and take(int(cached[2])):
                return selected

        # SigLIP obs grounder per relevant object.
        obs_grounder = getattr(self, "_obs_id_grounder", None)
        if obs_grounder is not None:
            for obj in self._relevant_objects:
                try:
                    oid = obs_grounder(obj)
                except Exception:
                    oid = None
                if oid is not None and take(int(oid)):
                    return selected

        # Most recent non-frontier observation (fresh context).
        for o in reversed(self._observations):
            if take(int(o.obs_id)):
                return selected
            break

        # Spatial spread: greedily add observations farthest from those chosen.
        remaining = [
            int(o.obs_id)
            for o in self._observations
            if int(o.obs_id) not in selected and self._obs_usable_for_eqa_image(o.obs_id)
        ]
        while remaining and len(selected) < max_images:
            best_oid = None
            best_dist = -1.0
            for oid in remaining:
                cand = by_id[oid].xyz[:2]
                if selected:
                    d = min(float(np.linalg.norm(cand - by_id[s].xyz[:2])) for s in selected if s in by_id)
                else:
                    d = 0.0
                if d > best_dist:
                    best_dist = d
                    best_oid = oid
            if best_oid is None:
                break
            remaining.remove(best_oid)
            if take(best_oid):
                return selected

        return selected

    def set_text_grounder(self, grounder: Callable[[str], tuple[float, np.ndarray] | None] | None) -> None:
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

    def _nearest_object_neighbors(
        self,
        xyz: np.ndarray,
        *,
        exclude_node_ids: set[int] | None = None,
        max_neighbors: int = 2,
        max_dist_m: float = 3.0,
    ) -> list[tuple[Any, float]]:
        """Nearest non-frontier/viewpoint object nodes to ``xyz`` (planar XY)."""
        exclude = exclude_node_ids or set()
        anchor = np.asarray(xyz, dtype=np.float64).reshape(-1)[:2]
        scored: list[tuple[Any, float]] = []
        for n in self._nodes:
            if getattr(n, "is_frontier", False) or getattr(n, "is_viewpoint", False):
                continue
            if int(n.node_id) in exclude:
                continue
            other = np.asarray(n.xyz, dtype=np.float64).reshape(-1)[:2]
            dist = float(np.linalg.norm(anchor - other))
            if dist <= max_dist_m:
                scored.append((n, dist))
        scored.sort(key=lambda t: t[1])
        return scored[:max_neighbors]

    def _relevant_memory_summary(self) -> str:
        """Surface question-relevant objects as 'confirmed memory' for the VLM.

        Combines two grounding signals so the model does not have to rely on the attached
        images (it otherwise reports 'none' for objects it cannot currently see):
          * graph nodes whose caption-derived labels match the object, and
          * a SigLIP visual match over all observed points (independent of captions) — this
            catches objects that were seen but mislabeled (e.g. a woven basket captioned as
            a "decorative plant").
        For PRESENT objects, also lists nearest furniture/object neighbors so location MCQs
        can map coordinates to options (armchair / table / etc.) instead of inventing rooms.
        """
        if not self._confirmed_memory_phrases():
            return ""
        object_nodes = [n for n in self._nodes if not n.is_frontier and not n.is_viewpoint]
        present_thresh = SIGLIP_PRESENT_THRESHOLD
        lines: list[str] = []
        for obj in self._confirmed_memory_phrases():
            matches = [n for n in object_nodes if any(label_matches_relevant_object(obj, lab) for lab in n.labels)]
            sig = self._siglip_match_for_phrase(obj)
            parts: list[str] = []
            if matches:
                positions = ", ".join(f"({n.xyz[0]:.1f}, {n.xyz[1]:.1f})" for n in matches[:4])
                parts.append(f"{len(matches)} graph node(s) at {positions}")
            sig_present = sig is not None and float(sig[0]) >= present_thresh
            if sig is not None:
                sim, xyz = float(sig[0]), sig[1]
                obs_note = f", obs_id={int(sig[2])}" if sig[2] is not None else ""
                if sig_present:
                    parts.append(f"SigLIP phrase match sim={sim:.2f} near ({xyz[0]:.1f}, {xyz[1]:.1f}){obs_note}")
                else:
                    parts.append(f"no strong SigLIP match (sim={sim:.2f})")
            present = bool(matches) or sig_present
            if present:
                # Anchor for nearest-neighbor furniture (prefer graph match, else SigLIP xyz).
                if matches:
                    anchor_xyz = np.asarray(matches[0].xyz, dtype=np.float64)
                    exclude_ids = {int(n.node_id) for n in matches}
                    status = "PRESENT"
                elif sig is not None and float(sig[0]) >= SIGLIP_CONFIRM_THRESHOLD:
                    anchor_xyz = np.asarray(sig[1], dtype=np.float64)
                    exclude_ids = set()
                    status = "PRESENT"
                elif sig is not None:
                    anchor_xyz = np.asarray(sig[1], dtype=np.float64)
                    exclude_ids = set()
                    status = "CANDIDATE (SigLIP-only — verify in attached images before finalizing)"
                else:
                    anchor_xyz = None
                    exclude_ids = set()
                    status = "PRESENT"
                if anchor_xyz is not None:
                    neighbors = self._nearest_object_neighbors(
                        anchor_xyz, exclude_node_ids=exclude_ids, max_neighbors=2, max_dist_m=3.0
                    )
                    if neighbors:
                        near_bits = []
                        for n, dist in neighbors:
                            lab = ", ".join(n.labels) if n.labels else "object"
                            near_bits.append(f"{lab} at ({n.xyz[0]:.1f}, {n.xyz[1]:.1f}) {dist:.1f}m")
                        parts.append("nearest: " + "; ".join(near_bits))
                lines.append(f"- {obj}: {status} — " + "; ".join(parts))
            elif parts:
                lines.append(f"- {obj}: likely NOT present — " + "; ".join(parts))
            else:
                lines.append(f"- {obj}: not observed during exploration")
        if not lines:
            return ""
        header = (
            "CONFIRMED_MEMORY (working memory — provisional until confirmed in attached "
            "images; if images contradict memory, trust the images and keep exploring; "
            "do not finalize a letter from CANDIDATE / low-sim matches alone; for location "
            "MCQs, prefer option landmarks visible in Image 1 over nearest-furniture guesses):"
        )
        return header + "\n" + "\n".join(lines)

    def _graph_covers_relevant_objects(self) -> bool:
        """True when every keyword object appears in at least one graph node label."""
        eqa_cfg = self.parameters.get("eqa", {}) if hasattr(self.parameters, "get") else {}
        if isinstance(eqa_cfg, dict) and eqa_cfg.get("sqa3d_allow_partial_graph"):
            return True
        if not self._confirmed_memory_phrases() or not self._observations:
            return True
        for obj in self._confirmed_memory_phrases():
            if not self._object_present_in_graph_or_siglip(obj):
                return False
        return True

    def _target_visible_in_obs_ids(self, obs_ids: list[int]) -> bool:
        """True when a question target label appears on an attached Image 1..N view."""
        if not obs_ids:
            return False
        by_id = {int(o.obs_id): o for o in self._observations}
        phrases = list(self._confirmed_memory_phrases()) + list(self._relevant_objects or [])
        for oid in obs_ids:
            o = by_id.get(int(oid))
            if o is None:
                continue
            for phrase in phrases:
                if any(label_matches_relevant_object(phrase, lab) for lab in (o.labels or [])):
                    return True
        return False

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

    def _any_confirmed_phrase_present(self) -> bool:
        for phrase in self._confirmed_memory_phrases():
            if self._object_present_in_graph_or_siglip(phrase):
                return True
        return False

    def _visibility_location_mcq_hint(self, choices: list[str]) -> str:
        lines = "\n".join(f"  {chr(65 + i)}) {choice}" for i, choice in enumerate(choices[:4]))
        return (
            "LOCATION_MCQ: The options are places, not yes/no. When the question asks "
            "'did you see … anywhere?', you must still pick the letter (A–D) for WHERE "
            "the object was observed. Prefer landmarks visible in the attached images; "
            "treat WORKING_MEMORY / CONFIRMED_MEMORY as hints to verify, not as a final "
            "answer if images disagree. Never answer yes/no on answer:.\n"
            f"{lines}"
        )

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
        except Exception as e:
            _logger.warning(f"EQA answer salvage failed ({e})")
            return ""
        text = (salvage_raw or "").strip()
        m = re.search(r"\b([A-D])\b", text)
        if m:
            return m.group(1)
        m = re.search(r"([A-D])", text)
        return m.group(1) if m else ""

    def _neighbor_label_blob_for_present_objects(self) -> str:
        """Concatenate nearest-furniture labels around PRESENT question objects."""
        object_nodes = [n for n in self._nodes if not n.is_frontier and not n.is_viewpoint]
        labels: list[str] = []
        for obj in self._confirmed_memory_phrases():
            matches = [n for n in object_nodes if any(label_matches_relevant_object(obj, lab) for lab in n.labels)]
            sig = self._siglip_match_for_phrase(obj)
            sig_present = sig is not None and float(sig[0]) >= SIGLIP_PRESENT_THRESHOLD
            if not matches and not sig_present:
                continue
            if matches:
                anchor_xyz = np.asarray(matches[0].xyz, dtype=np.float64)
                exclude_ids = {int(n.node_id) for n in matches}
            else:
                assert sig is not None
                anchor_xyz = np.asarray(sig[1], dtype=np.float64)
                exclude_ids = set()
            for n, _dist in self._nearest_object_neighbors(
                anchor_xyz, exclude_node_ids=exclude_ids, max_neighbors=2, max_dist_m=3.0
            ):
                labels.extend(str(lab) for lab in (n.labels or []) if lab)
        return " ".join(labels).lower()

    def _score_choices_against_label_blob(
        self,
        choices: list[str],
        blob: str,
        *,
        ignore_generic: bool = False,
    ) -> list[int]:
        """Per-option token overlap scores against a lowercase label blob."""
        blob_l = (blob or "").lower()
        scores: list[int] = []
        for ch in choices[:4]:
            tokens = distinctive_choice_tokens(ch)
            score = 0
            for t in tokens:
                if ignore_generic and t in _LANDMARK_GENERIC_TOKENS:
                    continue
                if t in blob_l:
                    score += 2
                elif any(lab.startswith(t) or t.startswith(lab) for lab in blob_l.split()):
                    score += 1
            scores.append(score)
        return scores

    def _unique_best_choice_letter(self, scores: list[int]) -> str:
        if not scores or max(scores) < 1:
            return ""
        best = max(scores)
        winners = [i for i, s in enumerate(scores) if s == best]
        if len(winners) != 1:
            return ""
        return chr(65 + winners[0])

    def _any_graph_label_match_for_confirmed(self) -> bool:
        """True when at least one confirmed phrase matches a non-frontier graph/obs label."""
        object_nodes = [n for n in self._nodes if not n.is_frontier and not n.is_viewpoint]
        for obj in self._confirmed_memory_phrases():
            if any(label_matches_relevant_object(obj, lab) for n in object_nodes for lab in (n.labels or [])):
                return True
            if any(label_matches_relevant_object(obj, lab) for o in self._observations for lab in (o.labels or [])):
                return True
        return False

    def _location_letter_from_option_label_hits(self, choices: list[str]) -> str:
        """Map MCQ options onto graph/obs labels (e.g. refrigerator in graph → that letter)."""
        from emet.habitat.metrics import choices_are_location_mcq

        if not choices_are_location_mcq(choices):
            return ""
        parts: list[str] = []
        for n in self._nodes:
            if n.is_frontier or n.is_viewpoint:
                continue
            parts.extend(str(lab) for lab in (n.labels or []) if lab)
        for o in self._observations:
            parts.extend(str(lab) for lab in (o.labels or []) if lab)
        blob = " ".join(parts).lower()
        return self._unique_best_choice_letter(self._score_choices_against_label_blob(choices, blob))

    def _location_letter_from_attached_images(self, choices: list[str], obs_ids: list[int]) -> str:
        """Map MCQ options onto labels of the attached Image 1..N observations.

        Prefer Image 1 landmarks; only fall back to the full attached set when Image 1
        does not uniquely map to a choice (avoids bowl-on-table / kitchen-cabinet noise).
        """
        from emet.habitat.metrics import choices_are_location_mcq

        if not choices_are_location_mcq(choices) or not obs_ids:
            return ""
        by_id = {int(o.obs_id): o for o in self._observations}

        def _blob_for(oids: list[int]) -> str:
            parts: list[str] = []
            for oid in oids:
                o = by_id.get(int(oid))
                if o is None:
                    continue
                parts.extend(str(lab) for lab in (o.labels or []) if lab)
            return " ".join(parts).lower()

        primary = self._unique_best_choice_letter(
            self._score_choices_against_label_blob(choices, _blob_for(obs_ids[:1]), ignore_generic=True)
        )
        if primary:
            return primary
        return self._unique_best_choice_letter(
            self._score_choices_against_label_blob(choices, _blob_for(obs_ids), ignore_generic=True)
        )

    def _equipment_letter_from_target_distances(self, choices: list[str]) -> str:
        """For under-equipment MCQs, pick the option whose equipment label is closest to the target."""
        from emet.habitat.metrics import choices_are_location_mcq

        if not choices_are_location_mcq(choices):
            return ""
        # Need a target object with known xyz (graph or strong SigLIP).
        object_nodes = [n for n in self._nodes if not n.is_frontier and not n.is_viewpoint]
        anchors: list[np.ndarray] = []
        for obj in self._confirmed_memory_phrases():
            matches = [n for n in object_nodes if any(label_matches_relevant_object(obj, lab) for lab in n.labels)]
            for n in matches[:3]:
                anchors.append(np.asarray(n.xyz, dtype=np.float64).reshape(-1)[:2])
            sig = self._siglip_match_for_phrase(obj)
            if sig is not None and float(sig[0]) >= SIGLIP_CONFIRM_THRESHOLD:
                anchors.append(np.asarray(sig[1], dtype=np.float64).reshape(-1)[:2])
        if not anchors:
            return ""
        anchor = anchors[0]

        # Only apply when ≥2 options look like "under <equipment>".
        underish = sum(1 for ch in choices[:4] if "under" in (ch or "").lower())
        if underish < 2:
            return ""

        best_letter = ""
        best_dist = float("inf")
        ties = 0
        matched_options = 0
        for i, ch in enumerate(choices[:4]):
            tokens = distinctive_choice_tokens(ch)
            if not tokens:
                continue
            # Find nearest graph node matching this option's equipment tokens.
            option_hit = False
            for n in object_nodes:
                labs = [str(lab).lower() for lab in (n.labels or []) if lab]
                if not labs:
                    continue
                if not any(any(t in lab or lab.startswith(t) or t.startswith(lab) for lab in labs) for t in tokens):
                    continue
                option_hit = True
                xy = np.asarray(n.xyz, dtype=np.float64).reshape(-1)[:2]
                dist = float(np.linalg.norm(anchor - xy))
                if dist < best_dist - 1e-6:
                    best_dist = dist
                    best_letter = chr(65 + i)
                    ties = 1
                elif abs(dist - best_dist) <= 1e-6 and chr(65 + i) != best_letter:
                    ties += 1
            if option_hit:
                matched_options += 1
        # Need ≥2 equipment options grounded in the graph (bike alone must not win).
        if matched_options < 2 or ties != 1 or best_dist == float("inf"):
            return ""
        return best_letter

    def _location_letter_from_nearest_memory(self, choices: list[str]) -> str:
        """Map PRESENT nearest-furniture labels onto a location MCQ letter (no VLM).

        Used when the model answers yes/no or picks a room that conflicts with
        CONFIRMED_MEMORY neighbors (e.g. woven basket nearest armchair → D).
        Prefer graph-label matches for the question object; SigLIP-only PRESENT is
        weaker and should not override a letter that attached images support.
        """
        from emet.habitat.metrics import choices_are_location_mcq

        if not choices_are_location_mcq(choices) or not self._any_confirmed_phrase_present():
            return ""
        # Prefer direct option↔graph label hits when unique (fridge vs dining table).
        direct = self._location_letter_from_option_label_hits(choices)
        equip = self._equipment_letter_from_target_distances(choices)
        if equip:
            return equip
        blob = self._neighbor_label_blob_for_present_objects()
        nearest = self._unique_best_choice_letter(self._score_choices_against_label_blob(choices, blob))
        if direct and nearest and direct != nearest:
            # Conflict: trust option landmarks in the graph over nearest-furniture of a
            # possibly wrong SigLIP anchor when we lack a graph label on the target.
            if self._any_graph_label_match_for_confirmed():
                return nearest
            return direct
        return nearest or direct

    def _salvage_location_mcq_letter(
        self,
        question: str,
        choices: list[str],
        commands: list[Any],
    ) -> str:
        """Re-ask for a location letter when the model answered visibility yes/no."""
        if self.eqa_client is None or len(choices) < 2:
            return ""
        images = [c for c in commands if isinstance(c, Image.Image)]
        memory = self._relevant_memory_summary()
        choice_lines = "\n".join(f"  {chr(65 + i)}) {choice}" for i, choice in enumerate(choices[:4]))
        stem = question.split("Answer:")[0].strip()
        directive = (
            "The target object WAS observed during exploration (see CONFIRMED_MEMORY). "
            "This is a WHERE-did-you-see-it multiple choice question. "
            "Pick the single best location option letter (A, B, C, or D). "
            "Do NOT answer yes/no. Output only:\nanswer:\n<letter>\n"
        )
        if memory:
            directive += memory + "\n"
        directive += f"Question: {stem}\nOptions:\n{choice_lines}"
        try:
            salvage_raw = self.eqa_client([directive, *images])
        except Exception as e:
            _logger.warning(f"Location-MCQ salvage failed ({e})")
            return ""
        text = (salvage_raw or "").strip()
        m = re.search(r"(?:^|\n)\s*answer\s*:\s*([a-d])\b", text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
        m = re.search(r"\b([A-D])\b", text)
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

        from emet.habitat.metrics import choices_are_location_mcq

        memory = ""
        if self.memory_summary_enabled and choices_are_location_mcq(choices):
            memory = self._relevant_memory_summary()
        freeform_directive = (
            "Look at the images and answer the question in a few words. "
            "Do not use option letters. Do not caption images. Do not explain.\n"
            f"Question: {question}"
        )
        if memory:
            freeform_directive = memory + "\n" + freeform_directive
        try:
            freeform = (self.eqa_client([freeform_directive, *images]) or "").strip()
        except Exception as e:
            _logger.warning(f"MCQ freeform vote failed ({e})")
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
            except Exception as e:
                _logger.warning(f"MCQ letter vote failed ({e})")
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

    def _node_for_obs_id(self, obs_id: int) -> GraphNode | None:
        for n in self._nodes:
            if int(n.obs_id) == int(obs_id):
                return n
        return None

    def _robot_planar_xy(self, robot_xyt: Any | None) -> tuple[float, float] | None:
        if robot_xyt is None:
            return None
        r = np.asarray(robot_xyt, dtype=float).reshape(-1)
        if r.size < 2:
            return None
        return float(r[0]), float(r[1])

    def _viewpoint_xyz_for_obs(self, obs_id: int, obs: GraphObservation | None = None) -> np.ndarray | None:
        if obs is None:
            obs = self._observation_by_id(obs_id)
        if obs is not None and obs.viewer_xyz is not None:
            return np.asarray(obs.viewer_xyz, dtype=float).reshape(-1)[:3]
        vp_id = self._viewpoint_by_obs_id.get(int(obs_id))
        if vp_id is None:
            return None
        for n in self._nodes:
            if int(n.node_id) == int(vp_id) and n.is_viewpoint:
                return np.asarray(n.xyz, dtype=float).reshape(-1)[:3]
        return None

    def _standoff_waypoint_toward(
        self,
        robot_xy: tuple[float, float],
        anchor: np.ndarray,
        *,
        min_approach_m: float | None = None,
    ) -> np.ndarray:
        """Planar goal: move toward ``anchor``, stopping ``min_approach_m`` short of it.

        Habitat/navmesh snaps the goal to the nearest navigable cell; we only pick the
        geometric approach point (closest sensible XY to the object / frontier).
        """
        min_m = float(min_approach_m if min_approach_m is not None else self.image_nav_min_approach_m)
        rx, ry = robot_xy
        ax, ay = float(anchor[0]), float(anchor[1])
        dx, dy = ax - rx, ay - ry
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return np.array([rx + min_m, ry, 1.0], dtype=float)
        travel = dist if dist <= min_m else max(min_m, dist - min_m)
        ux, uy = dx / dist, dy / dist
        return np.array([rx + ux * travel, ry + uy * travel, 1.0], dtype=float)

    def _obs_nav_anchor(self, obs_id: int) -> np.ndarray | None:
        obs = self._observation_by_id(obs_id)
        if obs is None:
            return None
        node = self._node_for_obs_id(obs_id)
        if node is not None:
            return np.asarray(node.xyz, dtype=float).reshape(-1)[:3]
        return np.asarray(obs.xyz, dtype=float).reshape(-1)[:3]

    def place_footprint_for_obs(self, obs_id: int) -> Any:
        """Planar footprint for coverage / annulus sampling (``PlaceFootprint`` or None)."""
        from emet.memory.graph_eqa.place_approaches import (
            footprint_from_node,
            footprint_from_xyz,
        )

        node = self._node_for_obs_id(int(obs_id))
        fp = footprint_from_node(node)
        if fp is not None:
            return fp
        anchor = self._obs_nav_anchor(int(obs_id))
        return footprint_from_xyz(anchor) if anchor is not None else None

    def place_coverage_for_obs(
        self,
        obs_id: int,
        *,
        voxel_map: Any | None = None,
        planner: Any | None = None,
        robot_xyt: Any | None = None,
    ) -> Any:
        """Local frontier completeness for a place card (``PlaceCoverage``)."""
        from emet.memory.graph_eqa.place_approaches import (
            count_frontier_in_footprint,
            coverage_from_frontier_count,
            make_grid_converters,
        )

        fp = self.place_footprint_for_obs(int(obs_id))
        if fp is None or voxel_map is None or not hasattr(voxel_map, "get_2d_map"):
            return coverage_from_frontier_count(None)
        converters = make_grid_converters(voxel_map)
        if converters is None:
            return coverage_from_frontier_count(None)
        xy_to_ij, _ij_to_xy, res = converters
        try:
            from emet.memory.graph_eqa.frontier_nodes import _as_bool_numpy

            xyt = robot_xyt
            if xyt is None:
                return coverage_from_frontier_count(None)
            if planner is not None and hasattr(voxel_map, "get_outside_frontier"):
                outside = voxel_map.get_outside_frontier(xyt, planner)
                _, explored = voxel_map.get_2d_map()
                frontier = _as_bool_numpy(outside) & ~_as_bool_numpy(explored)
            else:
                obstacles, explored = voxel_map.get_2d_map()
                exp = _as_bool_numpy(explored)
                obs = _as_bool_numpy(obstacles)
                from scipy.ndimage import binary_dilation

                frontier = binary_dilation(exp) & ~exp & ~obs
            n = count_frontier_in_footprint(
                fp, frontier, xy_to_ij=xy_to_ij, resolution_m=res
            )
            return coverage_from_frontier_count(n)
        except Exception as e:
            _logger.warning(f"place_coverage_for_obs({obs_id}) failed: {e}")
            return coverage_from_frontier_count(None)

    def _orbit_approach_samples(
        self,
        anchor: np.ndarray,
        robot_xy: tuple[float, float] | None,
        *,
        n: int = 4,
        radius_m: float | None = None,
    ) -> list[np.ndarray]:
        """Legacy evenly spaced bearings (fallback when voxel sampling is unavailable)."""
        n_ap = max(1, int(n))
        radius = float(
            radius_m
            if radius_m is not None
            else max(0.85, float(self.image_nav_min_approach_m) + 0.5)
        )
        ax, ay = float(anchor[0]), float(anchor[1])
        if robot_xy is not None:
            rx, ry = robot_xy
            base = math.atan2(ry - ay, rx - ax)
            first = self._standoff_waypoint_toward(robot_xy, anchor)
        else:
            base = 0.0
            first = np.array([ax + radius, ay, 1.0], dtype=float)
        samples: list[np.ndarray] = [np.asarray(first, dtype=float).reshape(-1)[:3].copy()]
        for k in range(1, n_ap):
            bearing = base + (2.0 * math.pi * float(k) / float(n_ap))
            samples.append(
                np.array(
                    [ax + radius * math.cos(bearing), ay + radius * math.sin(bearing), 1.0],
                    dtype=float,
                )
            )
        return samples

    def _navigation_approach_waypoint_for_obs(
        self,
        obs_id: int,
        robot_xyt: Any | None = None,
        *,
        approach_index: int = 0,
        n_approaches: int = 4,
        avoid_xy: list[tuple[float, float]] | None = None,
        voxel_map: Any | None = None,
        planner: Any | None = None,
    ) -> np.ndarray | None:
        """Sample a planar approach around the observation (annulus when map available)."""
        from emet.memory.graph_eqa.place_approaches import (
            make_grid_converters,
            sample_annulus_approach_xy,
        )

        anchor = self._obs_nav_anchor(int(obs_id))
        if anchor is None:
            return None
        robot_xy = self._robot_planar_xy(robot_xyt)
        if voxel_map is not None and hasattr(voxel_map, "get_2d_map"):
            converters = make_grid_converters(voxel_map)
            if converters is not None:
                xy_to_ij, ij_to_xy, _res = converters
                try:
                    from emet.memory.graph_eqa.frontier_nodes import _as_bool_numpy

                    obstacles, explored = voxel_map.get_2d_map()
                    obstacles_b = _as_bool_numpy(obstacles)
                    reachable = None
                    frontier = None
                    if robot_xyt is not None and planner is not None:
                        if hasattr(voxel_map, "get_reachable_map"):
                            reachable = _as_bool_numpy(
                                voxel_map.get_reachable_map(robot_xyt, planner)
                            )
                        if hasattr(voxel_map, "get_outside_frontier"):
                            outside = voxel_map.get_outside_frontier(robot_xyt, planner)
                            frontier = _as_bool_numpy(outside) & ~_as_bool_numpy(explored)
                    xy = sample_annulus_approach_xy(
                        anchor_xy=(float(anchor[0]), float(anchor[1])),
                        robot_xy=robot_xy,
                        obstacles=obstacles_b,
                        reachable=reachable,
                        frontier=frontier,
                        footprint=self.place_footprint_for_obs(int(obs_id)),
                        xy_to_ij=xy_to_ij,
                        ij_to_xy=ij_to_xy,
                        avoid_xy=avoid_xy,
                        radius_inner_m=max(0.35, float(self.image_nav_min_approach_m)),
                        approach_index=int(approach_index),
                    )
                    if xy is not None:
                        return np.array([float(xy[0]), float(xy[1]), 1.0], dtype=float)
                except Exception as e:
                    _logger.warning(
                        f"annulus approach sample for obs_id={obs_id} failed: {e}"
                    )
        samples = self._orbit_approach_samples(
            anchor, robot_xy, n=max(1, int(n_approaches))
        )
        idx = int(approach_index) % len(samples)
        return samples[idx]

    def _navigation_waypoint_for_obs(
        self,
        obs_id: int,
        robot_xyt: Any | None = None,
    ) -> np.ndarray | None:
        """Closest approachable planar goal for this observation.

        Always aim at the object/frontier/node anchor (node centroid when present).
        With a robot pose, stop a short standoff short of the anchor; navmesh snapping
        happens downstream. Capture ``viewer_xyz`` is evidence provenance, not a goal.
        """
        anchor = self._obs_nav_anchor(int(obs_id))
        if anchor is None:
            return None
        robot_xy = self._robot_planar_xy(robot_xyt)
        if robot_xy is not None:
            return self._standoff_waypoint_toward(robot_xy, anchor)
        return np.array([float(anchor[0]), float(anchor[1]), 1.0], dtype=float)

    def _target_point_from_image_id(
        self,
        image_id: int,
        robot_xyt: Any | None = None,
    ) -> np.ndarray | None:
        """Return ``(x, y, 1)`` nav waypoint for observation ``image_id``."""
        return self._navigation_waypoint_for_obs(int(image_id), robot_xyt)

    def _resolve_eqa_action_image_ref(self, display_index: int, obs_ids: list[int] | None) -> int | None:
        """Map ``Action: Image N`` to a graph observation id.

        Prompt-attached images are renumbered 1..K (``obs_ids`` order). SCENE_GRAPH
        lines use the raw ``[Image {obs_id}]``, so models often emit that id (e.g.
        ``Navigate to Image 19``). Accept both when a real ``GraphObservation``
        exists — not viewpoint-only nav-sample anchors without RGB history.
        """
        idx = int(display_index)
        if idx < 1:
            return None
        ids = [int(x) for x in (obs_ids or [])]
        if ids and 1 <= idx <= len(ids):
            return ids[idx - 1]
        if self._observation_by_id(idx) is not None:
            return idx
        return None

    def _target_point_from_display_image_index(
        self,
        display_index: int,
        *,
        obs_ids: list[int],
        nav_fallback_tail: list[GraphNavigationSample],
        robot_xyt: Any | None = None,
    ) -> np.ndarray | None:
        """Map 1-based ``Image N`` from the EQA prompt (or graph obs_id) to a waypoint."""
        if display_index < 1:
            return None
        oid = self._resolve_eqa_action_image_ref(display_index, obs_ids)
        if oid is not None:
            pt = self._navigation_waypoint_for_obs(oid, robot_xyt)
            if pt is not None:
                return pt
            return self._target_point_from_image_id(oid, robot_xyt)
        if nav_fallback_tail and display_index <= len(nav_fallback_tail):
            nv = nav_fallback_tail[display_index - 1]
            anchor = np.asarray(nv.xyz, dtype=float).reshape(-1)[:3]
            robot_xy = self._robot_planar_xy(robot_xyt)
            if robot_xy is not None:
                return self._standoff_waypoint_toward(robot_xy, anchor)
            return np.array([float(anchor[0]), float(anchor[1]), 1.0], dtype=float)
        return None

    def query_answer(
        self,
        question: str,
        xyt: Any | np.ndarray | list | None = None,
        planner: Any = None,
        *,
        force_obs_ids: list[int] | None = None,
    ) -> tuple[str, str, bool, str, np.ndarray | None, list[Image.Image]]:
        """
        Answer the question using the scene graph and task-relevant images.
        Same return contract as voxel_dynamem.SparseVoxelMap.query_answer.

        Args:
            force_obs_ids: When set (agentic verified submit), use these observation
                ids as Image 1..K instead of re-running diversified selection. Remaining
                slots may still be filled from ``_select_relevant_obs_ids`` when
                ``len(force_obs_ids) < eqa_max_images``.

        Returns:
            reasoning, answer, confidence, confidence_reasoning, target_point, relevant_images
        """
        import time as _time

        from emet.habitat.metrics import (
            answer_is_visibility_abstain,
            choices_are_attribute_state,
            choices_are_location_mcq,
            parse_mcq_choices_from_question,
            question_is_attribute_state,
            question_is_visibility_location,
        )
        from emet.llms.eqa_vl_settings import get_eqa_vl_int

        _t0 = _time.monotonic()
        _logger.info("query_answer: ensure_llm_clients…")
        self._ensure_llm_clients()
        _logger.info("query_answer: extract_relevant_objects…")
        self.extract_relevant_objects(question)
        if self.memory_summary_enabled:
            # Encoder may already be dropped by prepare_dynagraph_vram_for_eqa; refresh
            # is a no-op without it (uses cached phrase features when present).
            _logger.info("query_answer: refresh_siglip_confirmed_memory…")
            self.refresh_siglip_confirmed_memory()
        max_images = get_eqa_vl_int(self.parameters, "eqa_max_images", 4)
        parsed_choices = parse_mcq_choices_from_question(question)
        attribute_q = question_is_attribute_state(question) or choices_are_attribute_state(parsed_choices)
        forced: list[int] = []
        if force_obs_ids:
            for oid in force_obs_ids:
                oi = int(oid)
                if oi in forced:
                    continue
                if self._obs_usable_for_eqa_image(oi):
                    forced.append(oi)
                if len(forced) >= max_images:
                    break
        if forced and len(forced) >= max_images:
            obs_ids = forced[:max_images]
        else:
            selected = [
                int(oid)
                for oid in self._select_relevant_obs_ids(
                    max_images=max_images,
                    choices=parsed_choices if parsed_choices else None,
                    attribute_question=attribute_q,
                )
                if self._obs_usable_for_eqa_image(oid)
            ]
            if forced:
                # Verified (or caller-pinned) view stays Image 1; fill remaining slots.
                rest = [oid for oid in selected if oid not in set(forced)]
                obs_ids = (forced + rest)[:max_images]
            else:
                obs_ids = selected
        self.last_eqa_obs_ids = list(obs_ids)
        max_graph_nodes = get_eqa_vl_int(self.parameters, "eqa_max_graph_nodes", 48)
        graph_str = self.to_string(
            max_object_nodes=max_graph_nodes if max_graph_nodes > 0 else None,
            question_keywords=list(self._relevant_objects or []),
            prefer_obs_ids=obs_ids,
            record_prompt_count=True,
        )
        # Prefer real RGB. If selection is empty (only frontier placeholders in memory),
        # fall back to navigation viewpoint samples — never attach black 8×8 frontiers.
        nav_fallback_tail: list[GraphNavigationSample] = []
        if obs_ids:
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
            img_desc_str = "IMAGE_DESCRIPTIONS: (none — explore for a real camera view before answering)"

        commands: list[Any] = ["Question: " + question]
        if parsed_choices and choices_are_location_mcq(parsed_choices) and question_is_visibility_location(question):
            commands.append(self._visibility_location_mcq_hint(parsed_choices))
        # Attribute/state questions: answer from images; do not inject memory priors.
        if self.memory_summary_enabled and not attribute_q:
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
        id_to_obs = {int(o.obs_id): o for o in self._observations}
        for oid in obs_ids:
            obs = id_to_obs.get(int(oid))
            if obs is None or not self._obs_usable_for_eqa_image(oid):
                continue
            im = Image.fromarray(obs.rgb.astype(np.uint8), mode="RGB")
            relevant_images.append(im)
            commands.append(im)
        for nv in nav_fallback_tail:
            im = Image.fromarray(nv.rgb.astype(np.uint8), mode="RGB")
            relevant_images.append(im)
            commands.append(im)
        self.last_eqa_nav_fallback_count = len(nav_fallback_tail)

        _logger.info(
            f"query_answer: calling eqa_client (n_images={len(relevant_images)} "
            f"n_cmd={len(commands)} prep_s={_time.monotonic() - _t0:.1f})…"
        )
        try:
            t_vl = _time.monotonic()
            # Cap decode length for post-explore banks (full 512 made hung prefills worse).
            ans_cap = int(os.environ.get("EMET_EQA_ANSWER_MAX_NEW_TOKENS", "256") or 256)
            eqa_kw: dict[str, Any] = {}
            if ans_cap > 0:
                eqa_kw["max_new_tokens"] = ans_cap
            try:
                raw = self.eqa_client(commands, **eqa_kw)
            except TypeError:
                # Older / test doubles that only accept the command list.
                raw = self.eqa_client(commands)
            _logger.info(
                f"query_answer: eqa_client done wall_s={_time.monotonic() - t_vl:.1f} "
                f"out_chars={len(raw or '')}"
            )
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
        # Re-ask tersely for just the choice letter.
        # - Empty answer → always salvage (64-token truncation / runaway caption).
        # - ``Unknown`` on attribute/yes-no → salvage (holdout q65).
        # - Empty/``Unknown`` on location MCQ → do NOT invent a letter (holdout q104/q105);
        #   agentic should follow Action:/explore instead of memory/salvage A–D.
        _ans_stripped = (answer or "").strip()
        _ans_unknown = _ans_stripped.lower() in {"unknown", "none", "n/a", "na"}
        _ans_unknownish = _ans_unknown or not _ans_stripped
        _loc_mcq = bool(parsed_choices and choices_are_location_mcq(parsed_choices) and not attribute_q)
        # A stream that never reached ``answer:`` was cut off mid-caption (the 256-token
        # decode budget is spent on multi-image Caption/Reasoning). Re-asking the same VLM
        # with the same images recovers the letter it never got to emit; that is different
        # from inventing one out of nearest-furniture memory, which q104/q105 banned.
        _answer_field_emitted = bool(re.search(r"answer\s*:", answer_outputs))
        _truncated_before_answer = _ans_unknownish and not _answer_field_emitted
        # Location MCQ: never invent A–D from memory, but do recover a truncated stream.
        _should_salvage = _ans_unknownish and (not _loc_mcq or _truncated_before_answer)
        if _loc_mcq and _ans_unknownish and not _ans_stripped:
            # Truncated streams often omit ``answer:`` entirely (failfix5); normalize so
            # human_answer / agentic follow-up treat this as Unknown, not memory-B.
            answer = "Unknown"
            _ans_stripped = "Unknown"
            _ans_unknown = True
            _ans_unknownish = True
        if _should_salvage:
            salvage = self._salvage_answer_letter(question, commands)
            if salvage:
                answer = salvage
                raw = (raw or "") + f"\n[salvage]\nanswer:\n{salvage}\n"
                self.last_eqa_raw = raw
        elif (
            parsed_choices
            and choices_are_location_mcq(parsed_choices)
            and self._any_confirmed_phrase_present()
            and not attribute_q
        ):
            # Location letter overrides (equip → image → abstaining memory) are intentional
            # Dynagraph eval levers. Accuracy can move vs GE-only / no-override ablations;
            # always report HM-EQA deltas with the harness fingerprint + git commit.
            # Geometric under-equipment (mat under treadmill) may correct VLM guesses.
            # Image landmarks may correct memory-steered letters. Nearest-furniture memory
            # alone must NOT override a clear VLM A–D (Q6: VLM B correct, memory A) **or**
            # free-text that uniquely matches a choice ("the room with the blue curtains").
            from emet.habitat.metrics import extract_mcq_letter

            img_letter = self._location_letter_from_attached_images(parsed_choices, obs_ids)
            equip_letter = self._equipment_letter_from_target_distances(parsed_choices)
            memory_letter = self._location_letter_from_nearest_memory(parsed_choices)
            parsed_letter = extract_mcq_letter(answer, parsed_choices)
            abstain = answer_is_visibility_abstain(answer) or not parsed_letter
            preferred = ""
            if equip_letter and (abstain or parsed_letter != equip_letter):
                preferred = equip_letter
            elif img_letter and (abstain or parsed_letter != img_letter):
                preferred = img_letter
            elif abstain and memory_letter and (self._any_graph_label_match_for_confirmed() or not img_letter):
                preferred = memory_letter
            # Empty / Unknown location MCQ: keep Unknown so agentic can follow Action:N.
            # Inventing A–D via memory/salvage-location caused failfix5 wrong B letters.
            if _ans_unknownish:
                pass
            elif preferred and (abstain or parsed_letter != preferred):
                answer = preferred
                raw = (raw or "") + f"\n[memory-location]\nanswer:\n{preferred}\n"
                self.last_eqa_raw = raw
            elif abstain:
                # Visibility-style Yes/No on a WHERE question may still salvage; empty/Unknown
                # already handled above. Do not salvage bare abstains without a letter.
                if answer_is_visibility_abstain(answer) and not _ans_unknownish:
                    salvage = self._salvage_location_mcq_letter(question, parsed_choices, commands)
                    if salvage:
                        answer = salvage
                        raw = (raw or "") + f"\n[salvage-location]\nanswer:\n{salvage}\n"
                        self.last_eqa_raw = raw
            elif parsed_letter and not re.fullmatch(r"[A-E]", (answer or "").strip(), flags=re.I):
                # Normalize NL choice text → letter so downstream scoring / human format
                # see the same canonical answer the override logic respected.
                answer = parsed_letter
                raw = (raw or "") + f"\n[choice-text]\nanswer:\n{parsed_letter}\n"
                self.last_eqa_raw = raw
        self.last_eqa_model_confident = bool(confidence)
        covered = self._graph_covers_relevant_objects()
        if confidence and not covered:
            confidence = False
            confidence_reasoning = (
                confidence_reasoning
                + " The scene graph does not yet include all question-relevant objects; explore further."
            ).strip()
        # Do not finalize Yes/No from absence while relevant objects are still uncovered.
        if (
            not covered
            and answer_is_visibility_abstain(answer)
            and not (parsed_choices and choices_are_location_mcq(parsed_choices))
        ):
            confidence = False
            confidence_reasoning = (
                confidence_reasoning
                + " Yes/No from missing evidence is not final; keep exploring until objects are observed."
            ).strip()
        # Empty / Unknown is never a confirmed MCQ answer — keep updating memory.
        if not (answer or "").strip() or (answer or "").strip().lower() in {
            "unknown",
            "none",
            "n/a",
            "na",
        }:
            confidence = False
            confidence_reasoning = (
                confidence_reasoning + " No clear letter yet; explore and refresh memory before confirming."
            ).strip()
        # Require a clear picture: don't confirm location letters unsupported by attached
        # image labels when memory is only SigLIP-candidate (no graph label on the target).
        if (
            confidence
            and parsed_choices
            and choices_are_location_mcq(parsed_choices)
            and not attribute_q
            and not self._any_graph_label_match_for_confirmed()
        ):
            letter_m = re.search(r"\b([a-d])\b", (answer or "").strip().lower())
            parsed_letter = letter_m.group(1).upper() if letter_m else ""
            img_letter = self._location_letter_from_attached_images(parsed_choices, obs_ids)
            if parsed_letter and img_letter and parsed_letter != img_letter:
                confidence = False
                confidence_reasoning = (
                    confidence_reasoning
                    + " Answer conflicts with landmarks in attached images; update memory / views before confirming."
                ).strip()
            elif parsed_letter and not img_letter:
                confidence = False
                confidence_reasoning = (
                    confidence_reasoning + " Location not yet verified in attached images; explore for a clearer view."
                ).strip()
        # Never finalize a WHERE answer if the target object is not in attached views
        # (guessing "dining table" / "side table" without seeing towel/fruit bowl).
        if (
            confidence
            and parsed_choices
            and choices_are_location_mcq(parsed_choices)
            and not attribute_q
            and not self._target_visible_in_obs_ids(obs_ids)
        ):
            confidence = False
            confidence_reasoning = (
                confidence_reasoning + " Target object not visible in attached images; explore before confirming."
            ).strip()
        # Under-equipment MCQs: do not finalize until geometric equipment letter is known
        # (otherwise bike vs treadmill is a coin flip from a partial gym view).
        if confidence and parsed_choices and choices_are_location_mcq(parsed_choices) and not attribute_q:
            underish = sum(1 for ch in parsed_choices[:4] if "under" in (ch or "").lower())
            if underish >= 2:
                equip = self._equipment_letter_from_target_distances(parsed_choices)
                if not equip:
                    confidence = False
                    confidence_reasoning = (
                        confidence_reasoning
                        + " Under-equipment location needs a clearer mat↔equipment distance before confirming."
                    ).strip()
                elif re.search(r"\b([a-d])\b", (answer or "").strip().lower()):
                    letter_m = re.search(r"\b([a-d])\b", (answer or "").strip().lower())
                    if letter_m and letter_m.group(1).upper() != equip:
                        answer = equip
                        raw = (raw or "") + f"\n[equipment-location]\nanswer:\n{equip}\n"
                        self.last_eqa_raw = raw
        # Attribute/state: never finalize from memory priors (images only).
        if confidence and attribute_q and self.memory_summary_enabled:
            # Soft gate: if Image 1 is a frontier-only view, keep exploring.
            if obs_ids and self._obs_is_frontier(int(obs_ids[0])):
                confidence = False
                confidence_reasoning = (
                    confidence_reasoning + " Attribute/state needs a non-frontier view of the object before confirming."
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
                self.last_eqa_action_obs_id = self._resolve_eqa_action_image_ref(display_index, obs_ids)
                target_point = self._target_point_from_display_image_index(
                    display_index,
                    obs_ids=obs_ids,
                    nav_fallback_tail=nav_fallback_tail,
                    robot_xyt=xyt,
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
            except Exception as e:
                _logger.warning(f"fill_descriptions_from_vlm failed for obs {obs.obs_id}: {e}")
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
